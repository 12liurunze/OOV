"""Experimental Llama EAGLE3 greedy decode-only TP2 candidate communication.

Explicit packed output contract: [p1,p2,id1,id2,first_argmax_id], FP32.
No target, extend, sampling, KV, tree budget or attention changes.
"""
import json
from pathlib import Path
import torch
import triton
import triton.language as tl

_AUDIT = None


@triton.jit
def _partial(X, S, STRIDE: tl.constexpr, LOCAL: tl.constexpr,
             OFFSET: tl.constexpr, NS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    idx = part * BLOCK + tl.arange(0, BLOCK)
    valid = idx < LOCAL
    x = tl.load(X + row * STRIDE + idx, valid, other=-float('inf')).to(tl.float32)
    maximum = tl.max(x, 0)
    z = tl.sum(tl.exp(x - tl.where(maximum == -float('inf'), 0., maximum)), 0)
    first = tl.min(tl.where(valid & (x == maximum), idx, 2147483647), 0)
    second_value = tl.max(tl.where(idx != first, x, -float('inf')), 0)
    second = tl.min(tl.where(valid & (idx != first) & (x == second_value), idx, 2147483647), 0)
    base = (row * NS + part) * 5
    tl.store(S + base, maximum)
    tl.store(S + base + 1, z)
    tl.store(S + base + 2, second_value)
    tl.store(S + base + 3, first + OFFSET)
    tl.store(S + base + 4, second + OFFSET)


@triton.jit
def _merge(S, O, ROWS: tl.constexpr, NS: tl.constexpr,
           WORLD: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    part = k // 2
    rank = part // NS
    split = part % NS
    valid = k < WORLD * NS * 2
    base = ((rank * ROWS + row) * NS + split) * 5
    value = tl.load(S + base + tl.where(k % 2 == 0, 0, 2), valid, other=-float('inf'))
    ids = tl.load(S + base + tl.where(k % 2 == 0, 3, 4), valid, other=2147483647.).to(tl.int32)
    maximum = tl.max(value, 0)
    z = tl.load(S + base + 1, valid & (k % 2 == 0), other=0.)
    term = tl.where(valid & (k % 2 == 0), z * tl.exp(value - maximum), 0.)
    denom = tl.sum(term, 0)
    first = tl.min(tl.where(valid & (ids < V) & (value == maximum), ids, 2147483647), 0)
    second_value = tl.max(tl.where(valid & (ids != first), value, -float('inf')), 0)
    second = tl.min(tl.where(valid & (ids < V) & (ids != first) & (value == second_value), ids, 2147483647), 0)
    p1, p2 = 1. / denom, tl.exp(second_value - maximum) / denom
    swap = p1 == p2  # Existing fused FP32 Top2's tie convention.
    tl.store(O + row * 5, p1)
    tl.store(O + row * 5 + 1, p2)
    tl.store(O + row * 5 + 2, tl.where(swap, second, first))
    tl.store(O + row * 5 + 3, tl.where(swap, first, second))
    tl.store(O + row * 5 + 4, first)


def compact(logits, lm_head, vocab, group):
    assert group.world_size == 2 and vocab == 32000
    assert logits.ndim == 2 and logits.shape[1] == 16000
    assert logits.dtype in (torch.bfloat16, torch.float32)
    shard = lm_head.shard_indices
    offset = shard.org_vocab_start_index
    assert shard.org_vocab_end_index - offset == 16000
    assert shard.num_elements_padded == 16000
    rows, local = logits.shape
    splits = triton.cdiv(local, 4096)
    summary = torch.empty((rows, splits, 5), device=logits.device, dtype=torch.float32)
    gathered = torch.empty((2 * rows, splits, 5), device=logits.device, dtype=torch.float32)
    result = torch.empty((rows, 5), device=logits.device, dtype=torch.float32)
    _partial[(rows, splits)](logits, summary, logits.stride(0), local, offset, splits, 4096, num_warps=4)
    group.all_gather_into_tensor(gathered, summary)
    _merge[(rows,)](gathered, result, rows, splits, 2, vocab,
                   triton.next_power_of_2(2 * splits * 2), num_warps=4)
    return result


def unpack(packed, topk, positions, draft_tokens, column):
    assert packed.shape[1] == 5 and topk in (1, 2)
    if topk == 1:
        ids = packed[:, 4:5].to(torch.int64)
        probs = torch.ones_like(ids, dtype=torch.float32)
        if draft_tokens is not None:
            draft_tokens[:, column:column + 1].copy_(ids)
    else:
        probs = packed[:, :2].contiguous()
        ids = packed[:, 2:4].to(torch.int64)
    positions.add_(1)
    return probs, ids


def shadow(packed, full_logits):
    global _AUDIT
    from sglang.kernels.ops.speculative.topk1 import draft_topk2_probs
    # Allocated on warmup before replay; updates below are captured GPU ops.
    if _AUDIT is None:
        _AUDIT = torch.zeros(6, device=packed.device, dtype=torch.float64)
    probs, ids = draft_topk2_probs(full_logits.float())
    _AUDIT[0].add_(packed.shape[0])
    _AUDIT[1].add_((packed[:, 2:4].to(torch.int64) != ids).any(1).sum())
    _AUDIT[2].add_(((packed[:, 0] < .5) != (probs[:, 0] < .5)).sum())
    error = (packed[:, :2] - probs).abs().max().to(torch.float64)
    _AUDIT[3].copy_(torch.maximum(_AUDIT[3], error))
    _AUDIT[4].add_((packed[:, 4].to(torch.int64) != full_logits.argmax(-1)).sum())
    _AUDIT[5].add_(1)


def write_audit(directory):
    if _AUDIT is None:
        return
    values = _AUDIT.cpu().tolist()
    result = dict(rows=int(values[0]), top2_id_mismatch_rows=int(values[1]),
                  threshold05_mismatch_rows=int(values[2]), max_probability_error=values[3],
                  argmax_mismatch_rows=int(values[4]), calls=int(values[5]),
                  includes_graph_capture_and_warmup=True, usable_as_speed=False)
    result['pass'] = all(values[i] == 0 for i in (1, 2, 4)) and values[3] < 2e-6 and values[0] > 0
    (Path(directory) / 'compact_shadow.json').write_text(json.dumps(result, indent=2))
