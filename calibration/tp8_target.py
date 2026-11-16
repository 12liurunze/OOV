"""Read actual Target Verify logits in the separate TP8 collection runtime."""

import functools
import json
import os
from pathlib import Path

from .contracts import prefix_hash


def tree_paths(children, siblings):
    """Map each verified node to its path of node indexes, including itself."""
    n = len(children)
    if not n or len(siblings) != n or siblings[0] != -1:
        raise ValueError("Invalid verification tree root or shape")
    paths = {}

    def visit(node, ancestors):
        while node != -1:
            if not 0 <= node < n or node in paths:
                raise ValueError("Verification tree has a cycle, duplicate parent or invalid link")
            path = (*ancestors, node)
            paths[node] = path
            visit(children[node], path)
            node = siblings[node]

    visit(0, ())
    if len(paths) != n:
        raise ValueError("Verification tree contains unreachable nodes")
    return paths


def exact_committed_prefix(request, seq_len, root_token):
    """Reconstruct the non-overlap KV prefix immediately before the bonus root."""
    if request.is_retracted or request.finished() or request.multimodal_inputs is not None:
        raise RuntimeError("Only a live text request has a supported Verify prefix")
    prompt = list(request.origin_input_ids)
    raw = list(request.output_ids)
    finished_len = getattr(request, "finished_len", None)
    if finished_len is not None:
        if type(finished_len) is not int or not 0 <= finished_len <= len(raw):
            raise ValueError("Invalid finished_len in Target Verify")
        raw = raw[:finished_len]
    committed_count = seq_len - len(prompt)
    if committed_count < 0 or committed_count != len(raw) - 1 or raw[-1] != root_token:
        raise RuntimeError(
            f"Target Verify prefix/root mismatch for {request.rid}: "
            f"kv={seq_len}, prompt={len(prompt)}, output={len(raw)}"
        )
    return prompt + raw[:committed_count]


def validate_mask_row(row, seq_len, paths, node):
    """Check the Triton FULL_MASK row against the reconstructed ancestry."""
    if any(value != 1 for value in row[:seq_len]):
        raise ValueError("Verify mask did not expose the full committed prefix")
    observed = row[seq_len:]
    expected = [int(index in paths[node]) for index in range(len(paths))]
    if observed != expected:
        raise ValueError("Verify mask does not match its tree path")


def verify_rows(verify, batch, logits_output, execution, *, graph_selected, grammar_mask):
    """Read-only collection after Target forward and before its sampler."""
    import torch

    if batch.enable_overlap or graph_selected is not False:
        raise RuntimeError("TP8 calibration requires eager, non-overlap Target Verify")
    sampling = batch.sampling_info
    if not sampling.is_all_greedy or batch.has_grammar or grammar_mask is not None:
        raise RuntimeError("Calibration requires unconstrained greedy Target labels")
    if any(getattr(sampling, key, None) is not None for key in
           ("acc_additive_penalties", "acc_scaling_penalties", "logit_bias")):
        raise RuntimeError("Calibration requires unmodified Target logits")
    requests = batch.reqs
    n, bs = verify.draft_token_num, len(requests)
    if n != 8:
        raise RuntimeError("TP8 calibration requires S7/Top2/N8")
    logits = logits_output.next_token_logits.detach()
    if logits.shape != (bs * n, 128256) or not torch.isfinite(logits).all().item():
        raise RuntimeError("Expected finite, full-vocabulary Target Verify logits on rank 0")
    lengths = batch.seq_lens.detach().cpu().tolist()
    tokens = verify.draft_token.reshape(bs, n).detach().cpu().tolist()
    positions = verify.positions.reshape(bs, n).detach().cpu().tolist()
    children = verify.retrieve_next_token.reshape(bs, n).detach().cpu().tolist()
    siblings = verify.retrieve_next_sibling.reshape(bs, n).detach().cpu().tolist()
    indexes = verify.retrieve_index.reshape(bs, n).detach().cpu().tolist()
    gold = logits.argmax(dim=-1).reshape(bs, n).cpu().tolist()

    mask = verify.custom_mask
    mask_count = sum(length + n for length in lengths) * n
    if mask is None or mask.dtype not in (torch.bool, torch.uint8) or mask.numel() < mask_count:
        raise ValueError("Expected readable Triton FULL_MASK for all live Verify rows")
    mask = mask.flatten()[:mask_count].detach().cpu().tolist()
    if len(lengths) != bs or len({str(req.rid) for req in requests}) != bs:
        raise ValueError("Target Verify batch lengths or request identities do not match")
    offset = 0
    for row, req in enumerate(requests):
        seq_len = lengths[row]
        visible = exact_committed_prefix(req, seq_len, tokens[row][0])
        visible_sha = prefix_hash(visible)
        paths = tree_paths(children[row], siblings[row])
        span = seq_len + n
        for node in range(n):
            parent = visible + [tokens[row][j] for j in paths[node]]
            if indexes[row][node] != row * n + node or positions[row][node] != len(parent) - 1:
                raise ValueError("Target Verify row index or position does not match its path")
            begin = offset + node * span
            validate_mask_row(mask[begin:begin + span], seq_len, paths, node)
            yield {
                "request_id": str(req.rid),
                "round_id": f"{visible_sha}-{tokens[row][0]}",
                "visible_prefix_sha256": visible_sha,
                "root_token": tokens[row][0],
                "parent_prefix_ids": parent,
                "target_prefix_sha256": prefix_hash(parent),
                "position": positions[row][node],
                "gold": gold[row][node],
                "target_execution_id": f"verify-{execution}-request-{row}-node-{node}",
                "capture_before_sampling": True,
                "mask_parity": True,
            }
        offset += span * n


def install(module):
    """Wrap the TP8 worker's real Verify preparation and sampling call sites."""
    if os.environ.get("OOV_TP8_COLLECTION") != "1" or getattr(module, "_oov_tp8_target_hook", False):
        return
    output = Path(os.environ["OOV_TP8_TARGET_JSONL"])
    original_prepare = module.eagle_prepare_for_verify
    original_sample = module.eagle_sample
    state = {"next_execution": 0, "graph_selected": None}

    @functools.wraps(original_prepare)
    def prepare(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        state["graph_selected"] = bool(prepared[1])
        return prepared

    @functools.wraps(original_sample)
    def sample(verify, batch, logits_output, *args, **kwargs):
        import torch
        from sglang.srt.distributed import get_tp_group

        group = get_tp_group()
        if group.world_size != 8:
            raise RuntimeError("The calibration Target hook received a non-TP8 worker")
        graph_selected = state["graph_selected"]
        state["graph_selected"] = None
        error = None
        if group.rank_in_group == 0 and not batch.forward_mode.is_idle() and batch.reqs:
            try:
                execution = state["next_execution"]
                state["next_execution"] += 1
                grammar = args[0] if args else kwargs.get("grammar_mask")
                rows = list(verify_rows(verify, batch, logits_output, execution,
                                        graph_selected=graph_selected, grammar_mask=grammar))
                with output.open("a", encoding="utf-8") as stream:
                    for row in rows:
                        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            except Exception as exc:
                error = exc
        # All ranks reach the same collective, including when only rank 0's
        # CPU validation or output file raises. Calibration does not time it.
        failed = torch.tensor([int(error is not None)], device=batch.seq_lens.device,
                              dtype=torch.int32)
        torch.distributed.all_reduce(failed, group=group.device_group)
        if failed.item():
            if error is not None:
                raise error
            raise RuntimeError("Target calibration capture failed on rank 0")
        return original_sample(verify, batch, logits_output, *args, **kwargs)

    module.eagle_prepare_for_verify = prepare
    module.eagle_sample = sample
    module._oov_tp8_target_hook = True
