"""Exact TP1 repair masks and packet finalization around the existing readout."""
import torch
import triton
import triton.language as tl


@triton.jit
def _invalid_probability(value):
    bits = value.to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    # Exact FP32 [0, 1] test, including signed zero and subnormal values.
    return (magnitude > 0x3F800000) | ((bits != magnitude) & (magnitude != 0))


@triton.jit
def _invalid_packet_value(value):
    bits = value.to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    return (bits == 0x7F800000) | (magnitude > 0x7F800000)


@triton.jit
def _inputs(
    P, IDS, SLOTS, VALID, PREFIX, REAL, OVERFLOW, KNOWN, PROMPT, ERROR,
    SAFE_IDS, SAFE_SLOTS, ACTIVE, AVAILABLE, ENABLED,
    ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
    POOL_ROWS: tl.constexpr, VOCAB: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    live_row = row < ROWS
    req = row // WIDTH
    slot = tl.load(SLOTS + req, live_row, other=0)
    real = tl.load(REAL + req, live_row, other=False)
    valid = tl.load(VALID + row, live_row, other=False)
    valid_index = (slot >= 0) & (slot < POOL_ROWS)
    # Slot zero is reserved for Graph padding. A malformed slot must stay
    # physically safe even though the real request raises at the tree barrier.
    safe_slot = tl.where(valid_index, slot, 0)
    active = real & valid
    disabled = tl.load(OVERFLOW + safe_slot, live_row, other=True)
    committed = tl.load(KNOWN + safe_slot, live_row, other=0)
    prompt = tl.load(PROMPT + safe_slot, live_row, other=0)
    prefix = tl.load(PREFIX + req, live_row, other=0).to(tl.int64)
    available = real & (safe_slot > 0) & ~disabled & (committed <= prefix - prompt)
    tl.store(ACTIVE + row, active, live_row)
    tl.store(ENABLED + row, active & available, live_row)
    first = live_row & (row % WIDTH == 0)
    tl.store(AVAILABLE + req, available, first)
    tl.store(SAFE_SLOTS + req, safe_slot, first)

    id0 = tl.load(IDS + row * 2, live_row, other=0)
    id1 = tl.load(IDS + row * 2 + 1, live_row, other=0)
    p0 = tl.load(P + row * 2, live_row, other=0)
    p1 = tl.load(P + row * 2 + 1, live_row, other=0)
    bad_id = (id0 < 0) | (id0 >= VOCAB) | (id1 < 0) | (id1 >= VOCAB)
    bad_p = _invalid_probability(p0) | _invalid_probability(p1)
    malformed = live_row & ((active & (bad_id | bad_p)) | (real & ~valid_index))
    tl.atomic_add(ERROR, tl.max(malformed.to(tl.int32), 0).to(tl.int64), sem="relaxed")
    tl.store(SAFE_IDS + row * 2, tl.maximum(0, tl.minimum(VOCAB - 1, id0)), live_row)
    tl.store(SAFE_IDS + row * 2 + 1, tl.maximum(0, tl.minimum(VOCAB - 1, id1)), live_row)


@triton.jit
def _finish(
    PACKET, AVAILABLE, REAL, ERROR, AUDIT, READOUT, COLD, USABLE,
    ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    live = row < ROWS
    req = row // WIDTH
    s = tl.load(PACKET + row * 4, live, other=-float("inf"))
    t = tl.load(PACKET + row * 4 + 1, live, other=float(2**30))
    a = tl.load(PACKET + row * 4 + 2, live, other=0.0)
    b = tl.load(PACKET + row * 4 + 3, live, other=0.0)
    real = tl.load(REAL + req, live, other=False)
    available = tl.load(AVAILABLE + req, live, other=True)
    posinf = float("inf")
    # Audit the original packet before applying the production merge.
    malformed = (_invalid_packet_value(s) | _invalid_packet_value(t)
                 | _invalid_packet_value(a) | _invalid_packet_value(b))
    tl.atomic_add(ERROR, tl.max((live & real & malformed).to(tl.int32), 0).to(tl.int64), sem="relaxed")
    disabled = live & real & (row % WIDTH == 0) & ~available
    tl.atomic_add(AUDIT + 7, tl.sum(disabled.to(tl.int64), 0), sem="relaxed")

    # Keep the production _merge floating operations and Triton NaN policy.
    # With TP1, the only values merged with the packet are its initial sentinels.
    best = tl.maximum(tl.full((BLOCK,), -float("inf"), tl.float32), s)
    sentinel = tl.full((BLOCK,), float(2**30), tl.float32)
    cold = tl.minimum(sentinel, tl.where(s == best, t, sentinel))
    zero = tl.full((BLOCK,), 0., tl.float32)
    tl.store(READOUT + row * 4, best, live)
    tl.store(READOUT + row * 4 + 2, zero + a, live)
    tl.store(READOUT + row * 4 + 3, zero + b, live)
    tl.store(COLD + row, cold.to(tl.int64), live)
    tl.store(USABLE + row, (best != posinf) & (best != -posinf) & (best == best), live)


def _launch_inputs(p, ids, slots, valid, prefix, real, overflow, known, prompt, error, width, vocab):
    rows = len(p)
    safe_ids = torch.empty_like(ids)
    safe_slots = torch.empty_like(slots)
    active = torch.empty((rows,), dtype=torch.bool, device=p.device)
    available = torch.empty_like(real)
    enabled = torch.empty_like(active)
    _inputs[(1,)](
        p, ids, slots, valid, prefix, real, overflow, known, prompt, error,
        safe_ids, safe_slots, active, available, enabled,
        ROWS=rows, WIDTH=width, BLOCK=triton.next_power_of_2(rows),
        POOL_ROWS=len(overflow), VOCAB=vocab, num_warps=4, enable_fp_fusion=False,
    )
    return safe_ids, safe_slots, active, available, enabled


def packet_inputs(p, ids, slots, valid, prefix, real, overflow, known, prompt, error, width, vocab):
    rows, requests, capacity = len(p), len(slots), len(overflow)
    if (p.device.type != "cuda" or width not in (1, 2)
            or not 0 < rows <= 64 or rows != requests * width or not capacity):
        return None
    if prefix.dtype not in (torch.int32, torch.int64):
        return None
    specs = (
        (p, (rows, 2), torch.float32), (ids, (rows, 2), torch.int64),
        (slots, (requests,), torch.int64), (valid, (requests, width), torch.bool),
        (prefix, (requests,), prefix.dtype), (real, (requests,), torch.bool),
        (overflow, (capacity,), torch.bool), (known, (capacity,), torch.int64),
        (prompt, (capacity,), torch.int64), (error, (1,), torch.int64),
    )
    if any(t.shape != shape or t.dtype != dtype or t.device != p.device
           or not t.is_contiguous() for t, shape, dtype in specs):
        return None
    return _launch_inputs(p, ids, slots, valid, prefix, real, overflow, known, prompt, error, width, vocab)


def finish_packet(packet, available, real, error, audit, width):
    rows = len(packet)
    out = torch.empty_like(packet)
    cold = torch.empty((rows,), dtype=torch.long, device=packet.device)
    usable = torch.empty((rows,), dtype=torch.bool, device=packet.device)
    _finish[(1,)](
        packet, available, real, error, audit, out, cold, usable,
        ROWS=rows, WIDTH=width, BLOCK=triton.next_power_of_2(rows),
        num_warps=4, enable_fp_fusion=False,
    )
    return out[:, 0], cold, out[:, 2:4], usable
