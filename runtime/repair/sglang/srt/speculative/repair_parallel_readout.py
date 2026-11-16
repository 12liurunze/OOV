"""Experimental parallel candidate readout for frozen cached TP1 repair."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _parallel_score(
    X, TW, DW, IDS, POOL, ROOT, HOT, SLOTS, VALID, REM, SCORES,
    WIDTH: tl.constexpr, CAP: tl.constexpr, XS: tl.constexpr,
    TWS: tl.constexpr, DWS: tl.constexpr,
    TLO: tl.constexpr, THI: tl.constexpr, DLO: tl.constexpr, DHI: tl.constexpr,
    VOCAB: tl.constexpr, D: tl.constexpr,
):
    column = tl.program_id(0)
    row = tl.program_id(1)
    k = tl.arange(0, D)
    if column <= CAP:
        req = row // WIDTH
        slot = tl.load(SLOTS + req).to(tl.int64)
        if column < CAP:
            token = tl.load(POOL + slot * CAP + column)
        else:
            token = tl.load(ROOT + slot)
        safe = tl.maximum(0, tl.minimum(VOCAB - 1, token))
        cold = tl.load(HOT + safe) < 0
        live = tl.load(VALID + row) & (tl.load(REM + req) > 0)
        owned = live & cold & (token >= TLO) & (token < THI)
        score = -float("inf")
        if owned:
            x = tl.load(X + row * XS + k).to(tl.float32)
            w = tl.load(TW + (token - TLO).to(tl.int64) * TWS + k).to(tl.float32)
            score = tl.sum(x * w, 0).to(tl.bfloat16).to(tl.float32)
    else:
        which = column - CAP - 1
        token = tl.load(IDS + row * 2 + which)
        index = tl.load(HOT + token)
        score = 0.0
        # Native scores do not depend on cold eligibility, matching the serial packet.
        if (index >= DLO) & (index < DHI):
            x = tl.load(X + row * XS + k).to(tl.float32)
            w = tl.load(DW + (index - DLO).to(tl.int64) * DWS + k).to(tl.float32)
            score = tl.sum(x * w, 0).to(tl.bfloat16).to(tl.float32)
    tl.store(SCORES + row * (CAP + 3) + column, score)


@tr.jit
def _parallel_reduce(
    SCORES, POOL, ROOT, SLOTS, OUT,
    WIDTH: tl.constexpr, CAP: tl.constexpr, BC: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(SLOTS + row // WIDTH).to(tl.int64)
    col = tl.arange(0, BC)
    pool = tl.load(POOL + slot * CAP + col, col < CAP, other=2**30)
    root = tl.load(ROOT + slot)
    scores = tl.load(SCORES + row * (CAP + 3) + col, col <= CAP, other=-float("inf"))
    nan = scores != scores
    nan_seen = tl.sum(nan.to(tl.int32), 0) > 0
    best = tl.max(tl.where(nan, -float("inf"), scores), 0)
    tokens = tl.where(col == CAP, root, pool)
    cold = tl.min(tl.where((col <= CAP) & (scores == best), tokens, 2**30), 0)
    # The serial packet reports any evaluated cold NaN, even when another score wins.
    # In the all -inf case it returns min(pool, root), including ineligible IDs.
    best = tl.where(nan_seen, float("nan"), best)
    cold = tl.where(nan_seen, 2**30, cold)
    native = tl.load(SCORES + row * (CAP + 3) + CAP + 1 + tl.arange(0, 2))
    tl.store(OUT + row * 4, best)
    tl.store(OUT + row * 4 + 1, cold.to(tl.float32))
    tl.store(OUT + row * 4 + 2 + tl.arange(0, 2), native)


def packet(x, tw, dw, ids, pool, roots, hot, slots, enabled, remaining, width,
           target_low, target_high, draft_low, draft_high, vocab=128256):
    rows, d = x.shape
    cap = pool.shape[1]
    assert d == 4096 and x.dtype == tw.dtype == dw.dtype == torch.bfloat16
    scores = torch.empty((rows, cap + 3), device=x.device, dtype=torch.float32)
    out = torch.empty((rows, 4), device=x.device, dtype=torch.float32)
    _parallel_score[(cap + 3, rows)](
        x, tw, dw, ids, pool, roots, hot, slots, enabled, remaining, scores,
        WIDTH=width, CAP=cap, XS=x.stride(0), TWS=tw.stride(0), DWS=dw.stride(0),
        TLO=target_low, THI=target_high, DLO=draft_low, DHI=draft_high,
        VOCAB=vocab, D=d, num_warps=4,
    )
    _parallel_reduce[(rows,)](
        scores, pool, roots, slots, out, WIDTH=width, CAP=cap,
        BC=tr.next_power_of_2(cap + 1), num_warps=4,
    )
    return out
