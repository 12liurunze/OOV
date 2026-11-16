"""Legacy TP1 decision after the unchanged FP64 posterior."""
import torch
import triton
import triton.language as tl


@triton.jit
def _proposal_commit(
    P, IDS, POSTERIOR, COLD, ACTIVE, USABLE, REMAINING, SLOTS, STOPS,
    ERROR, AUDIT, OUT_P, OUT_IDS, TERMINAL,
    ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
    STOP_ROWS: tl.constexpr, STOP_COUNT: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    in_bounds = row < ROWS
    request = row // WIDTH
    active = tl.load(ACTIVE + row, in_bounds, other=False)
    usable = tl.load(USABLE + row, in_bounds, other=False)
    remaining = tl.load(REMAINING + request, in_bounds, other=0)
    live = active & (remaining > 0)

    post0 = tl.load(POSTERIOR + 4 * row, in_bounds, other=0)
    post1 = tl.load(POSTERIOR + 4 * row + 1, in_bounds, other=0)
    post2 = tl.load(POSTERIOR + 4 * row + 2, in_bounds, other=0)
    post3 = tl.load(POSTERIOR + 4 * row + 3, in_bounds, other=0)
    finite0 = (post0 == post0) & (tl.abs(post0) != float("inf"))
    finite1 = (post1 == post1) & (tl.abs(post1) != float("inf"))
    finite2 = (post2 == post2) & (tl.abs(post2) != float("inf"))
    finite3 = (post3 == post3) & (tl.abs(post3) != float("inf"))
    bad_posterior = in_bounds & active & ~(finite0 & finite1 & finite2 & finite3)

    # Comparing on the FP64 posterior is necessary for near ties.
    use = live & usable & (post2 > post1)
    token0 = tl.load(IDS + 2 * row, in_bounds, other=0)
    token1 = tl.load(IDS + 2 * row + 1, in_bounds, other=0)
    alternative = tl.load(COLD + row, in_bounds, other=0)
    repaired1 = tl.where(use, alternative, token1)
    p0 = tl.load(P + 2 * row, in_bounds, other=0)
    p1 = tl.load(P + 2 * row + 1, in_bounds, other=0)
    tl.store(OUT_P + 2 * row, tl.where(use, post0.to(tl.float32), p0), in_bounds)
    tl.store(OUT_P + 2 * row + 1, tl.where(use, post2.to(tl.float32), p1), in_bounds)
    tl.store(OUT_IDS + 2 * row, token0, in_bounds)
    tl.store(OUT_IDS + 2 * row + 1, repaired1, in_bounds)

    # Check the final candidate IDs. Do so even for inactive/padding rows,
    # as the original candidate_terminal did. Negative stop-list fill is ignored.
    slot = tl.load(SLOTS + request, in_bounds, other=0)
    slot = tl.where(slot < 0, slot + STOP_ROWS, slot)
    valid_slot = (slot >= 0) & (slot < STOP_ROWS)
    col = tl.arange(0, STOP_COUNT)
    stop = tl.load(STOPS + slot[:, None] * STOP_COUNT + col[None, :],
                   (in_bounds & valid_slot)[:, None], other=-1)
    is_stop = stop >= 0
    stop0 = tl.max((is_stop & (stop == token0[:, None])).to(tl.int32), axis=1) != 0
    stop1 = tl.max((is_stop & (stop == repaired1[:, None])).to(tl.int32), axis=1) != 0
    tl.store(TERMINAL + 2 * row, stop0, in_bounds)
    tl.store(TERMINAL + 2 * row + 1, stop1, in_bounds)

    error = tl.max((bad_posterior | (in_bounds & ~valid_slot)).to(tl.int32), axis=0)
    tl.atomic_add(ERROR, error.to(tl.int64), sem="relaxed")
    tl.atomic_add(AUDIT, tl.sum((in_bounds & live).to(tl.int64), axis=0), sem="relaxed")
    tl.atomic_add(AUDIT + 1, tl.sum((in_bounds & live & usable).to(tl.int64), axis=0), sem="relaxed")
    tl.atomic_add(AUDIT + 2, tl.sum((in_bounds & use).to(tl.int64), axis=0), sem="relaxed")


def _launch(p, ids, posterior, cold, active, usable, remaining, slots, stop_ids,
            error, audit, width):
    rows = len(p)
    scores = torch.empty_like(p)
    repaired = torch.empty_like(ids)
    terminal = torch.empty_like(ids, dtype=torch.bool)
    _proposal_commit[(1,)](
        p, ids, posterior, cold, active, usable, remaining, slots, stop_ids,
        error, audit, scores, repaired, terminal,
        ROWS=rows, WIDTH=width, BLOCK=triton.next_power_of_2(rows),
        STOP_ROWS=len(stop_ids), STOP_COUNT=32, num_warps=4,
        enable_fp_fusion=False,
    )
    return scores, repaired, terminal


def proposal_commit(p, ids, posterior, cold, active, usable, remaining, slots,
                    stop_ids, error, audit, width):
    """Return (scores, IDs, terminal), or None for the unchanged fallback path."""
    rows = len(p)
    requests = len(slots)
    if rows == 0 or rows > 64 or width not in (1, 2) or rows != requests * width:
        return None
    specs = (
        (p, (rows, 2), torch.float32),
        (ids, (rows, 2), torch.int64),
        (posterior, (rows, 4), torch.float64),
        (cold, (rows,), torch.int64),
        (active, (rows,), torch.bool),
        (usable, (rows,), torch.bool),
        (remaining, (requests,), torch.int64),
        (slots, (requests,), torch.int64),
        (stop_ids, (len(stop_ids), 32), torch.int64),
        (error, (1,), torch.int64),
    )
    if (p.device.type != "cuda" or not len(stop_ids) or audit.ndim != 1
            or len(audit) < 3 or audit.dtype != torch.int64
            or audit.device != p.device or not audit.is_contiguous()
            or any(tuple(t.shape) != shape or t.dtype != dtype or t.device != p.device
                   or not t.is_contiguous() for t, shape, dtype in specs)):
        return None
    return _launch(p, ids, posterior, cold, active, usable, remaining, slots,
                   stop_ids, error, audit, width)
