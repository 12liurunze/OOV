"""Scatter one uploaded context matrix into the graph-visible request arrays."""

import triton as tr
import triton.language as tl


@tr.jit
def scatter(
    SRC, POOL, OVERFLOW, COUNTS, PROMPTS, LIMITS, STOPS,
    CAP: tl.constexpr, STOP_CAP: tl.constexpr, STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    base = SRC + row * STRIDE
    slot = tl.load(base).to(tl.int64)
    col = tl.arange(0, CAP)
    pool = tl.load(base + 5 + col)
    stop = tl.load(base + 5 + CAP + col, col < STOP_CAP, other=-1)
    tl.store(POOL + slot * CAP + col, pool)
    tl.store(STOPS + slot * STOP_CAP + col, stop, col < STOP_CAP)
    tl.store(OVERFLOW + slot, tl.load(base + 1) != 0)
    tl.store(COUNTS + slot, tl.load(base + 2))
    tl.store(PROMPTS + slot, tl.load(base + 3))
    tl.store(LIMITS + slot, tl.load(base + 4))
