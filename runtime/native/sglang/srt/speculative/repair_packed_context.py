"""One ordered upload of already validated ContextRepair rows."""

from dataclasses import dataclass

import numpy as np


CAPACITY = 64
STOP_CAPACITY = 32
SCALAR_COLUMNS = 5
ROW_WIDTH = SCALAR_COLUMNS + CAPACITY + STOP_CAPACITY
RING_SIZE = 3
MAX_BATCH = 64


class PackedStaticRows:
    """CPU memoization only. All six fields are still uploaded and scattered."""

    def __init__(self):
        self.rows = None
        self.signatures = {}
        self.hits = 0
        self.rebuilt = 0

    def write(self, target, slots, pools, stops, total_slots):
        if self.rows is None or len(self.rows) != total_slots:
            self.rows = np.empty((total_slots, CAPACITY + STOP_CAPACITY), dtype=np.int64)
            self.signatures.clear()
        for slot, pool, stop in zip(slots, pools, stops, strict=True):
            if len(pool) > CAPACITY or len(stop) != STOP_CAPACITY:
                raise ValueError("Invalid packed ContextRepair row")
            stop = tuple(stop)
            signature = (pool, stop)
            if self.signatures.get(slot) == signature:
                self.hits += 1
                continue
            current = self.rows[slot]
            current[:CAPACITY].fill(-1)
            current[:len(pool)] = pool
            current[CAPACITY:] = stop
            self.signatures[slot] = signature
            self.rebuilt += 1
        target[:, SCALAR_COLUMNS:] = self.rows[slots]


def pack_rows(matrix, slots, pools, disabled, counts, prompts, stops, output_limits,
              static_rows, total_slots):
    """Write only the active rows of a reusable contiguous int64 NumPy view."""
    active = matrix[:len(slots)]
    active[:, 0] = slots
    active[:, 1] = disabled
    active[:, 2] = counts
    active[:, 3] = prompts
    active[:, 4] = output_limits
    static_rows.write(active, slots, pools, stops, total_slots)


@dataclass
class HostSlot:
    tensor: object
    matrix: object
    release: object
    issued: bool = False


def available_host_slot(ring, cursor):
    for offset in range(len(ring)):
        idx = (cursor + offset) % len(ring)
        slot = ring[idx]
        if not slot.issued or slot.release.query():
            return idx
    return None


class PackedContextUpload:
    def __init__(self, device, max_rows):
        import torch
        from .repair_packed_context_kernel import scatter

        self.torch = torch
        self.device = torch.device(device)
        self.max_rows = min(max_rows, MAX_BATCH)
        shape = (self.max_rows, ROW_WIDTH)
        self.staging = torch.empty(shape, dtype=torch.int64, device=self.device)
        self.allocation_stream = torch.cuda.current_stream(self.device)
        self.ring = []
        for _ in range(RING_SIZE):
            host = torch.empty(shape, dtype=torch.int64, device="cpu", pin_memory=True)
            self.ring.append(HostSlot(host, host.numpy(), torch.cuda.Event(enable_timing=False)))
        self.scatter = scatter
        self.cursor = 0
        self.stream_id = None
        self.last_release = None
        self.stream_changed = False
        self.packed_calls = 0
        self.fallback_calls = 0
        self.static_rows = PackedStaticRows()

    def begin_update(self):
        """Observe stream changes before even the unchanged stale/epoch writes."""
        if self.stream_changed:
            return None
        if self.torch.cuda.is_current_stream_capturing():
            return None
        stream = self.torch.cuda.current_stream(self.device)
        if self.stream_id is None:
            if stream.cuda_stream != self.allocation_stream.cuda_stream:
                stream.wait_stream(self.allocation_stream)
            self.stream_id = stream.cuda_stream
            self.staging.record_stream(stream)
        elif self.stream_id != stream.cuda_stream:
            # Establish order before writing the original buffers by the fallback.
            if self.last_release is not None:
                stream.wait_event(self.last_release)
            self.stream_changed = True
            return None
        return stream

    def upload(self, stream, state, slots, pools, disabled, counts, prompts, stops, output_limits):
        count = len(slots)
        # The original index_copy_ path remains authoritative outside this shape.
        if (
            not 0 < count <= self.max_rows
            or len(set(slots)) != count
            or any(not 0 < slot < state.pool.shape[0] for slot in slots)
            or stream is None
        ):
            self.fallback_calls += 1
            return False
        index = available_host_slot(self.ring, self.cursor)
        if index is None:
            self.fallback_calls += 1
            return False
        slot = self.ring[index]
        pack_rows(slot.matrix, slots, pools, disabled, counts, prompts, stops, output_limits,
                  self.static_rows, state.pool.shape[0])
        try:
            # On this stream the next upload cannot overwrite staging until
            # scatter is done. The release event also prevents early host writes.
            self.staging[:count].copy_(slot.tensor[:count], non_blocking=True)
            self.scatter[(count,)](
                self.staging, state.pool, state.overflow, state.known_counts,
                state.prompt_lengths, state.output_limits, state.stop_ids,
                CAP=CAPACITY, STOP_CAP=STOP_CAPACITY, STRIDE=ROW_WIDTH, num_warps=4,
            )
        finally:
            slot.release.record(stream)
            slot.issued = True
            self.last_release = slot.release
        self.cursor = (index + 1) % len(self.ring)
        self.packed_calls += 1
        return True
