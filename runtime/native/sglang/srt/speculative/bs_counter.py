"""Decode phase spans, drained asynchronously during the measured workload."""
import functools
import json
import mmap
import os
from collections import deque
from pathlib import Path

import torch

BUFFER = None
TRACE = None
current = None
pending = deque()
rows = []
free_events = []
path = os.getenv('SWEEP_BS_COUNTER')
if path:
    with open(path, 'r+b') as handle:
        mapping = mmap.mmap(handle.fileno(), 0)
    BUFFER = memoryview(mapping).cast('Q')
    TRACE = Path(path).with_name('draft_stages.json')


def event():
    return free_events.pop() if free_events else torch.cuda.Event(enable_timing=True)


def phase(name):
    def decorate(fn):
        if BUFFER is None:
            return fn

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if current is None:
                return fn(*args, **kwargs)
            start, end = event(), event()
            start.record()
            result = fn(*args, **kwargs)
            end.record()
            current[name] = (start, end)
            return result

        return wrapped
    return decorate


def drain(wait=False):
    while pending:
        size, graph, phases = pending[0]
        last = next(reversed(phases.values()))[1]
        if wait:
            last.synchronize()
        elif not last.query():
            break
        pending.popleft()
        row = {'bs': size, 'target_graph': graph}
        for name, (start, end) in phases.items():
            row[name + '_ms'] = start.elapsed_time(end)
            free_events.extend((start, end))
        rows.append(row)


def record_decode(fn):
    if BUFFER is None:
        return fn

    @functools.wraps(fn)
    def wrapped(self, batch, *args, **kwargs):
        global current
        gate = BUFFER[0]
        if gate == 2 and not TRACE.exists():
            drain(wait=True)
            tmp = TRACE.with_suffix('.tmp')
            tmp.write_text(json.dumps(rows))
            tmp.replace(TRACE)
        active = gate == 1 and not batch.forward_mode.is_idle()
        size = len(batch.reqs) if active and not batch.forward_mode.is_extend() and not batch.is_extend_in_batch else 0
        if not size:
            return fn(self, batch, *args, **kwargs)
        assert 1 <= size <= 512 and current is None
        current = {}
        try:
            result = fn(self, batch, *args, **kwargs)
            assert set(current) == {'draft', 'extend'}, list(current)
            graph = bool(result.can_run_cuda_graph)
            pending.append((size, graph, current))
            BUFFER[size] += 1
            if graph:
                BUFFER[512 + size] += 1
            if len(pending) >= 16:
                drain()
            return result
        finally:
            current = None

    return wrapped
