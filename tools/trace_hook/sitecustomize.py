"""Observe actual scheduler rows and committed outputs without CUDA synchronization."""
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
from time import monotonic_ns

if os.environ.get("REDHAT_BS32_TRACE"):
    root = Path(os.environ["REDHAT_BS32_TRACE"])
    target = "sglang.srt.managers.scheduler"

    def install(module):
        original = module.Scheduler.process_batch_result
        trace = None
        stream = None

        @functools.wraps(original)
        def record(self, batch, result):
            nonlocal trace, stream
            if self.ps.tp_rank != 0:
                return original(self, batch, result)
            phase = batch.forward_mode.is_decode() and not batch.is_extend_in_batch
            requests = batch.reqs
            rid = requests[0].rid if requests else ""
            label = rid.split("~", 1)[0] if rid.startswith("measure_") else ""
            if not label:
                if stream is not None:
                    stream.close()
                    stream = trace = None
                return original(self, batch, result)
            if label != trace:
                if stream is not None:
                    stream.close()
                stream = (root / f"{label}.jsonl").open("x", buffering=1, encoding="utf-8")
                trace = label
            requests = list(requests)
            if any(not req.rid.startswith(label + "~") for req in requests):
                raise RuntimeError("Mixed measured request groups")
            before = [len(r.output_ids_through_stop) for r in requests]
            returned = original(self, batch, result)
            done_ns = monotonic_ns()
            emitted = [len(r.output_ids_through_stop) - n for r, n in zip(requests, before)]
            row = {
                "step": batch.forward_iter, "launch_ns": round(batch.launch_ts * 1e9),
                "done_ns": done_ns, "phase": "decode" if phase else "other",
                "bs": len(requests), "rids": [r.rid for r in requests],
                "graph": bool(result.can_run_cuda_graph),
                "gap": bool(batch.after_idle_gap), "emitted": sum(emitted),
                "positive": sum(n > 0 for n in emitted),
                "finished": sum(r.finished() for r in requests),
                "accepted": int(result.num_correct_drafts) if phase else 0,
            }
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            return returned

        module.Scheduler.process_batch_result = record

    class Loader(importlib.abc.Loader):
        def __init__(self, loader):
            self.loader = loader

        def create_module(self, spec):
            return self.loader.create_module(spec)

        def exec_module(self, module):
            self.loader.exec_module(module)
            install(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target_module=None):
            if fullname == target:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                if spec and spec.loader:
                    spec.loader = Loader(spec.loader)
                return spec
            return None

    sys.meta_path.insert(0, Finder())
