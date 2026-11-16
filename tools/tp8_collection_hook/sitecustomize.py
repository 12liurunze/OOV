"""Opt-in import hooks in the eight spawned SGLang calibration workers."""

import os

if os.environ.get("OOV_TP8_COLLECTION") == "1":
    import importlib.abc
    import importlib.machinery
    import sys

    TARGET = "sglang.srt.speculative.eagle_worker_common"
    DRAFT = {
        "sglang.srt.speculative.context_repair",
        "sglang.srt.speculative.bound_frontier",
        "sglang.srt.model_executor.model_runner",
    }

    class CollectorLoader(importlib.abc.Loader):
        def __init__(self, original, entrypoint):
            self.original = original
            self.entrypoint = entrypoint

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            self.entrypoint(module)

    class CollectorFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == TARGET:
                from calibration.tp8_target import install
            elif fullname in DRAFT:
                from calibration.tp8_draft import install
            else:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = CollectorLoader(spec.loader, install)
            return spec

    sys.meta_path.insert(0, CollectorFinder())
