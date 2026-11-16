"""Stage the experimental 8192-wide TP8 repair port in a separate runtime."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from install_runtime import verify_installed


SPEC = Path("sglang/srt/speculative")
CONTEXT = SPEC / "context_repair.py"
READOUT = SPEC / "repair_row_readout.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f"Unexpected source layout for {old!r}")
    return text.replace(old, new)


def expected_files(source):
    context = (source / CONTEXT).read_text(encoding="utf-8")
    start = (
        "        if self.target.ndim != 2 or self.draft.ndim != 2 or "
        "self.target.shape[1] != 4096 or self.draft.shape[1] != 4096:\n"
        "            raise ValueError('Unsupported hidden dimensions')"
    )
    context = replace(
        context,
        start,
        "        if self.target.ndim != 2 or self.draft.ndim != 2 or "
        "self.target.shape[1] != 8192 or self.draft.shape[1] != 8192:\n"
        "            raise ValueError('The 70B port requires two 8192-wide output heads')\n"
        "        if group.world_size != 8 or self.mode != 'trained_experiment' or "
        "self.backend != 'fused_cached_v1':\n"
        "            raise ValueError('The 70B port requires TP8 and matched experimental fused-readout assets')\n"
        "        self.hidden_size = self.target.shape[1]",
    )
    substitutions = (
        ("alignment_cpu.shape != (4096,4096)", "alignment_cpu.shape != (self.hidden_size, self.hidden_size)", 1),
        ("hidden_size=4096", "hidden_size=self.hidden_size", 1),
        ("h.shape[1]==4096", "h.shape[1]==self.hidden_size", 1),
        ("D=4096", "D=self.hidden_size", 2),
        ("BLOCK=4096", "BLOCK=self.hidden_size", 2),
    )
    for old, new, count in substitutions:
        context = replace(context, old, new, count)
    readout = (source / READOUT).read_text(encoding="utf-8")
    readout = replace(readout, "d==4096 and x.dtype", "d==tw.shape[1]==dw.shape[1]==8192 and x.dtype")
    readout = replace(readout, "B=4096", "B=tr.next_power_of_2(d)")
    changes = {CONTEXT: context, READOUT: readout}
    for path, content in changes.items():
        compile(content, str(path), "exec")
    return changes


def inventory(source, changes=None):
    changes = changes or {}
    return {
        relative.as_posix(): (
            hashlib.sha256(changes[relative].encode()).hexdigest() if relative in changes else sha(path)
        )
        for path in sorted(source.rglob("*.py"))
        if (relative := path.relative_to(source)).parts[0] == "sglang"
    }


def stage(root):
    verify_installed(root)
    source, target = root / "repair", root / "repair70b"
    changes = expected_files(source)
    expected = inventory(source, changes)
    receipt = {
        "schema": "oov-70b-tp8-port-v1",
        "gpu_validation": "pending",
        "changed_files": {path.as_posix(): expected[path.as_posix()] for path in changes},
        "base_receipt_sha256": sha(root / "receipt.json"),
    }
    if target.exists():
        if inventory(target) != expected or json.loads((target / "receipt.json").read_text()) != receipt:
            raise ValueError("Existing 70B runtime differs from the staged experiment")
        return target
    with tempfile.TemporaryDirectory(prefix=".repair70b-", dir=root) as folder:
        pending = Path(folder) / "candidate"
        shutil.copytree(source, pending, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path, content in changes.items():
            (pending / path).write_bytes(content.encode("utf-8"))
        if inventory(pending) != expected:
            raise ValueError("70B source inventory mismatch")
        (pending / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        pending.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path("generated/runtime"))
    args = parser.parse_args()
    print(stage(args.runtime_root.resolve()))


if __name__ == "__main__":
    main()
