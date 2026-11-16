"""Build a 70B calibration runtime without enabling untrained online repair."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from stage_70b import CONTEXT, inventory, replace, sha, stage


def patch_context(context):
    context = replace(
        context,
        "        if self.mode not in ('validated', 'legacy_diagnostic', 'trained_experiment'):\n"
        "            raise ValueError('Unknown repair mode')",
        "        if self.mode != 'calibration_collect' or os.environ.get('OOV_TP8_COLLECTION') != '1':\n"
        "            raise ValueError('This isolated runtime only supports opt-in TP8 calibration')",
    )
    context = replace(
        context,
        "group.world_size != 8 or self.mode != 'trained_experiment' or",
        "group.world_size != 8 or self.mode != 'calibration_collect' or",
    )
    context = replace(
        context,
        "The 70B port requires TP8 and matched experimental fused-readout assets",
        "The collection runtime requires TP8 and the same fused cached readout as the benchmark",
    )
    old = "        alignment_cpu = torch.load(root/'alignment.pt', map_location='cpu', weights_only=True)['transform']"
    context = replace(
        context,
        old,
        "        alignment_path = root / 'alignment.pt'\n"
        "        self.alignment_sha256 = sha256_file(alignment_path)\n"
        "        if self.alignment_sha256 != os.environ.get('OOV_TP8_ALIGNMENT_SHA256'):\n"
        "            raise ValueError('Alignment file differs from the collection plan')\n"
        "        if hot_map_digest(hot_cpu_list) != os.environ.get('OOV_TP8_HOT_MAP_SHA256'):\n"
        "            raise ValueError('Loaded Draft hot map differs from the collection plan')\n"
        + old,
    )
    # The classifier loader, including its strict asset check in the benchmark
    # runtime, is left untouched. There is no posterior in a collection worker.
    begin = context.index("        self.raw_budget = 26\n") + len("        self.raw_budget = 26\n")
    end = context.index("        self.pool = torch.full(", begin)
    context = context[:begin] + (
        "        self.score_mode = 'calibration_collect'\n"
        "        self.candidate_pair_policy = 'preserve_native1_v1'\n"
        "        self.asset_hash = None\n"
        "        self.mapped_tree_scores = False\n"
        "        self.legacy_mixed_tree_scores = False\n"
        "        self.fused_proposal_commit = False\n"
        "        self.draft_kv_ancestry = DRAFT_KV_ANCESTRY\n"
        "        self._has_x_feature_table = None\n"
    ) + context[end:]
    context = replace(
        context,
        "        width=1 if step==0 else 2\n        real=real_request_mask",
        "        if self.mode == 'calibration_collect':\n"
        "            raise RuntimeError('TP8 calibration proposal hook was not installed')\n"
        "        width=1 if step==0 else 2\n        real=real_request_mask",
    )
    compile(context, str(CONTEXT), "exec")
    return context


def stage_collection(root):
    source = stage(root)
    target = root / "collect70b"
    changes = {CONTEXT: patch_context((source / CONTEXT).read_text(encoding="utf-8"))}
    expected = inventory(source, changes)
    receipt = {
        "schema": "oov-70b-tp8-calibration-runtime-v1",
        "base_receipt_sha256": sha(source / "receipt.json"),
        "changed_files": {CONTEXT.as_posix(): expected[CONTEXT.as_posix()]},
    }
    if target.exists():
        if inventory(target) != expected or json.loads((target / "receipt.json").read_text(encoding="utf-8")) != receipt:
            raise ValueError("Existing collection runtime differs from the verified source")
        return target
    with tempfile.TemporaryDirectory(prefix=".collect70b-", dir=root) as folder:
        pending = Path(folder) / "runtime"
        shutil.copytree(source, pending, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path, content in changes.items():
            (pending / path).write_bytes(content.encode("utf-8"))
        if inventory(pending) != expected:
            raise ValueError("Staged collection source inventory mismatch")
        (pending / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        pending.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path("generated/runtime"))
    args = parser.parse_args()
    print(stage_collection(args.runtime_root.resolve()))


if __name__ == "__main__":
    main()
