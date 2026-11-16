"""Fit and package a research gate using independently collected 70B TP8 rows."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .contracts import asset_metadata_errors, hot_map_digest, sha256_file
from .fit_alignment import DRAFT_VOCAB, HIDDEN, TARGET_VOCAB


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_contract(contract, alignment_dir):
    failures = asset_metadata_errors(contract, contract)
    expected = {
        "hidden_size": HIDDEN,
        "target_vocab_size": TARGET_VOCAB,
        "draft_vocab_size": DRAFT_VOCAB,
        "tp_size": 8,
        "dtype": "bfloat16",
        "capacity": 64,
        "readout_backend": "fused_cached_v1",
        "readout_numeric_version": "cached_WA_bf16_v1",
        "candidate_policy": "recent_distinct_v1",
        "alignment_sha256": sha256_file(alignment_dir / "alignment.pt"),
        "hot_map_sha256": hot_map_digest(read_json(alignment_dir / "hot_map.json")),
    }
    failures.extend(f"unexpected setting {key}" for key, value in expected.items()
                    if contract.get(key) != value)
    if failures:
        raise ValueError("Invalid 70B TP8 collection contract: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows, contract, alignment_dir = (p.resolve(strict=True) for p in
                                     (args.rows, args.contract, args.alignment_dir))
    verify_contract(read_json(contract), alignment_dir)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fit = out / "fit"
    subprocess.run([
        sys.executable, "-m", "calibration.train_exact",
        "--rows", str(rows), "--contract", str(contract),
        "--hot-map", str(alignment_dir / "hot_map.json"), "--out", str(fit),
        "--evaluate-keep-native1",
    ], check=True)
    candidate = fit / "candidate_pending_validation.json"
    payload = read_json(candidate)
    if (payload["deployment_validated"] is not False or payload["publication_ready"] is not False
            or payload["metadata"] != read_json(contract)):
        raise ValueError("Unexpected trained asset or deployment state")
    stage = out / "parameters"
    stage.mkdir()
    for source, name in (
        (alignment_dir / "alignment.pt", "alignment.pt"),
        (contract, "runtime_contract.json"),
        (candidate, "calibration_asset.json"),
    ):
        shutil.copyfile(source, stage / name)
    receipt = {
        "status": "research_only",
        "deployment_validated": False,
        "collection_receipt_checked_by_trainer": False,
        "files_sha256": {p.name: sha256_file(p) for p in sorted(stage.iterdir())},
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(out / "receipt.json")


if __name__ == "__main__":
    main()
