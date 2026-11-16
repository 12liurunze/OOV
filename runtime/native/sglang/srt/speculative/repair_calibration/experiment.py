"""Load a freshly trained artifact for a pinned research run."""

import json
from pathlib import Path

from .contracts import ASSET_SCHEMA, CLASSES, ContractError, asset_metadata_errors, sha256_file
from .model import validate_parameters


def load_experiment_asset(path, expected, expected_hash):
    path = Path(path)
    if not expected_hash or sha256_file(path) != expected_hash:
        raise ContractError("experiment calibration asset differs from the pinned training output")
    asset = json.loads(path.read_text(encoding="utf-8"))
    errors = asset_metadata_errors(asset.get("metadata", {}), expected)
    if asset.get("schema") != ASSET_SCHEMA or asset.get("classes") != list(CLASSES):
        errors.append("experiment classifier schema or class order differs")
    if asset.get("model_name") != "exact_prefix_candidate_pending_validation":
        errors.append("experiment requires a newly trained exact-prefix classifier")
    if asset.get("deployment_validated") is not False or asset.get("publication_ready") is not False:
        errors.append("experiment assets must retain their research-only status")
    training = asset.get("training", {})
    if not training.get("training_question_ids") or training.get("fit", {}).get("converged") is not True:
        errors.append("experiment classifier has no successful fit receipt")
    attestations = asset.get("collection_attestations", {})
    for key in ("matching_parent_target_prefix_hashes", "actual_target_forward_declared"):
        if attestations.get(key) is not True:
            errors.append("missing collection attestation: " + key)
    if errors:
        raise ContractError("experiment classifier rejected: " + "; ".join(errors))
    validate_parameters(asset.get("parameters", {}))
    return asset
