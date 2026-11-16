"""Fail-closed deployment support and exact-prefix collection contracts."""
import hashlib
import json
import struct
from pathlib import Path
from .candidates import (generator_sha256, ROOT_POLICY, TIE_POLICY, POLICIES,
                         FrozenHotVocabulary, CandidateGenerator)

ASSET_SCHEMA = "context_repair_calibration_asset_v1"
ROW_SCHEMA = "exact_prefix_target_event_v1"
FEATURE_SCHEMA = "joint11_has_x_v1"
FEATURE_NAMES = ("log_p1", "log_p2", "log_rest", "depth_over_6", "p1", "p2",
                 "p1_minus_p2", "depth_squared", "legacy_margin", "margin_plus_log_p2",
                 "margin_times_depth", "has_x")
CLASSES = ("native1", "native2", "x", "other")
CANDIDATE_PAIR_POLICY = "posterior_top2_stable_v1"
DRAFT_KV_ANCESTRY = "logical_parent_exact_v1"
BOUND_FIELDS = (
    "target_model_id", "target_weights_sha256", "draft_model_id", "draft_weights_sha256",
    "tokenizer_sha256", "hot_map_sha256", "alignment_sha256",
    "candidate_generator_sha256", "candidate_policy", "capacity", "root_policy", "tie_policy",
    "readout_backend", "readout_numeric_version", "hidden_size", "target_vocab_size",
    "draft_vocab_size", "dtype", "tp_size", "max_steps", "feature_schema", "feature_names",
    "label_schema", "sampling_mode", "feature_dim", "score_mode",
    "candidate_pair_policy", "draft_kv_ancestry")
HASH_FIELDS = tuple(k for k in BOUND_FIELDS if k.endswith("sha256"))


class ContractError(ValueError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hot_map_digest(target_ids):
    """Hash the ordered draft-index -> target-ID map as signed little-endian i64.

    JSON whitespace/order of object keys is not part of model identity. Map ITEM
    order is essential: sorting or converting to a set would change its meaning.
    The JSON artifact's byte SHA belongs in collection/training receipts instead.
    """
    values = list(target_ids)
    if any(type(v) is not int or not 0 <= v < 2**63 for v in values):
        raise ContractError("hot map must contain nonnegative signed-int64 token IDs")
    return hashlib.sha256(struct.pack("<"+str(len(values))+"q", *values)).hexdigest()


def validate_tp_binding(asset_tp_size, runtime_tp_size):
    """Accept positive widths, but never borrow calibration across TP layouts.

    Hardware availability and model head/shard divisibility are engine checks.
    This contract deliberately does not assume powers of two or claim that every
    positive width is executable for every model. BF16 numerical parity and
    calibration evidence belong to the EXACT declared runtime width.
    """
    for label, value in (("asset", asset_tp_size), ("runtime", runtime_tp_size)):
        if type(value) is not int or value < 1:
            raise ContractError(label + " tp_size must be a positive integer")
    if asset_tp_size != runtime_tp_size:
        raise ContractError("TP calibration mismatch: asset TP" + str(asset_tp_size) +
                            " cannot be used for runtime TP" + str(runtime_tp_size) +
                            "; collect or independently validate a matching TP asset")
    return runtime_tp_size


def prefix_hash(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


def contract_hash(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_qid(row):
    """Do not include source or repetition in identity: aliases remain one question."""
    if row.get("canonical_question_id"):
        return str(row["canonical_question_id"])
    if row.get("domain") and row.get("id") is not None:
        return f"{row['domain']}:{row['id']}"
    raise ContractError("missing canonical question identity")


def split_question(qid, seed="context-repair-20261114-v1"):
    bucket = int(hashlib.sha256((seed + "|" + qid).encode()).hexdigest()[:16], 16) % 100
    return "train" if bucket < 60 else "validation" if bucket < 80 else "test"


def asset_metadata_errors(metadata, expected):
    errors = []
    if set(metadata) - set(BOUND_FIELDS):
        errors.append("unknown asset metadata fields; use a new contract version")
    for k in BOUND_FIELDS:
        if k not in metadata or metadata[k] is None or metadata[k] == "":
            errors.append(f"missing support field: {k}")
        if k not in expected:
            errors.append(f"runtime must supply expected field: {k}")
        elif metadata.get(k) != expected[k]:
            errors.append(f"support mismatch: {k}")
    try:
        validate_tp_binding(metadata.get("tp_size"), expected.get("tp_size"))
    except ContractError as e:
        errors.append(str(e))
    for k in HASH_FIELDS:
        v = metadata.get(k)
        if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            errors.append(f"invalid sha256: {k}")
    if metadata.get("feature_schema") != FEATURE_SCHEMA:
        errors.append("unsupported feature schema")
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        errors.append("feature order mismatch")
    if metadata.get("feature_dim") != len(FEATURE_NAMES):
        errors.append("feature dimension mismatch")
    if metadata.get("score_mode") != "unified_target_event_v2":
        errors.append("unsupported probability semantics")
    if metadata.get("candidate_pair_policy") != CANDIDATE_PAIR_POLICY:
        errors.append("unsupported candidate pair selection policy")
    if metadata.get("draft_kv_ancestry") != DRAFT_KV_ANCESTRY:
        errors.append("unsupported draft KV ancestry policy")
    if metadata.get("sampling_mode") != "greedy":
        errors.append("only the greedy target-event label is implemented")
    for k in ("capacity", "hidden_size", "target_vocab_size", "draft_vocab_size", "tp_size", "max_steps"):
        if type(metadata.get(k)) is not int or metadata[k] < 1:
            errors.append("invalid positive integer support field: " + k)
    if metadata.get("max_steps") != 7:
        errors.append("current feature depth schema requires 7 steps")
    if metadata.get("candidate_policy") not in POLICIES:
        errors.append("unsupported candidate policy")
    if metadata.get("dtype") not in ("bfloat16", "float16", "float32"):
        errors.append("unsupported readout dtype")
    if metadata.get("label_schema") != ROW_SCHEMA:
        errors.append("unsupported target label provenance")
    if metadata.get("candidate_generator_sha256") != generator_sha256():
        errors.append("candidate generator differs from this implementation")
    if metadata.get("root_policy") != ROOT_POLICY or metadata.get("tie_policy") != TIE_POLICY:
        errors.append("candidate root or tie semantics mismatch")
    return errors


def load_deployment_asset(path, expected):
    """No allow_diagnostic escape hatch. Offline scripts use a separate loader."""
    asset = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = asset_metadata_errors(asset.get("metadata", {}), expected)
    if asset.get("schema") != ASSET_SCHEMA:
        errors.append("asset schema mismatch")
    for k in ("deployment_validated", "publication_ready"):
        if asset.get(k) is not True:
            errors.append(k + " is not true")
    evidence = asset.get("validation_evidence", {})
    for k in ("exact_prefix_labels", "per_row_parity", "no_x_observed",
              "native1_parent_observed", "native2_parent_observed", "x_parent_observed",
              "deployment_readout_parity", "independent_question_evaluation"):
        if evidence.get(k) is not True:
            errors.append("unvalidated evidence: " + k)
    if asset.get("classes") != list(CLASSES):
        errors.append("class order mismatch")
    from .model import validate_parameters
    try:
        validate_parameters(asset.get("parameters", {}))
    except ValueError as e:
        errors.append(str(e))
    if errors:
        raise ContractError("deployment rejected: " + "; ".join(errors))
    return asset


def load_diagnostic_asset(path):
    """Explicitly offline; result must never be passed to deployment constructor."""
    asset = json.loads(Path(path).read_text(encoding="utf-8"))
    if asset.get("schema") != ASSET_SCHEMA:
        raise ContractError("not a calibration diagnostic asset")
    from .model import validate_parameters
    validate_parameters(asset["parameters"])
    return asset


def validate_contract(path, runtime_contract):
    """Stable runtime alias; unlike diagnostic loading there is no override flag."""
    return load_deployment_asset(path, runtime_contract)


def validate_collection_row(row, metadata, hot_ids, *, candidate_generator=None):
    """Strict new data only. Legacy rows cannot be silently promoted to this schema."""
    required = ("schema", "canonical_question_id", "source", "question_content_sha256",
                "parent_kind", "parent_prefix_sha256", "target_prefix_sha256",
                "visible_prefix_sha256", "native_parity", "readout_parity",
                "candidate_generator_sha256", "prompt_ids", "committed_output_ids",
                "root_token", "candidate_ids", "candidate_scores", "x_token", "has_x",
                "native1", "native2", "gold", "p1", "p2", "depth", "native2_logit",
                "runtime_contract_sha256", "source_manifest_sha256", "target_execution_id", "label_origin")
    missing = [k for k in required if k not in row]
    if missing:
        raise ContractError("missing row fields: " + ",".join(missing))
    if row["schema"] != ROW_SCHEMA:
        raise ContractError("row schema mismatch")
    if row["runtime_contract_sha256"] != contract_hash(metadata):
        raise ContractError("row runtime/numeric contract mismatch")
    if row["label_origin"] != "actual_target_forward" or not row["target_execution_id"]:
        raise ContractError("actual conditional target execution evidence required")
    for k in ("question_content_sha256", "parent_prefix_sha256", "target_prefix_sha256",
              "source_manifest_sha256", "visible_prefix_sha256"):
        v = row[k]
        if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v):
            raise ContractError("invalid row hash: " + k)
    if row["parent_prefix_sha256"] != row["target_prefix_sha256"]:
        raise ContractError("target label came from a different conditional prefix")
    if not row["parent_prefix_sha256"]:
        raise ContractError("empty prefix proof")
    if row["native_parity"] is not True or row["readout_parity"] is not True:
        raise ContractError("parity failed or unknown")
    if row["parent_kind"] not in ("root", "native1", "native2", "x"):
        raise ContractError("unknown parent provenance")
    if row["parent_kind"] == "root" and row["depth"] != 0:
        raise ContractError("root parent must have depth zero")
    if row["candidate_generator_sha256"] != generator_sha256():
        raise ContractError("row uses a different generator")
    visible = list(row["prompt_ids"]) + list(row["committed_output_ids"])
    if row["visible_prefix_sha256"] != prefix_hash(visible):
        raise ContractError("visible committed prefix hash mismatch")
    if candidate_generator is None:
        candidate_generator = CandidateGenerator(hot_ids, capacity=metadata["capacity"],
            policy=metadata["candidate_policy"], target_vocab_size=metadata["target_vocab_size"])
    elif (not isinstance(hot_ids, FrozenHotVocabulary) or
          candidate_generator.hot_vocabulary is not hot_ids or
          candidate_generator.capacity != metadata["capacity"] or
          candidate_generator.policy != metadata["candidate_policy"] or
          candidate_generator.target_vocab_size != metadata["target_vocab_size"]):
        raise ContractError("prevalidated generator does not match this collection contract")
    pool = candidate_generator(row["prompt_ids"], row["committed_output_ids"], root_token=row["root_token"])
    if list(pool.candidate_ids) != row["candidate_ids"]:
        raise ContractError("candidate list not generated by the deployment policy")
    from .candidates import select_scored_candidate
    x, _ = select_scored_candidate(row["candidate_ids"], row["candidate_scores"])
    if row["has_x"] is not (x is not None) or row["x_token"] != x:
        raise ContractError("X availability or score selection mismatch")
    ids = [row["native1"], row["native2"]] + ([] if x is None else [x])
    if len(set(ids)) != len(ids):
        raise ContractError("overlapping four-class events")
    if row["native1"] not in hot_ids or row["native2"] not in hot_ids:
        raise ContractError("native proposal outside declared hot-map")
    if type(row["gold"]) is not int or not 0 <= row["gold"] < metadata["target_vocab_size"]:
        raise ContractError("invalid gold target token")
    return True
