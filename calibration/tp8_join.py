"""Join rank-0 Draft and Target observations into reviewed training rows."""

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

from .candidates import CandidateGenerator, FrozenHotVocabulary
from .contracts import (ROW_SCHEMA, asset_metadata_errors, contract_hash, hot_map_digest,
                        prefix_hash, sha256_file, split_question, validate_collection_row)


FEATURE_FIELDS = ("parent_kind", "depth", "prompt_ids", "committed_output_ids", "root_token",
                  "native1", "native2", "p1", "p2", "native2_logit", "candidate_ids",
                  "candidate_scores", "x_token", "has_x", "readout_parity", "native_parity")


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("row is not an object")
                yield row
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid JSONL record in {path.name} at line {line_number}") from error


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_prompt_manifest(path):
    by_id = {}
    qid_to_content = {}
    content_to_qid = {}
    for entry in jsonl(path):
        rid, qid, source, digest, prompt = (entry[key] for key in
            ("request_id", "canonical_question_id", "source", "question_content_sha256", "prompt_ids"))
        if (not all(isinstance(v, str) and v for v in (rid, qid, source))
                or not is_digest(digest) or not isinstance(prompt, list) or not prompt or rid in by_id):
            raise ValueError("Invalid or duplicate input-manifest identity")
        if (qid_to_content.setdefault(qid, digest) != digest
                or content_to_qid.setdefault(digest, qid) != qid):
            raise ValueError("Input-manifest aliases would invalidate the by-question split")
        by_id[rid] = entry
    if not by_id:
        raise ValueError("Input manifest is empty")
    return by_id


def key_for(row, vocabulary_size, prompts):
    request = row["request_id"]
    visible_sha, root_token = row["visible_prefix_sha256"], row["root_token"]
    prefix = row["parent_prefix_ids"]
    if (request not in prompts or not is_digest(visible_sha)
            or type(root_token) is not int or not 0 <= root_token < vocabulary_size
            or not isinstance(prefix, list) or not prefix
            or any(type(v) is not int or not 0 <= v < vocabulary_size for v in prefix)
            or row.get("round_id") != f"{visible_sha}-{root_token}"):
        raise ValueError("Malformed request, prefix or round anchor in calibration trace")
    return request, visible_sha, root_token, prefix_hash(prefix)


def index_target(path, vocabulary_size, prompts):
    target = {}
    ambiguous = set()
    executions = set()
    count = 0
    for row in jsonl(path):
        count += 1
        key = key_for(row, vocabulary_size, prompts)
        execution, gold = row["target_execution_id"], row["gold"]
        if (row.get("capture_before_sampling") is not True or row.get("mask_parity") is not True
                or row["target_prefix_sha256"] != key[-1]
                or row["position"] != len(row["parent_prefix_ids"]) - 1
                or not isinstance(execution, str) or not execution or execution in executions
                or type(gold) is not int or not 0 <= gold < vocabulary_size):
            raise ValueError("Invalid Target verification provenance")
        executions.add(execution)
        if key in target and target[key][0] != gold:
            ambiguous.add(key)
        target.setdefault(key, (gold, execution))
    if not count:
        raise ValueError("No actual Target verification evidence")
    return target, ambiguous, count


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def make_draft_record(row, key, prompt_entry, hot, contract, contract_sha, source_sha, target, target_sha):
    prefix = row["parent_prefix_ids"]
    depth = row["depth"]
    visible = row["prompt_ids"] + row["committed_output_ids"]
    kind = row["parent_kind"]
    if (row["prompt_ids"] != prompt_entry["prompt_ids"] or prefix_hash(visible) != key[1]
            or type(depth) is not int or not 0 <= depth < contract["max_steps"]
            or len(prefix) != len(visible) + depth + 1
            or prefix[:len(visible)] != visible or prefix[len(visible)] != key[2]
            or (kind == "root") != (depth == 0)
            or (depth > 0 and (kind == "x") != (prefix[-1] not in hot))):
        raise ValueError("Draft ancestry or committed prefix does not match its capture")
    scores = row["candidate_scores"]
    if (not all(finite_number(score) for score in scores)
            or not finite_number(row["native2_logit"])
            or not all(finite_number(row[k]) for k in ("p1", "p2"))
            or not 0 <= row["p2"] <= row["p1"] <= 1
            or row["p1"] + row["p2"] > 1.00001):
        raise ValueError("Draft scores or probabilities lack complete finite values")
    gold, execution = target
    record = {k: row[k] for k in FEATURE_FIELDS}
    for field in ("canonical_question_id", "source", "question_content_sha256"):
        record[field] = prompt_entry[field]
    record.update(
        schema=ROW_SCHEMA,
        split=split_question(record["canonical_question_id"]),
        visible_prefix_sha256=key[1],
        parent_prefix_sha256=key[-1],
        target_prefix_sha256=key[-1],
        label_origin="actual_target_forward",
        target_execution_id=f"{target_sha}:{execution}",
        source_manifest_sha256=source_sha,
        runtime_contract_sha256=contract_sha,
        candidate_generator_sha256=contract["candidate_generator_sha256"],
        gold=gold,
    )
    return record


def join_capture(*, draft_path, target_path, prompts_path, contract_path, hot_map_path, output_dir):
    """Join an intact same-campaign trace, auditing unmatched and ineligible rows."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    errors = asset_metadata_errors(contract, contract)
    if errors or contract.get("tp_size") != 8 or contract.get("hidden_size") != 8192:
        raise ValueError("Invalid 70B TP8 collection contract: " + "; ".join(errors))
    contract_sha, source_sha = contract_hash(contract), sha256_file(prompts_path)
    hot_ids = json.loads(hot_map_path.read_text(encoding="utf-8"))
    if len(hot_ids) != contract["draft_vocab_size"] or hot_map_digest(hot_ids) != contract["hot_map_sha256"]:
        raise ValueError("Hot vocabulary differs from the TP8 runtime contract")
    vocab_size = contract["target_vocab_size"]
    frozen_hot = FrozenHotVocabulary(hot_ids, vocab_size)
    if len(frozen_hot) != len(hot_ids):
        raise ValueError("The offset vocabulary must map each Draft index to a different Target ID")
    generator = CandidateGenerator(frozen_hot, capacity=contract["capacity"], policy=contract["candidate_policy"])
    prompts = read_prompt_manifest(prompts_path)
    target, ambiguous, target_count = index_target(target_path, vocab_size, prompts)
    target_sha = sha256_file(target_path)
    final = output_dir / "exact_rows.jsonl.gz"
    audit_path = output_dir / "collection_audit.json"
    staged = output_dir / ".exact_rows.incomplete.jsonl.gz"
    if final.exists() or audit_path.exists() or staged.exists():
        raise FileExistsError("Use a new output directory for the joined collection")

    counts, kept_kinds, kept_events, excluded, split_rows, distinct = (
        Counter(), Counter(), Counter(), Counter(), Counter(), {})
    seen = {}
    with gzip.open(staged, "xt", encoding="utf-8") as out:
        for draft in jsonl(draft_path):
            counts["draft_rows"] += 1
            counts["selected_for_probe"] += draft.get("selected_for_probe") is True
            counts["probe_replaced_b"] += draft.get("probe_replaced_b") is True
            key = key_for(draft, vocab_size, prompts)
            if key in ambiguous:
                excluded["target_labels_disagree_for_same_prefix"] += 1
                continue
            if key not in target:
                excluded["not_in_actual_verify_tree"] += 1
                if draft.get("parent_kind") == "x":
                    counts["x_parent_missing_from_verify"] += 1
                continue
            if draft.get("native_parity") is not True or draft.get("readout_parity") is not True:
                excluded["draft_parity_unavailable_or_failed"] += 1
                continue
            record = make_draft_record(draft, key, prompts[key[0]], frozen_hot.ids,
                                       contract, contract_sha, source_sha, target[key], target_sha)
            validate_collection_row(record, contract, frozen_hot, candidate_generator=generator)
            record_hash = contract_hash(record)
            if key in seen:
                if seen[key] != record_hash:
                    raise ValueError("Conflicting Draft observations for one verification-round prefix")
                excluded["duplicate_draft_prefix"] += 1
                continue
            out.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            seen[key] = record_hash
            counts["retained_rows"] += 1
            kind, gold = record["parent_kind"], record["gold"]
            kept_kinds[kind] += 1
            event = ("native1" if gold == record["native1"] else
                     "native2" if gold == record["native2"] else
                     "x" if gold == record["x_token"] else "other")
            kept_events[event] += 1
            split_rows[record["split"]] += 1
            distinct.setdefault(record["split"], set()).add(record["canonical_question_id"])
    report = {
        "schema": "oov_tp8_exact_prefix_collection_v1",
        "runtime_contract_sha256": contract_sha,
        "source_manifest_sha256": source_sha,
        "draft_capture_sha256": sha256_file(draft_path),
        "target_capture_sha256": target_sha,
        "target_rows": target_count,
        "target_ambiguous_prefixes": len(ambiguous),
        "counts": dict(counts),
        "kept_parent_kinds": dict(kept_kinds),
        "kept_target_events": dict(kept_events),
        "excluded": dict(excluded),
        "split_rows": dict(split_rows),
        "split_questions": {split: len(qids) for split, qids in distinct.items()},
        "deployment_validated": False,
    }
    if counts["retained_rows"]:
        staged.rename(final)
        report["exact_rows_sha256"] = sha256_file(final)
    else:
        staged.unlink()
    with audit_path.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    if not counts["retained_rows"]:
        raise RuntimeError("No rows passed exact Target-prefix verification. See collection_audit.json.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--hot-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = join_capture(draft_path=args.draft, target_path=args.target, prompts_path=args.prompts,
                          contract_path=args.contract, hot_map_path=args.hot_map, output_dir=args.output)
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
