#!/usr/bin/env python3
"""Prepare the pinned eight-subset Red Hat workload with a local Llama tokenizer."""

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


DATASET_REPO = "RedHatAI/speculator_benchmarks"
DATASET_REVISION = "2ae86affa2cb97a972b7fc681dd51c04fbff083e"
SOURCE_ROWS = {
    "HumanEval": (164, "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2"),
    "math_reasoning": (80, "b73b686f1c585eef7db1306564688da8668ef633ff8e58d55949aac04db8c9b0"),
    "qa": (80, "17c7c2fb9f36abd16e3e2a8ab5487ae678e9762d96cb9736eec442cbaba9f506"),
    "question": (80, "9db8ea4678e058c26f21a9cc0bf79f95a9750e370cafb4f90a8e3b058c2a2aa4"),
    "rag": (80, "d1644a209c444e1dd7ff65bf3ce9e8a4daa5d609ac93cbf4f6bebc10400d8d91"),
    "summarization": (80, "e7d2945062f77aae25553f7f7531deefc358f5d0b87b9c623e6ba8ba555ff1be"),
    "translation": (80, "bd27352726c318b72a263158672b0ff0c15feb459ae735f723b27623a0ba595d"),
    "writing": (80, "9db8ea4678e058c26f21a9cc0bf79f95a9750e370cafb4f90a8e3b058c2a2aa4"),
}
SUBSETS = tuple(SOURCE_ROWS)
ARCHIVED_8B_DIGESTS = {
    "HumanEval": "9f49550258c63fe1b0b012e6d5ef6c13991446153bfa82d0d6e5c23ee1b12b7b",
    "math_reasoning": "db7f0dd5b2f6376580d6ff41c24a4a5998eaa1b72427cbd1166d39f1c0b81d70",
    "qa": "c6f56087ef969552adcc66a4744ee202bdd03470608dfed4024b47418dc7090c",
    "question": "5f14714ea07ee9f46875236958b64ccbe01a0db76ecf89dba50a07f4f7734cbd",
    "rag": "dfa2288490f7a6545da89b0a4f43b458a29f930fc0bfd4150edd030d86922965",
    "summarization": "0c624f5420672122654e3d725962750b9b5d3845812a878485c581989e918327",
    "translation": "f521d0efde3f44c44fc09d171638e20899e60c4479a148637b6170e2d36a8af6",
    "writing": "5f14714ea07ee9f46875236958b64ccbe01a0db76ecf89dba50a07f4f7734cbd",
}
SELECTED_PER_SUBSET = 80
REQUESTS_PER_SUBSET = 2 * SELECTED_PER_SUBSET
TARGET_VOCAB_SIZE = 128256
SOURCE_SIZE_LIMIT = 2_000_000


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(data):
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def token_digest(ids):
    return sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8"))


def frozen_source(source_dir, subset, download=False):
    filename = f"{subset}.jsonl"
    path = source_dir / filename
    expected_count, expected_sha = SOURCE_ROWS[subset]
    url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{filename}"
    downloading = not path.exists() and download
    if downloading:
        with urlopen(url, timeout=60) as response:
            payload = response.read(SOURCE_SIZE_LIMIT + 1)
        if len(payload) > SOURCE_SIZE_LIMIT:
            raise ValueError(f"Pinned download is unexpectedly large: {filename}")
    else:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Supply the pinned local file or use --download.")
        payload = path.read_bytes()
    if sha256(payload) != expected_sha:
        raise ValueError(f"Pinned source SHA-256 mismatch: {path}. Restore the file from {url}.")
    rows = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
    if len(rows) != expected_count:
        raise ValueError(f"Unexpected number of rows in {path}: {len(rows)}")
    selected = rows[:SELECTED_PER_SUBSET]
    if any(not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"]
           for row in selected):
        raise ValueError(f"Missing source prompt in the first {SELECTED_PER_SUBSET} rows of {path}")
    if downloading:
        write_frozen(path, payload)
    return [row["prompt"] for row in selected], {
        "filename": filename, "url": url, "source_sha256": expected_sha,
        "source_rows": expected_count, "selected_rows": len(selected),
    }


def tokenize_prompt(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_dict=False,
    )
    return ids["input_ids"] if isinstance(ids, Mapping) else ids


def validate_inputs(inputs, allow_tokenization_drift=False):
    if not isinstance(inputs, dict) or tuple(inputs) != SUBSETS:
        raise ValueError(f"Prepared data must contain these subsets in order: {', '.join(SUBSETS)}")
    summaries = {}
    drifted = []
    for subset, rows in inputs.items():
        if not isinstance(rows, list) or len(rows) != REQUESTS_PER_SUBSET:
            raise ValueError(f"{subset} requires {REQUESTS_PER_SUBSET} requests")
        all_ids = []
        for index, row in enumerate(rows):
            ids = row.get("input_ids") if isinstance(row, dict) else None
            if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < TARGET_VOCAB_SIZE
                                                         for i in ids):
                raise ValueError(f"Invalid input_ids in {subset} at row {index}")
            all_ids.append(ids)
        if all_ids[:SELECTED_PER_SUBSET] != all_ids[SELECTED_PER_SUBSET:]:
            raise ValueError(f"{subset} must repeat the first 80 token sequences in their original order")
        distinct = len({tuple(ids) for ids in all_ids})
        if distinct != SELECTED_PER_SUBSET:
            raise ValueError(f"Expected {SELECTED_PER_SUBSET} distinct token sequences in {subset}, got {distinct}")
        fingerprint = token_digest(all_ids)
        archival = ARCHIVED_8B_DIGESTS[subset]
        if fingerprint != archival:
            drifted.append(subset)
        summaries[subset] = {
            "requests": len(all_ids), "distinct_inputs": distinct,
            "min_input_tokens": min(map(len, all_ids)), "max_input_tokens": max(map(len, all_ids)),
            "input_ids_sha256": fingerprint, "archived_8b_input_ids_sha256": archival,
            "matches_archived_8b_inputs": fingerprint == archival,
        }
    if summaries["question"]["input_ids_sha256"] != summaries["writing"]["input_ids_sha256"]:
        raise ValueError("Pinned question and writing sources are identical and must tokenize identically")
    if drifted and not allow_tokenization_drift:
        raise ValueError(
            "Tokenized inputs differ from the archived 8B workload in: " + ", ".join(drifted)
            + ". Check the model path and local chat template. For the approved 70B tokenizer, "
              "use --allow-tokenization-drift to record a new workload fingerprint and use it for both arms."
        )
    return summaries


def prepare(source_dir, tokenizer, *, download=False, allow_tokenization_drift=False):
    inputs, sources = {}, {}
    for subset in SUBSETS:
        prompts, sources[subset] = frozen_source(source_dir, subset, download)
        selected = [{"input_ids": tokenize_prompt(tokenizer, prompt)} for prompt in prompts]
        inputs[subset] = selected + selected
    return inputs, sources, validate_inputs(inputs, allow_tokenization_drift)


def write_frozen(path, payload):
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite a different file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Directory with the eight pinned upstream JSONL files")
    parser.add_argument("--model", type=Path, help="Local target model directory with its tokenizer and chat template")
    parser.add_argument("--output", type=Path, help="Destination for the benchmark input_ids JSON and manifest")
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned JSONL files from Hugging Face")
    parser.add_argument("--check-prepared", type=Path, help="Validate a prepared JSON file without importing transformers")
    parser.add_argument("--allow-tokenization-drift", action="store_true",
                        help="Permit a different tokenization fingerprint from the historical 8B workload")
    args = parser.parse_args()
    if args.check_prepared:
        if args.source_dir or args.model or args.output or args.download:
            parser.error("--check-prepared is an alternative to --source-dir, --model and --output")
        payload = args.check_prepared.read_bytes()
        summary = validate_inputs(json.loads(payload), args.allow_tokenization_drift)
        print(json.dumps({"input_file_sha256": sha256(payload), "subsets": summary}, indent=2))
        return
    if not all((args.source_dir, args.model, args.output)):
        parser.error("Preparation requires --source-dir, --model and --output")
    if not args.model.is_dir():
        parser.error(f"--model must be a local tokenizer directory: {args.model}")

    import transformers

    model = args.model.resolve()
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    inputs, sources, summary = prepare(args.source_dir, tokenizer, download=args.download,
                                       allow_tokenization_drift=args.allow_tokenization_drift)
    payload = json_bytes(inputs)
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                       "special_tokens_map.json", "chat_template.jinja")
    manifest = {
        "schema_version": 1, "dataset": DATASET_REPO, "revision": DATASET_REVISION,
        "subset_order": list(SUBSETS), "request_count": len(SUBSETS) * REQUESTS_PER_SUBSET,
        "selection": "First 80 prompts in source order per subset; repeat those same 80 in sequence",
        "sources": sources, "subsets": summary, "input_file_sha256": sha256(payload),
        "tokenizer_directory_name": model.name, "transformers_version": transformers.__version__,
        "tokenizer_files_sha256": {name: sha256((model / name).read_bytes()) for name in tokenizer_files
                                   if (model / name).is_file()},
        "chat_template_sha256": sha256(json_bytes(tokenizer.chat_template)),
        "tokenizer_policy": "Local target chat template with one user message and add_generation_prompt=True",
        "identical_input_subsets": [["question", "writing"]],
    }
    manifest_path = args.output.with_name(f"{args.output.stem}.manifest.json")
    write_frozen(args.output, payload)
    write_frozen(manifest_path, json_bytes(manifest))
    print(json.dumps({"requests": manifest["request_count"], "input_file_sha256": manifest["input_file_sha256"],
                      "matches_archived_8b_inputs": all(row["matches_archived_8b_inputs"] for row in summary.values()),
                      "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
