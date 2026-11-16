"""Reconstruct the frozen 70B calibration prompts from pinned public datasets."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


MANIFEST = Path(__file__).resolve().parents[1] / "calibration" / "prompts_manifest.json"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_source(path, receipt):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != receipt["bytes"] or digest != receipt["sha256"]:
        raise ValueError(f"Pinned source size or SHA-256 differs: {path}")


def source_path(cache, receipt, offline):
    path = cache / receipt["cache_path"]
    if path.is_file():
        verify_source(path, receipt)
        return path
    if offline:
        raise FileNotFoundError(f"Missing offline source: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as pending:
        pending_path = Path(pending.name)
    try:
        with urlopen(receipt["url"], timeout=120) as response, pending_path.open("wb") as output:
            remaining = receipt["bytes"]
            while block := response.read(min(1024 * 1024, remaining + 1)):
                remaining -= len(block)
                if remaining < 0:
                    raise ValueError(f"Source exceeds its pinned size: {receipt['url']}")
                output.write(block)
        verify_source(pending_path, receipt)
        pending_path.replace(path)
    finally:
        pending_path.unlink(missing_ok=True)
    return path


def read_source(path, columns):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Parquet sources require pyarrow; install it with python -m pip install pyarrow") from error
    return parquet.read_table(path, columns=columns).to_pylist()


def source_prompt(source, receipt, index, row):
    if source == "bigcodebench":
        return str(row["task_id"]), f"Complete the following Python function. Return only Python code.\n\n{row['complete_prompt'].strip()}"
    if source == "hendrycks_math":
        return f"{receipt['category']}/{index}", f"Solve the following problem. Show the reasoning and give the final answer.\n\n{row['problem'].strip()}"
    if source == "dolly":
        instruction = row["instruction"].strip()
        context = row["context"].strip()
        return str(index), f"{instruction}\n\nContext:\n{context}" if context else instruction
    if source == "xsum":
        return str(row["id"]), f"Summarize the following article in one sentence.\n\n{row['document'].strip()}"
    raise ValueError(f"Unknown source: {source}")


def collect_prompts(manifest, cache, offline):
    requested = {
        (row["source"], row["source_row_id"])
        for file in manifest["files"].values()
        for row in file["row_provenance"]
    }
    columns = {
        "bigcodebench": ["task_id", "complete_prompt"],
        "hendrycks_math": ["problem"],
        "dolly": ["instruction", "context"],
        "xsum": ["id", "document"],
    }
    prompts = {}
    for source, sources in manifest["source_receipts"].items():
        for receipt in sources:
            rows = read_source(source_path(cache, receipt, offline), columns[source])
            if len(rows) != receipt["rows"]:
                raise ValueError(f"Unexpected source row count: {receipt['cache_path']}")
            for index, row in enumerate(rows):
                row_id, prompt = source_prompt(source, receipt, index, row)
                key = (source, row_id)
                if key in requested:
                    if key in prompts:
                        raise ValueError(f"Ambiguous source row: {key}")
                    prompts[key] = prompt.strip()
    if missing := requested - prompts.keys():
        raise ValueError(f"Frozen source rows are missing: {sorted(missing)}")
    return prompts


def render(file, prompts):
    lines = []
    for row in file["row_provenance"]:
        prompt = prompts[(row["source"], row["source_row_id"])]
        if sha256(prompt.encode("utf-8")) != row["question_content_sha256"]:
            raise ValueError(f"Frozen prompt differs: {row['canonical_question_id']}")
        payload = {"canonical_question_id": row["canonical_question_id"], "source": row["source"], "prompt": prompt}
        lines.append(json.dumps(payload, ensure_ascii=False) + "\n")
    return "".join(lines).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Cache for the pinned source dataset files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the two calibration JSONL files")
    parser.add_argument("--offline", action="store_true", help="Use only verified files already in the cache")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_bytes())
    prompts = collect_prompts(manifest, args.cache_dir, args.offline)
    results = {}
    for filename, file in manifest["files"].items():
        data = render(file, prompts)
        if len(data.splitlines()) != file["rows"] or sha256(data) != file["sha256"]:
            raise ValueError(f"Frozen output differs: {filename}")
        output = args.output_dir / filename
        if output.exists() and output.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite different contents: {output}")
        results[output] = data
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for output, data in results.items():
        if not output.exists():
            with output.open("xb") as stream:
                stream.write(data)
        print(f"{output}: SHA-256 {sha256(data)}")


if __name__ == "__main__":
    main()
