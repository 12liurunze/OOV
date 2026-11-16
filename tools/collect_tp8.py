"""Collect labeled OOV classifier examples from one real eight-rank SGLang engine."""

import argparse
from collections.abc import Mapping
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration.candidates import ROOT_POLICY, TIE_POLICY, generator_sha256
from calibration.contracts import (
    CANDIDATE_PAIR_POLICY, DRAFT_KV_ANCESTRY, FEATURE_NAMES, FEATURE_SCHEMA, ROW_SCHEMA,
    hot_map_digest, prefix_hash, sha256_file,
)
from calibration.prepare_experiment import verify_contract
from stage_tp8_collection import stage_collection


CONTEXT_LENGTH = 2048
VOCAB_SIZE = 128256
DRAFT_VOCAB_SIZE = 32000
PROBE_ID = "cal70"
COLLECTION = "calibration_collect"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield obj


def text_field(row, field):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a nonempty string in {field}")
    return value


def chat_messages(row):
    if ("prompt" in row) == ("messages" in row):
        raise ValueError("Each question must supply exactly one of prompt and messages")
    if "prompt" in row:
        return [{"role": "user", "content": text_field(row, "prompt")}]
    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")):
            raise ValueError("Only plain Llama chat roles and text are accepted")
        text_field(message, "content")
    if messages[-1]["role"] != "user":
        raise ValueError("The final calibration message must have the user role")
    return messages


def checked_ids(value):
    if (not isinstance(value, list) or not value
            or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in value)):
        raise ValueError("Invalid Llama target token IDs")
    return value


def evaluation_prefixes(path):
    if path is None:
        return set()
    data = load(path)
    if not isinstance(data, dict) or not data:
        raise ValueError("--eval-inputs needs the prepared Red Hat subset mapping")
    fingerprints = set()
    for subset, prompts in data.items():
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"Empty or invalid evaluation subset {subset!r}")
        for prompt in prompts:
            fingerprints.add(prefix_hash(checked_ids(prompt["input_ids"])))
    return fingerprints


def prepare(inputs, tokenizer, eval_hashes, limit, max_new_tokens):
    seen_ids, seen_content, prepared = set(), set(), []
    for row in rows(inputs):
        qid, source = (text_field(row, field) for field in ("canonical_question_id", "source"))
        if qid != qid.strip() or qid in seen_ids:
            raise ValueError(f"Question ID is duplicated or padded: {qid!r}")
        messages = chat_messages(row)
        output = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        ids = checked_ids(output["input_ids"] if isinstance(output, Mapping) else output)
        # Leave the fixed Draft budget available at the end of the experiment.
        if len(ids) + max_new_tokens + 8 > CONTEXT_LENGTH:
            raise ValueError(f"Question {qid!r} exceeds the configured context budget")
        content = prefix_hash(ids)
        if content in seen_content or content in eval_hashes:
            raise ValueError(f"Question {qid!r} duplicates calibration or evaluation token IDs")
        seen_ids.add(qid)
        seen_content.add(content)
        prepared.append(dict(
            request_id=f"{PROBE_ID}~{len(prepared):06d}",
            canonical_question_id=qid, source=source, prompt_ids=ids,
            question_content_sha256=content,
        ))
        if len(prepared) == limit:
            break
    if not prepared:
        raise ValueError("Calibration input contained no prompts")
    return prepared


def model_identity_helpers(runtime):
    path = runtime / "sglang/srt/speculative/repair_model_identity.py"
    spec = importlib.util.spec_from_file_location("oov_collection_model_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_stamps(identities, target, draft):
    result = {}
    for directory, label, token_files in (
        (target, "LlamaForCausalLM", ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                                    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")),
        (draft, "LlamaForCausalLMEagle3", ()),
    ):
        names = set(identities[label]["shards_bytes"])
        names.add("config.json")
        names.update(name for name in (*token_files, "model.safetensors.index.json",
                                       "generation_config.json") if (directory / name).is_file())
        result[label] = {name: list(stamp(directory / name)) for name in sorted(names)}
    return result


def stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def runtime_contract(runtime, target, draft, alignment):
    helper = model_identity_helpers(runtime)
    target_id, target_sha = helper.model_identity(target)
    draft_id, draft_sha = helper.model_identity(draft)
    hot_map = load(alignment / "hot_map.json")
    if (not isinstance(hot_map, list) or len(hot_map) != DRAFT_VOCAB_SIZE
            or any(type(x) is not int or not 0 <= x < VOCAB_SIZE for x in hot_map)
            or len(set(hot_map)) != DRAFT_VOCAB_SIZE):
        raise ValueError("Expected a unique, ordered 32000-token Draft-to-Target hot map")
    contract = dict(
        target_model_id=target_id, target_weights_sha256=target_sha,
        draft_model_id=draft_id, draft_weights_sha256=draft_sha,
        tokenizer_sha256=helper.tokenizer_identity(target),
        hot_map_sha256=hot_map_digest(hot_map),
        alignment_sha256=sha256_file(alignment / "alignment.pt"),
        candidate_generator_sha256=generator_sha256(), candidate_policy="recent_distinct_v1",
        capacity=64, root_policy=ROOT_POLICY, tie_policy=TIE_POLICY,
        readout_backend="fused_cached_v1", readout_numeric_version="cached_WA_bf16_v1",
        hidden_size=8192, target_vocab_size=VOCAB_SIZE, draft_vocab_size=DRAFT_VOCAB_SIZE,
        dtype="bfloat16", tp_size=8, max_steps=7, sampling_mode="greedy",
        feature_schema=FEATURE_SCHEMA, feature_names=list(FEATURE_NAMES),
        feature_dim=len(FEATURE_NAMES), label_schema=ROW_SCHEMA,
        score_mode="unified_target_event_v2", candidate_pair_policy=CANDIDATE_PAIR_POLICY,
        draft_kv_ancestry=DRAFT_KV_ANCESTRY,
    )
    verify_contract(contract, alignment)
    generator = runtime / "sglang/srt/speculative/repair_calibration/candidates.py"
    if sha256_file(generator) != contract["candidate_generator_sha256"]:
        raise ValueError("Runtime and trainer use different candidate generation source")
    return contract


def configuration(plan):
    from benchmark import configuration as benchmark_config

    cfg = benchmark_config(plan, "repair")
    cfg.update(
        disable_cuda_graph=True, disable_decode_cuda_graph=True, disable_prefill_cuda_graph=True,
        cuda_graph_backend_decode="disabled", cuda_graph_backend_prefill="disabled",
        cuda_graph_bs_decode=[], cuda_graph_max_bs_decode=None,
        random_seed=plan["seed"],
    )
    return cfg


def check_engine(info, cfg, runtime):
    from benchmark import repair_snapshots

    current = info.get("server_args", info)
    required = ("tp_size", "dp_size", "pp_size", "dtype", "disable_cuda_graph",
                "disable_decode_cuda_graph", "disable_prefill_cuda_graph",
                "disable_overlap_schedule", "model_path", "speculative_draft_model_path",
                "speculative_num_steps", "speculative_num_draft_tokens",
                "speculative_eagle_topk", "speculative_adaptive_tree")
    if any(current.get(key) != cfg[key] for key in required):
        raise RuntimeError("Effective SGLang launch settings differ from the eager TP8 plan")
    snapshots = list(repair_snapshots(info))
    expected_sha = sha256_file(runtime / "sglang/srt/speculative/context_repair.py")
    if not snapshots or any(
        row.get("mode") != COLLECTION or row.get("tp_size") != 8
        or row.get("score_mode") != COLLECTION
        or row.get("backend") != "fused_cached_v1"
        or row.get("numeric_version") != "cached_WA_bf16_v1"
        or row.get("cached_weight_projection") is not True
        or row.get("calibration_sha256") is not None
        or row.get("correctness_barrier_per_round") is not True
        or row.get("module_sha256") != expected_sha
        or row.get("gpu", {}).get("tp_decision_mismatch_calls", 0)
        for row in snapshots
    ):
        raise RuntimeError("The expected untrained calibration runtime is not active")
    return snapshots


def worker(plan_path):
    from benchmark import local_model_check

    if os.environ.get("OOV_TP8_COLLECTION") != "1":
        raise RuntimeError("Calibration worker must be started by the collection launcher")
    plan = load(plan_path)
    output, runtime = plan_path.parent, Path(plan["runtime"])
    if Path(os.environ["OOV_TP8_OUTPUT"]) != output:
        raise RuntimeError("Unexpected worker output directory")
    target, draft = Path(plan["target"]), Path(plan["draft"])
    if checkpoint_stamps(local_model_check(target, draft), target, draft) != plan["checkpoint_stamps"]:
        raise RuntimeError("Checkpoint or tokenizer files changed before launch")
    manifest = output / "collection_inputs.jsonl"
    if sha256_file(manifest) != plan["manifest_sha256"]:
        raise RuntimeError("Prepared prompt manifest changed while launching")
    prompts = list(rows(manifest))

    import sglang
    import torch

    if (Path(sglang.__file__).resolve() != runtime / "sglang/__init__.py"
            or torch.cuda.device_count() != 8
            or any(torch.cuda.get_device_capability(i) != (8, 0) for i in range(8))):
        raise RuntimeError("Expected the staged calibration source and eight visible A100 GPUs")
    cfg = configuration(plan)
    engine = sglang.Engine(**cfg)
    try:
        info = engine.get_server_info()
        before = check_engine(info, cfg, runtime)
        capacity = info["max_total_num_tokens"]
        needed = plan["bs"] * (max(len(row["prompt_ids"]) for row in prompts)
                               + plan["max_new_tokens"] + 32)
        if capacity < needed:
            raise RuntimeError(f"KV capacity {capacity} below conservative need {needed}")
        save(output / "engine_info.json", {"config": cfg, "capacity": capacity, "initial_repair": before})
        finished = []
        for start in range(0, len(prompts), plan["bs"]):
            batch = prompts[start:start + plan["bs"]]
            request_ids = [row["request_id"] for row in batch]
            result = engine.generate(
                input_ids=[row["prompt_ids"] for row in batch],
                rid=request_ids,
                sampling_params=dict(temperature=0, max_new_tokens=plan["max_new_tokens"], ignore_eos=False),
            )
            if not isinstance(result, list) or len(result) != len(request_ids):
                raise RuntimeError("Incorrect number of completed calibration requests")
            for rid, item in zip(request_ids, result, strict=True):
                meta = item["meta_info"]
                count = meta["completion_tokens"]
                finish = meta["finish_reason"]["type"]
                if (meta["id"] != rid or not 0 < count <= plan["max_new_tokens"]
                        or finish not in ("stop", "length") or meta.get("num_retractions") != 0):
                    raise RuntimeError(f"Incomplete or retracted calibration request {rid}")
                finished.append({"request_id": rid, "generated_tokens": count,
                                 "finish": finish, "verify_rounds": meta.get("spec_verify_ct", 0)})
            print(f"Completed {len(finished)}/{len(prompts)} calibration requests", flush=True)
        final = check_engine(engine.get_server_info(), cfg, runtime)
        save(output / "worker_receipt.json", {"requests": finished, "final_repair": final})
    finally:
        engine.shutdown()


def child_environment(plan, output):
    filtered = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("EAGLE_CONTEXT_", "OOV_TP8_", "PROFILE_", "SGLANG_", "REDHAT_"))
        and key not in ("CUDA_LAUNCH_BLOCKING", "RANK", "WORLD_SIZE", "LOCAL_RANK",
                        "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    }
    filtered.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=",".join(d["uuid"] for d in plan["devices"]),
        PYTHONPATH=os.pathsep.join((str(ROOT / "tools/tp8_collection_hook"), plan["runtime"], str(ROOT))),
        PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        SGLANG_ENABLE_METADATA_GLUE_GRAPH="0", LLAMA_TP_COMPACT_MODE="off",
        EAGLE_CONTEXT_REPAIR_MODE=COLLECTION,
        EAGLE_CONTEXT_REPAIR_ROOT=plan["alignment_dir"],
        EAGLE_CONTEXT_READOUT_BACKEND="fused_cached_v1",
        EAGLE_CONTEXT_CACHE_WEIGHTS="1",
        EAGLE_CONTEXT_POOL_POLICY="recent_distinct_v1",
        OOV_TP8_COLLECTION="1",
        OOV_TP8_OUTPUT=str(output),
        OOV_TP8_ALIGNMENT_SHA256=plan["alignment_sha256"],
        OOV_TP8_HOT_MAP_SHA256=plan["hot_map_sha256"],
        OOV_TP8_DRAFT_JSONL=str(output / "draft.jsonl"),
        OOV_TP8_TARGET_JSONL=str(output / "target.jsonl"),
        OOV_TP8_PROBE_RATE=str(plan["x_probe_rate"]),
        OOV_TP8_PROBE_SEED=str(plan["seed"]),
    )
    return filtered


def run_worker(plan_path, plan):
    from benchmark import foreign_processes, wait_until_released

    output = plan_path.parent
    if busy := foreign_processes(plan["devices"]):
        raise RuntimeError(f"Selected A100 GPUs have existing compute processes: {busy}")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--internal-worker", str(plan_path)]
    with (output / "run.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT,
            env=child_environment(plan, output),
        )
        try:
            while proc.poll() is None:
                if busy := foreign_processes(plan["devices"], proc.pid):
                    raise RuntimeError(f"Another workload entered the selected GPUs: {busy}")
                time.sleep(2)
            if not proc.returncode:  # TP ranks may still be exiting; they are not a foreign workload
                wait_until_released(plan["devices"], proc.pid)
            if proc.returncode or foreign_processes(plan["devices"]):
                raise RuntimeError(f"Collection worker failed or lost isolation. Inspect {output / 'run.log'}")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path,
                        help="Local calibration JSONL. Each row needs canonical_question_id, source and prompt or messages.")
    parser.add_argument("--eval-inputs", type=Path, help="Prepared held-out Red Hat JSON for exact token deduplication")
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory for this collection")
    parser.add_argument("--runtime", default=ROOT / "generated/runtime", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--bs", default=2, type=int, choices=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--limit", type=int, help="Optional prompt limit for the first pilot")
    parser.add_argument("--max-new-tokens", default=96, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--x-probe-rate", default=0.2, type=float)
    parser.add_argument("--mem-fraction", default=0.78, type=float)
    args = parser.parse_args(argv)
    if (not 8 <= args.max_new_tokens <= 512 or not 0 <= args.x_probe_rate <= 1
            or not 0 < args.mem_fraction < 1 or not 0 <= args.seed < 2**31
            or args.limit is not None and args.limit < args.bs):
        parser.error("Invalid output budget, probe rate, memory fraction, seed or prompt limit")
    if sys.platform != "linux":
        parser.error("Running the TP8 collector requires a Linux host with eight A100 GPUs")
    if os.environ.get("WORLD_SIZE", "1") != "1" or "LOCAL_RANK" in os.environ:
        parser.error("Run this command once with Python. SGLang starts its own eight worker ranks.")

    from benchmark import foreign_processes, gpu_devices, local_model_check
    from calibration.tp8_join import join_capture
    from transformers import AutoTokenizer

    output = args.output.resolve()
    if output.exists():
        parser.error("--output must identify a fresh directory; retain partial runs for inspection")
    target, draft, alignment = (p.resolve(strict=True) for p in
                                (args.target, args.draft, args.alignment_dir))
    identities = local_model_check(target, draft)
    devices = gpu_devices(tuple(value.strip() for value in args.gpus.split(",")))
    if busy := foreign_processes(devices):
        raise RuntimeError(f"Selected A100 GPUs have existing compute processes: {busy}")
    checkpoint_stats = checkpoint_stamps(identities, target, draft)
    runtime = stage_collection(args.runtime.resolve())
    contract = runtime_contract(runtime, target, draft, alignment)
    tokenizer = AutoTokenizer.from_pretrained(str(target), local_files_only=True)
    prompt_source_hash = sha256_file(args.prompts)
    eval_source_hash = sha256_file(args.eval_inputs) if args.eval_inputs else None
    prompts = prepare(args.prompts, tokenizer, evaluation_prefixes(args.eval_inputs),
                      args.limit, args.max_new_tokens)
    if (sha256_file(args.prompts) != prompt_source_hash
            or args.eval_inputs and sha256_file(args.eval_inputs) != eval_source_hash):
        raise RuntimeError("Calibration or held-out input changed while preparing prompts")
    if len(prompts) < args.bs:
        raise ValueError("Fewer prompts than requested batch size")

    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "collection_inputs.jsonl"
    with manifest.open("x", encoding="utf-8") as handle:
        for row in prompts:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    contract_path = output / "runtime_contract.json"
    save(contract_path, contract)
    plan = dict(
        target=str(target), draft=str(draft), runtime=str(runtime), alignment_dir=str(alignment),
        devices=devices, checkpoint_stamps=checkpoint_stats,
        manifest_sha256=sha256_file(manifest), prompt_source_sha256=prompt_source_hash,
        eval_source_sha256=eval_source_hash, contract_file_sha256=sha256_file(contract_path),
        alignment_sha256=contract["alignment_sha256"], hot_map_sha256=contract["hot_map_sha256"],
        bs=args.bs, seed=args.seed, max_new_tokens=args.max_new_tokens, x_probe_rate=args.x_probe_rate,
        mem_fraction=args.mem_fraction,
    )
    plan_path = output / "collection_plan.json"
    save(plan_path, plan)
    print(f"Starting {len(prompts)} prompts. Follow {output / 'run.log'}", flush=True)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    run_worker(plan_path, plan)
    if (checkpoint_stamps(identities, target, draft) != checkpoint_stats
            or sha256_file(contract_path) != plan["contract_file_sha256"]
            or sha256_file(manifest) != plan["manifest_sha256"]):
        raise RuntimeError("A checkpoint or sealed collection record changed during launch")
    verify_contract(contract, alignment)
    report = join_capture(
        draft_path=output / "draft.jsonl", target_path=output / "target.jsonl",
        prompts_path=manifest, contract_path=contract_path,
        hot_map_path=alignment / "hot_map.json", output_dir=output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--internal-worker":
        worker(Path(sys.argv[2]).resolve())
    else:
        main()
