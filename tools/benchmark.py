"""Single-node TP8 Llama-3.3-70B + SpecForge EAGLE3 benchmark."""

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from install_runtime import verify_installed
from stage_70b import stage as stage_repair


SUBSETS = ("HumanEval", "math_reasoning", "qa", "question", "rag",
           "summarization", "translation", "writing")
ROOT = Path(__file__).resolve().parents[1]
METHODS = ("eagle728", "repair")
SCOPE = {"tp_size": 8, "hidden_size": 8192, "target_vocab_size": 128256,
         "draft_vocab_size": 32000, "dtype": "bfloat16",
         "readout_backend": "fused_cached_v1", "readout_numeric_version": "cached_WA_bf16_v1",
         "capacity": 64, "candidate_policy": "recent_distinct_v1",
         "max_steps": 7, "sampling_mode": "greedy"}
CAP = 512


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, obj):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def local_model_check(target, draft):
    if not any((target / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise ValueError("The local 70B Target tokenizer is missing")
    identity = {}
    for directory, vocab_key, architecture, vocab in (
        (target, "vocab_size", "LlamaForCausalLM", 128256),
        (draft, "draft_vocab_size", "LlamaForCausalLMEagle3", 32000),
    ):
        cfg = load(directory / "config.json")
        if (cfg.get("hidden_size") != 8192 or cfg.get(vocab_key) != vocab
                or architecture not in cfg.get("architectures", ())):
            raise ValueError(f"Unexpected 70B checkpoint configuration under {directory}")
        if cfg.get("quantization_config"):
            raise ValueError("This experiment uses unquantized BF16 checkpoints")
        index = directory / "model.safetensors.index.json"
        weights = set(load(index)["weight_map"].values()) if index.is_file() else {"model.safetensors"}
        if (not weights or any(Path(name).name != name or not name.endswith(".safetensors")
                               or not (directory / name).is_file() for name in weights)):
            raise ValueError(f"Missing local safetensors checkpoint members under {directory}")
        identity[architecture] = {
            "config_sha256": file_sha(directory / "config.json"),
            "index_sha256": file_sha(index) if index.is_file() else None,
            "shards_bytes": {name: (directory / name).stat().st_size for name in sorted(weights)},
        }
    return identity


def asset_receipt(directory):
    if directory is None:
        raise ValueError("70B repair requires --assets with externally fitted TP8 parameters. See agent.md.")
    paths = {n: directory / n for n in ("alignment.pt", "runtime_contract.json", "calibration_asset.json")}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Missing TP8 repair file {path}")
    contract = load(paths["runtime_contract.json"])
    if any(contract.get(k) != value for k, value in SCOPE.items()):
        raise ValueError("Asset contract is incompatible with 70B TP8, BF16 and 32K Draft")
    asset = load(paths["calibration_asset.json"])
    if asset.get("metadata") != contract:
        raise ValueError("Classifier metadata differs from the runtime contract")
    hashes = {n: file_sha(path) for n, path in paths.items()}
    if hashes["alignment.pt"] != contract.get("alignment_sha256"):
        raise ValueError("Alignment digest differs from the calibration contract")
    return hashes


def request_inputs(path, limit, bs):
    data = load(path)
    if set(data) != set(SUBSETS):
        raise ValueError("Expected the eight Red Hat subsets prepared by tools/prepare_inputs.py")
    result = {}
    for name in SUBSETS:
        entries = data[name]
        if len(entries) != 160:
            raise ValueError(f"Expected 160 requests for {name}")
        ids = [row["input_ids"] for row in entries]
        if ids[:80] != ids[80:] or len({tuple(r) for r in ids[:80]}) != 80:
            raise ValueError(f"Expected one ordered repetition of the 80 unique prompts for {name}")
        if any(not row or len(row) + CAP > 2048
               or any(type(x) is not int or not 0 <= x < 128256 for x in row) for row in ids):
            raise ValueError(f"Invalid token IDs or context length in {name}")
        result[name] = ids[:limit]
    if result["question"] != result["writing"]:
        raise ValueError("Expected the shared question/writing input in the frozen Red Hat protocol")
    if bs > limit:
        raise ValueError("--limit is smaller than --bs")
    return result


def smi(args):
    return subprocess.check_output(["nvidia-smi", *args], text=True, timeout=20).splitlines()


def gpu_devices(ids):
    inventory = {}
    for row in smi(["--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"]):
        index, uuid, name, memory = (field.strip() for field in row.split(","))
        inventory[index] = {"index": index, "uuid": uuid, "name": name, "memory_mib": int(memory)}
    if len(ids) != 8 or len(set(ids)) != 8 or any(i not in inventory for i in ids):
        raise ValueError("--gpus must identify 8 distinct physical nvidia-smi GPU indexes")
    devices = [inventory[i] for i in ids]
    if any("A100" not in d["name"] for d in devices):
        raise ValueError("This release's TP8 protocol is for NVIDIA A100")
    return devices


def gpu_processes(devices):
    uuids = {d["uuid"] for d in devices}
    rows = smi(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
    pairs = (tuple(field.strip() for field in row.split(",")[:2]) for row in rows)
    return [(uuid, int(pid)) for uuid, pid in pairs if uuid in uuids]


def foreign_processes(devices, own_pid=None):
    return [(uuid, pid) for uuid, pid in gpu_processes(devices) if not belongs_to(pid, own_pid)]


def belongs_to(pid, root):
    """True if pid descends from root or still sits in the session root leads.

    Workers start with start_new_session=True. During engine shutdown a TP rank can outlive its
    parent and be reparented to init; it keeps the worker's session, which the parent walk misses."""
    if root is None:
        return False
    try:
        if os.getsid(pid) == root:
            return True
    except OSError:
        pass
    while pid > 1:
        if pid == root:
            return True
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            pid = int(fields[1])
        except (OSError, IndexError, ValueError):
            return False
    return False


def wait_until_released(devices, root, timeout=120):
    """Wait until no process of the worker's session holds the selected GPUs."""
    deadline = time.monotonic() + timeout
    while own := [(uuid, pid) for uuid, pid in gpu_processes(devices) if belongs_to(pid, root)]:
        if time.monotonic() > deadline:
            raise RuntimeError(f"Worker processes still hold the selected GPUs after {timeout} s: {own}")
        time.sleep(2)


def configuration(plan, method):
    bs = plan["bs"]
    repair = method == "repair"
    return dict(
        model_path=plan["target"], speculative_draft_model_path=plan["draft"],
        trust_remote_code=False, tp_size=8, dp_size=1, pp_size=1, dcp_size=1, attn_cp_size=1,
        nnodes=1, node_rank=0, dtype="bfloat16", max_running_requests=bs, context_length=2048,
        mem_fraction_static=plan["mem_fraction"],
        attention_backend="triton", sampling_backend="pytorch", disable_radix_cache=False,
        speculative_algorithm="EAGLE3", speculative_num_steps=7, speculative_num_draft_tokens=8,
        speculative_eagle_topk=2, speculative_adaptive_tree=repair,
        speculative_adaptive_confidence="top1", speculative_adaptive_threshold=0.5 if repair else 0.0,
        speculative_use_rejection_sampling=False,
        cuda_graph_backend_decode="full", cuda_graph_backend_prefill="disabled",
        disable_cuda_graph=False, disable_decode_cuda_graph=False, disable_prefill_cuda_graph=True,
        cuda_graph_bs_decode=[n for n in (1, 2, 4, 8, 16, 24, 32) if n <= bs],
        cuda_graph_max_bs_decode=bs, enable_profile_cuda_graph=False, enable_dp_attention=False,
        random_seed=42, schedule_conservativeness=0.15, chunked_prefill_size=4096,
        max_prefill_tokens=4096, max_total_tokens=66048, log_level="info",
        enable_mixed_chunk=False, prefill_decode_interval=0, enable_prefill_delayer=False,
        min_free_slots_delay=1, schedule_policy="fcfs", disable_overlap_schedule=True,
    )


def repair_snapshots(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "context_repair" and isinstance(child, dict):
                yield child
            else:
                yield from repair_snapshots(child)
    elif isinstance(value, list):
        for child in value:
            yield from repair_snapshots(child)


def verify_runtime(info, cfg, plan, runtime, method):
    current = info.get("server_args", info)
    fixed = ("tp_size", "model_path", "speculative_draft_model_path", "max_running_requests",
             "disable_radix_cache", "disable_overlap_schedule", "disable_cuda_graph",
             "disable_decode_cuda_graph", "speculative_algorithm", "speculative_num_steps",
             "speculative_num_draft_tokens", "speculative_eagle_topk", "speculative_adaptive_tree")
    if any(current.get(key) != cfg[key] for key in fixed):
        raise RuntimeError("Effective SGLang launch settings differ from the plan")
    rows = list(repair_snapshots(info))
    if method == "eagle728":
        if rows:
            raise RuntimeError("Native EAGLE unexpectedly started a repair worker")
        return
    expected_module = runtime / "sglang/srt/speculative/context_repair.py"
    if not rows or any(
        row.get("mode") != "trained_experiment" or row.get("score_mode") != "legacy_mixed_v1"
        or row.get("tp_size") != 8 or row.get("backend") != "fused_cached_v1"
        or row.get("cached_weight_projection") is not True
        or row.get("candidate_pair_policy") != "preserve_native1_v1"
        or row.get("calibration_sha256") != plan["asset_sha256"]["calibration_asset.json"]
        or row.get("module_sha256") != file_sha(expected_module)
        or row.get("gpu", {}).get("tp_decision_mismatch_calls", 0)
        for row in rows
    ):
        raise RuntimeError("70B repair activation, contract or TP agreement check failed")


def drain(engine):
    for _ in range(100):
        if engine.flush_cache().success:
            return
        time.sleep(0.2)
    raise RuntimeError("SGLang did not return to an idle state after generation")


def response_rows(outputs, request_ids, repair):
    if len(outputs) != len(request_ids):
        raise RuntimeError("Wrong number of completed requests")
    rows = []
    for item, rid in zip(outputs, request_ids, strict=True):
        ids, meta = item["output_ids"], item["meta_info"]
        finish = meta["finish_reason"]
        accepted, proposed, rounds = (meta.get(k, 0) for k in (
            "spec_num_correct_drafts", "spec_num_proposed_drafts", "spec_verify_ct"))
        if (meta["id"] != rid or not 0 < len(ids) <= CAP or len(ids) != meta["completion_tokens"]
                or finish.get("type") not in ("stop", "length") or meta.get("num_retractions") != 0
                or proposed != 7 * rounds or not 0 <= accepted <= proposed):
            raise RuntimeError(f"Invalid generated output or verification accounting for {rid}")
        provenance = meta.get("spec_emitted_audit")
        if repair and (not provenance or not provenance.get("complete")
                       or provenance.get("audited_rounds") != rounds
                       or provenance["emitted_drafts"] + provenance["discarded_drafts"] != accepted):
            raise RuntimeError(f"Missing repair provenance for {rid}")
        rows.append(dict(rid=rid, output_tokens=len(ids), output_hash=digest(ids), finish=finish,
                         accepted=accepted, proposed=proposed, rounds=rounds, provenance=provenance))
    return rows


def measured(name, inputs, rows, elapsed, trace, bs):
    with trace.open(encoding="utf-8") as handle:
        steps = [json.loads(line) for line in handle]
    decode = [row for row in steps if row["phase"] == "decode"]
    hist = Counter(step["bs"] for step in decode)
    full = [step for step in decode if step["bs"] == bs]
    tokens = sum(r["output_tokens"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    proposed = sum(r["proposed"] for r in rows)
    rounds = sum(r["rounds"] for r in rows)
    if (not full or not rounds or sum(s["bs"] for s in decode) != rounds
            or sum(s["emitted"] for s in decode) + len(rows) != tokens
            or sum(s["accepted"] for s in decode) != accepted
            or any(not step["graph"] for step in full)):
        raise RuntimeError(f"{name} did not reach full Decode Graph BS or trace accounting failed")
    return dict(subset=name, requests=len(rows), output_tokens=tokens, seconds=elapsed,
                throughput=tokens / elapsed, accepted=accepted, proposed=proposed,
                verify_rounds=rounds, acceptance_rate=accepted / proposed,
                acceptance_length=accepted / rounds,
                cap_hits=sum(r["finish"]["type"] == "length" for r in rows),
                input_ids_sha256=digest(inputs), output_records_sha256=digest(rows),
                decode_bs_histogram=dict(hist), full_bs_graph_steps=len(full),
                emitted_oov_drafts=sum((r["provenance"] or {}).get("emitted_context_oov_drafts") or 0
                                       for r in rows) if rows[0]["provenance"] and rows[0]["provenance"].get("complete") else None)


def worker(plan_path, method, directory):
    plan = load(plan_path)
    if local_model_check(Path(plan["target"]), Path(plan["draft"])) != plan["checkpoint_receipts"]:
        raise RuntimeError("Model configuration, index or weight shard lengths changed")
    data_path = Path(plan["inputs"])
    if file_sha(data_path) != plan["input_sha256"]:
        raise RuntimeError("Prepared input file changed during the campaign")
    inputs = request_inputs(data_path, plan["limit"], plan["bs"])
    runtime = Path(plan["runtime"]) / ("repair70b" if method == "repair" else "native")
    if method == "repair":
        path = Path(plan["assets"])
        if asset_receipt(path) != plan["asset_sha256"]:
            raise RuntimeError("Calibration files changed during the campaign")
        os.environ.update(
            EAGLE_CONTEXT_REPAIR_ROOT=str(path),
            EAGLE_CONTEXT_RUNTIME_CONTRACT=str(path / "runtime_contract.json"),
            EAGLE_CONTEXT_REPAIR_MODE="trained_experiment",
            EAGLE_CONTEXT_READOUT_BACKEND="fused_cached_v1",
            EAGLE_CONTEXT_CACHE_WEIGHTS="1",
            EAGLE_CONTEXT_POOL_POLICY="recent_distinct_v1",
            EAGLE_CONTEXT_TREE_SCORE_MODE="legacy_mixed_v1",
            EAGLE_CONTEXT_EXPERIMENT_ASSET_SHA256=plan["asset_sha256"]["calibration_asset.json"],
        )
    import sglang
    import torch
    if (Path(sglang.__file__).resolve() != runtime / "sglang/__init__.py"
            or torch.cuda.device_count() != 8
            or any(torch.cuda.get_device_capability(i) != (8, 0) for i in range(8))):
        raise RuntimeError("Wrong SGLang source or CUDA visibility for eight A100 GPUs")
    cfg = configuration(plan, method)
    engine = sglang.Engine(**cfg)
    try:
        info = engine.get_server_info()
        verify_runtime(info, cfg, plan, runtime, method)
        capacity = info["max_total_num_tokens"]
        needed = plan["bs"] * (max(len(row) for rows in inputs.values() for row in rows) + CAP + 32)
        if capacity < needed:
            raise RuntimeError(f"Insufficient KV capacity for the planned BS: {capacity} < {needed}")
        save(directory / "capacity.json", {"available": capacity, "conservative_needed": needed})
        save(directory / "config.json", cfg)
        warm = inputs[SUBSETS[0]][:plan["bs"]]
        engine.generate(input_ids=warm, sampling_params=dict(temperature=0, max_new_tokens=24, ignore_eos=True))
        drain(engine)
        metrics = {}
        for name, batch in inputs.items():
            label = "measure_" + name
            prime = list(map(list, dict.fromkeys(map(tuple, batch))))
            prime_outputs = engine.generate(input_ids=prime, sampling_params=dict(temperature=0, max_new_tokens=1))
            if len(prime_outputs) != len(prime) or any(len(r["output_ids"]) != 1 for r in prime_outputs):
                raise RuntimeError(f"Radix priming failed for {name}")
            ids = [f"{label}~{j:03d}" for j in range(len(batch))]
            tick = time.perf_counter()
            outputs = engine.generate(input_ids=batch, rid=ids,
                                      sampling_params=dict(temperature=0, max_new_tokens=CAP, ignore_eos=False))
            elapsed = time.perf_counter() - tick
            drain(engine)
            rows = response_rows(outputs, ids, method == "repair")
            metrics[name] = measured(name, batch, rows, elapsed, directory / (label + ".jsonl"), plan["bs"])
            save(directory / f"{name}.outputs.json", rows)
            print(method, name, metrics[name]["throughput"], flush=True)
        verify_runtime(engine.get_server_info(), cfg, plan, runtime, method)
    finally:
        engine.shutdown()
    save(directory / "metrics.json", metrics)


def run_arm(plan, plan_path, method, directory):
    busy = foreign_processes(plan["devices"])
    if busy:
        raise RuntimeError(f"Selected A100s have active compute processes: {busy}")
    if directory.exists():
        directory.rename(directory.with_name(f"{directory.name}.incomplete-{time.time_ns()}"))
    directory.mkdir()
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("EAGLE_CONTEXT_", "PROFILE_", "SGLANG_", "REDHAT_"))
        and k not in ("CUDA_LAUNCH_BLOCKING", "RANK", "WORLD_SIZE", "LOCAL_RANK",
                      "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    }
    arm_runtime = Path(plan["runtime"]) / ("repair70b" if method == "repair" else "native")
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=",".join(d["uuid"] for d in plan["devices"]),
        PYTHONPATH=os.pathsep.join((str(ROOT / "tools/trace_hook"), str(arm_runtime))),
        PYTHONDONTWRITEBYTECODE="1",
        REDHAT_BS32_TRACE=str(directory),
        SGLANG_ENABLE_METADATA_GLUE_GRAPH="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    with (directory / "run.log").open("x", encoding="utf-8") as log:
        command = [sys.executable, "-B", __file__, "--worker", str(plan_path), method, str(directory)]
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while proc.poll() is None:
                if busy := foreign_processes(plan["devices"], proc.pid):
                    raise RuntimeError(f"Another workload entered the selected devices: {busy}")
                time.sleep(2)
            if not proc.returncode:  # TP ranks may still be exiting; they are not a foreign workload
                wait_until_released(plan["devices"], proc.pid)
            if proc.returncode or foreign_processes(plan["devices"]):
                raise RuntimeError(f"Arm failed or lost GPU isolation. Inspect {directory / 'run.log'}")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    (directory / "accepted.ok").write_text(digest(load(directory / "metrics.json")) + "\n", encoding="utf-8")


def aggregate(rows):
    sums = {k: sum(r[k] for r in rows) for k in
            ("requests", "seconds", "output_tokens", "accepted", "proposed", "verify_rounds")}
    return dict(sums, throughput=sums["output_tokens"] / sums["seconds"],
                acceptance_rate=sums["accepted"] / sums["proposed"],
                acceptance_length=sums["accepted"] / sums["verify_rounds"])


def summarize(plan, runs):
    table = {}
    for subset in (*SUBSETS, "all", "unique"):
        by_method = {}
        for method in plan["methods"]:
            rows = []
            for repeat in runs:
                readings = repeat[method]
                if subset == "all":
                    row = aggregate(list(readings.values()))
                elif subset == "unique":
                    row = aggregate([reading for name, reading in readings.items() if name != "writing"])
                else:
                    row = readings[subset]
                rows.append(row)
            keys = ("seconds", "throughput", "acceptance_rate", "acceptance_length", "output_tokens")
            by_method[method] = dict(runs=rows, mean={k: statistics.mean(r[k] for r in rows) for k in keys},
                                    sample_sd={k: statistics.stdev(r[k] for r in rows) if len(rows) > 1 else None
                                               for k in keys})
        delta = None
        if set(by_method) == set(METHODS):
            a, b = (by_method[m]["mean"] for m in METHODS)
            delta = dict(throughput_percent=100 * (b["throughput"] / a["throughput"] - 1),
                         seconds_percent=100 * (b["seconds"] / a["seconds"] - 1),
                         acceptance_pp=100 * (b["acceptance_rate"] - a["acceptance_rate"]),
                         acceptance_length=b["acceptance_length"] - a["acceptance_length"])
        table[subset] = dict(methods=by_method, repair_minus_native=delta)
    return {"plan": plan, "summary": table}


def render(result):
    lines = ["# Llama-3.3-70B-Instruct / EAGLE3 TP8", "",
             "Full-subset wall time includes prefill and decode. Input-token preparation, model load and explicit warm-up are outside the timer. Acceptance excludes Target bonus.", "",
             "| Subset | Method | Throughput (tok/s) | E2E (s) | Output tokens (mean total) | Draft tokens/round | Acceptance | Δ throughput vs native |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for subset, entry in result["summary"].items():
        for method, pack in entry["methods"].items():
            m = pack["mean"]
            comparison = entry["repair_minus_native"]
            delta = f'{comparison["throughput_percent"]:+.2f}%' if comparison and method == "repair" else "—"
            lines.append(f'| {subset} | {method} | {m["throughput"]:.2f} | {m["seconds"]:.2f} | '
                         f'{m["output_tokens"]:.0f} | {m["acceptance_length"]:.3f} | {m["acceptance_rate"]:.2%} | {delta} |')
    lines.extend(("", "The unique aggregate drops writing because it shares question's input. Per-repeat results and sample standard deviations are in results.json.", ""))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--runtime", type=Path, default=ROOT / "generated/runtime")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["eagle728"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bs", type=int, choices=(2, 4, 8, 16, 24, 32), default=32)
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--mem-fraction", type=float, default=0.78)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 0 < args.mem_fraction < 1 or not 1 <= args.limit <= 160 or args.repeats < 1:
        parser.error("Invalid memory fraction, limit or repeats")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Each method may only appear once")
    checkpoints = local_model_check(args.target, args.draft)
    request_inputs(args.inputs, args.limit, args.bs)
    receipt = verify_installed(args.runtime.resolve())
    devices = gpu_devices(tuple(args.gpus.split(",")))
    assets = asset_receipt(args.assets.resolve() if args.assets else None) if "repair" in args.methods else None
    if assets:
        repair = stage_repair(args.runtime.resolve())
        receipt["repair70b"] = load(repair / "receipt.json")
    plan = dict(runtime=str(args.runtime.resolve()), runtime_receipt=receipt,
                target=str(args.target.resolve()), draft=str(args.draft.resolve()),
                checkpoint_receipts=checkpoints,
                inputs=str(args.inputs.resolve()), input_sha256=file_sha(args.inputs),
                asset_sha256=assets, assets=str(args.assets.resolve()) if assets else None,
                devices=devices, methods=args.methods, repeats=args.repeats,
                bs=args.bs, limit=args.limit, mem_fraction=args.mem_fraction,
                software={name: importlib.metadata.version(name)
                          for name in ("torch", "triton", "transformers", "sglang-kernel", "flashinfer-python")})
    if args.preflight:
        print(json.dumps(plan, indent=2))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = output / "plan.json"
        if plan_path.exists():
            if load(plan_path) != plan:
                raise ValueError("Existing output directory was created for a different plan")
        else:
            save(plan_path, plan)
        all_runs = []
        for i in range(1, args.repeats + 1):
            completed = {}
            for method in args.methods if i % 2 else reversed(args.methods):
                directory = output / f"r{i}" / method
                directory.parent.mkdir(exist_ok=True)
                marker = directory / "accepted.ok"
                if not marker.is_file():
                    run_arm(plan, plan_path, method, directory)
                metrics = load(directory / "metrics.json")
                if marker.read_text(encoding="utf-8").strip() != digest(metrics):
                    raise ValueError(f"Completed measurement changed under {directory}")
                completed[method] = metrics
            if set(completed) == set(METHODS) and any(
                completed[METHODS[0]][s]["input_ids_sha256"] != completed[METHODS[1]][s]["input_ids_sha256"]
                for s in SUBSETS
            ):
                raise RuntimeError("Paired measurements used different inputs")
            all_runs.append(completed)
        result = summarize(plan, all_runs)
        report_file = output / "REPORT.md"
        result_file = output / "results.json"
        if not result_file.exists():
            save(result_file, result)
        elif load(result_file) != result:
            raise RuntimeError("Existing aggregate differs from complete measurements")
        if not report_file.exists():
            with report_file.open("x", encoding="utf-8") as stream:
                stream.write(render(result))
        print(report_file)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
    else:
        def interrupt(_signal, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        main()
