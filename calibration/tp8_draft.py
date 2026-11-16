"""Eager-only TP8 Draft capture. All eight ranks execute the readout collectives."""

from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path

from .contracts import prefix_hash


SUPPORTED_MODULES = ("context_repair", "bound_frontier", "model_runner")
MAX_STEPS = 7
NATIVE_PROBABILITY_ATOL = 1e-6
NATIVE_PROBABILITY_RTOL = 1e-5


def _eager():
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Calibration capture must execute outside CUDA Graph")


def _cpu(tensor):
    return tensor.detach().cpu().tolist()


def _writer(group):
    return group.rank_in_group == 0


def _probe_request(seed, rate, request_id, visible_hash, root):
    key = json.dumps([seed, request_id, visible_hash, root], separators=(",", ":"))
    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return bucket < rate * (1 << 64)


def _align_request(req, seq_len, root):
    """In non-overlap mode, the visible last token is the pending bonus root."""
    prompt = list(req.origin_input_ids)
    output = list(req.output_ids)
    finished = getattr(req, "finished_len", None)
    if finished is not None:
        if type(finished) is not int or not 0 <= finished <= len(output):
            raise RuntimeError("Invalid finished_len in Draft request")
        output = output[:finished]
    length = seq_len - len(prompt)
    if length != len(output) - 1 or length < 0 or output[length] != root:
        raise RuntimeError(f"Draft committed/root boundary mismatch for {req.rid}")
    if getattr(req, "is_retracted", False) or (callable(getattr(req, "finished", None)) and req.finished()):
        raise RuntimeError(f"Inactive request in Draft capture for {req.rid}")
    committed = output[:length]
    return prompt, committed, prefix_hash(prompt + committed)


@dataclass
class Seed:
    token: int
    logits: object
    hidden: object


class Capture:
    def __init__(self):
        self.repair = None
        self.requests = []
        self.paths = []
        self.kinds = []
        self.seeds = {}
        self.pending = None
        self.observed = None
        self.step = None

    def begin(self, repair, batch):
        _eager()
        if repair.group.world_size != 8 or repair.hidden_size != 8192:
            raise ValueError("Expected 70B TP8 for calibration")
        if batch.enable_overlap or not batch.sampling_info.is_all_greedy:
            raise ValueError("Calibration requires non-overlap greedy generation")
        self.repair = repair
        self.requests, self.paths, self.kinds = [], [], []
        self.pending = self.observed = self.step = None
        slots = _cpu(batch.req_pool_indices)
        lengths = _cpu(batch.seq_lens)
        roots = _cpu(repair.roots[batch.req_pool_indices.long()])
        for req, slot, length, root in zip(batch.reqs, slots, lengths, roots, strict=True):
            prompt, output, visible = _align_request(req, length, root)
            self.requests.append(dict(
                request_id=str(req.rid), slot=slot, seq_len=length, root_token=root,
                prompt_ids=prompt, committed_output_ids=output,
                visible_prefix_sha256=visible, round_id=f"{visible}-{root}",
            ))
            self.paths.append([[root]])
            self.kinds.append(["root"])

    def native_logits(self, h, step):
        import torch

        if step == 0:
            seeds = [self.seeds.pop(row["slot"], None) for row in self.requests]
            if any(seed is None or seed.token != row["root_token"]
                   or not torch.equal(seed.hidden, h[index])
                   for index, (seed, row) in enumerate(zip(seeds, self.requests, strict=True))):
                return None
            return torch.stack([seed.logits for seed in seeds])
        observed = self.observed
        self.observed = None
        if observed is None or observed[0] != step or not torch.equal(observed[1], h):
            return None
        return observed[2]

    def observe_seed(self, batch, logits, hidden):
        if logits is None or hidden is None or logits.ndim != 2 or logits.shape[1] != 32000:
            return
        mode = batch.forward_mode
        extend = getattr(mode, "is_draft_extend_v2", lambda: False)()
        if not extend and not getattr(mode, "is_prefill", lambda: False)():
            return
        slots = _cpu(batch.req_pool_indices)
        token_ids = _cpu(batch.input_ids)
        if extend:
            counts = _cpu(batch.spec_info.num_accept_tokens)
            width = batch.spec_info.num_tokens_per_req
            if (len(counts) != len(slots) or len(token_ids) != len(slots) * width
                    or len(logits) != len(token_ids) or len(hidden) != len(token_ids)
                    or not all(1 <= count <= width for count in counts)):
                return
            indices = [i * width + count - 1 for i, count in enumerate(counts)]
            hidden_indices = indices
        else:
            lengths = _cpu(batch.extend_seq_lens)
            if (len(lengths) != len(slots) or sum(lengths) != len(token_ids)
                    or len(logits) != len(slots) or len(hidden) != len(slots)
                    or any(length < 1 for length in lengths)):
                return
            indices, end = [], -1
            for length in lengths:
                end += length
                indices.append(end)
            hidden_indices = range(len(slots))
        for slot, position, row in zip(slots, indices, hidden_indices, strict=True):
            self.seeds[slot] = Seed(token_ids[position], logits[row].detach().clone(), hidden[row].detach().clone())


_CAPTURE = Capture()


def _reference_readout(module, repair, h, ids, slots, enabled, remaining, width):
    """Compute every local candidate separately from the production packed kernel."""
    import torch

    normalized = repair.norm(h.clone())
    if isinstance(normalized, tuple):
        normalized = normalized[0]
    rows, hidden = normalized.shape
    full_local = torch.empty((rows, repair.capacity + 3), device=h.device, dtype=torch.float32)
    scores = torch.empty((rows, repair.capacity + 1), device=h.device, dtype=torch.float32)
    module._query[(rows, repair.capacity + 1)](
        normalized, repair.query_weight, repair.pool, repair.roots, repair.hot, slots,
        enabled, remaining, scores, WIDTH=width, CAP=repair.capacity,
        D=hidden, XS=normalized.stride(0), TWS=repair.query_weight.stride(0),
        TLO=repair.target_low, THI=repair.target_high, VOCAB=repair.vocab,
        BLOCK=hidden, num_warps=4,
    )
    native = torch.empty((rows, 2), device=h.device, dtype=torch.float32)
    module._native_pair[(rows, 2)](
        normalized, repair.draft, ids.contiguous(), repair.hot, native,
        D=hidden, HS=normalized.stride(0), WS=repair.draft.stride(0),
        LO=repair.draft_low, HI=repair.draft_high, BLOCK=hidden, num_warps=4,
    )
    full_local[:, :repair.capacity + 1] = scores
    full_local[:, repair.capacity + 1:] = native
    gathered = full_local.new_empty((repair.group.world_size * rows, full_local.shape[1]))
    repair.group.all_gather_into_tensor(gathered, full_local)
    return gathered.reshape(repair.group.world_size, rows, full_local.shape[1])


def _native_parity(capture, h, p, ids, step):
    """Compare against the observed full Draft projection, including softmax mass."""
    import torch

    logits = capture.native_logits(h, step)
    if logits is None or logits.shape != (len(h), capture.repair.hot_ids.numel()):
        return [False] * len(h)
    indices = capture.repair.hot[ids]
    valid = (indices >= 0).all(-1) & (indices[:, 0] != indices[:, 1])
    probs = torch.softmax(logits, dim=-1).float()
    chosen = probs.gather(1, indices.clamp_min(0))
    values = probs.topk(2, dim=-1).values
    # Identical probability ties need not use the same stable sort implementation.
    ranked = (chosen == values).all(-1)
    measured = torch.isclose(chosen, p.float(), atol=NATIVE_PROBABILITY_ATOL, rtol=NATIVE_PROBABILITY_RTOL).all(-1)
    return _cpu(valid & ranked & measured)


def _record_step(capture, h, p, ids, slots, active, remaining, step, selected, used,
                 actual, reference, raw_path):
    import torch

    repair = capture.repair
    width = 1 if step == 0 else 2
    best, cold, native, usable = actual
    query_ids = torch.cat((repair.pool[slots.long()], repair.roots[slots.long(), None]), dim=1)
    local_scores = reference[..., :repair.capacity + 1]
    count = torch.isfinite(local_scores).sum(0)
    if (torch.isnan(local_scores) | torch.isposinf(local_scores)).any().item():
        raise RuntimeError("Nonfinite score from a vocabulary shard during capture")
    all_scores = _cpu(local_scores.amax(0))
    owners = _cpu(count)
    reference_native = _cpu(reference[..., repair.capacity + 1:].sum(0))
    native_parity = _native_parity(capture, h, p, ids, step)
    query_ids_cpu, p_cpu, ids_cpu, live_rows, remaining_cpu, selected_cpu, used_cpu = map(
        _cpu, (query_ids, p, ids, active, remaining, selected, used),
    )
    best_cpu, cold_cpu, native_cpu, usable_cpu = map(_cpu, (best, cold, native, usable))
    rows = []
    for index, live in enumerate(live_rows):
        request, lane = divmod(index, width)
        if not live or remaining_cpu[request] <= 0:
            continue
        context = capture.requests[request]
        path = capture.paths[request][lane]
        kind = capture.kinds[request][lane]
        if path is None or len(path) != step + 1:
            raise RuntimeError("Draft frontier row is missing its actual parent path")
        expected = repair.candidate_generator(
            context["prompt_ids"], context["committed_output_ids"], root_token=context["root_token"]
        )
        candidates = list(expected.candidate_ids)
        observed_scores = {}
        duplicate_agreement = True
        ownership_ok = True
        for token, value, owned in zip(query_ids_cpu[request], all_scores[index], owners[index], strict=True):
            if owned > 1:
                ownership_ok = False
            if owned != 1 or not math.isfinite(value):
                continue
            if token in observed_scores and observed_scores[token] != value:
                duplicate_agreement = False
            observed_scores[token] = value
        reference_x = min(observed_scores, key=lambda t: (-observed_scores[t], t)) if observed_scores else None
        actual_x = cold_cpu[index] if usable_cpu[index] else None
        readout_parity = (
            ownership_ok and duplicate_agreement and set(observed_scores) == set(candidates)
            and reference_x == actual_x and native_cpu[index] == reference_native[index]
            and (reference_x is None or observed_scores[reference_x] == best_cpu[index])
        )
        a, b = ids_cpu[index]
        p1, p2 = p_cpu[index]
        rows.append(dict(
            **{key: value for key, value in context.items() if key not in ("slot", "seq_len")},
            depth=step, parent_kind=kind,
            parent_prefix_ids=context["prompt_ids"] + context["committed_output_ids"] + path,
            candidate_ids=candidates, candidate_scores=[observed_scores.get(t) for t in candidates],
            x_token=actual_x, has_x=bool(usable_cpu[index]), native1=a, native2=b,
            p1=p1, p2=p2, native2_logit=native_cpu[index][1],
            selected_for_probe=bool(selected_cpu[index]), probe_replaced_b=bool(used_cpu[index]),
            native_parity=bool(native_parity[index]), readout_parity=bool(readout_parity),
        ))
    with raw_path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def advance_paths(step, paths, selected, valid, archive_ids, offers):
    """Follow selected physical children and require matching proposal identity."""
    next_paths, next_kinds = [], []
    for request, previous in enumerate(paths):
        out, kinds = [], []
        for lane, token in enumerate(selected[request]):
            if not valid[request][lane]:
                out.append(None)
                kinds.append(None)
                continue
            index = lane if step == 0 else archive_ids[request][lane] - (2 + (step - 1) * 4)
            if not 0 <= index < 2 * len(previous):
                raise RuntimeError("Selected archive index is outside the current proposal block")
            parent, choice = divmod(index, 2)
            if previous[parent] is None or offers[request][parent][choice][0] != token:
                raise RuntimeError("Draft frontier token does not follow the actual previous proposal")
            out.append(previous[parent] + [token])
            kinds.append(offers[request][parent][choice][1])
        next_paths.append(out)
        next_kinds.append(kinds)
    return next_paths, next_kinds


def _sync_failure(repair, failure, stage, device):
    import torch

    total = repair.group.all_reduce(
        torch.tensor([int(failure is not None)], dtype=torch.int64, device=device)
    )
    if total.item():
        raise RuntimeError(f"TP8 Draft capture failed at {stage}: {failure or 'rank 0 capture'}") from failure


def _install_context(module, raw_path, seed, rate):
    import torch

    cls = module.ContextRepair
    original_update = cls.update_known_context

    @wraps(original_update)
    def update(self, batch):
        result = original_update(self, batch)
        if self.mode == "calibration_collect":
            _CAPTURE.begin(self, batch)
        return result

    def propose(self, h, p, ids, slots, valid, remaining, step, prefix_lengths,
                real_request_mask, native_log_partition=None):
        _eager()
        capture = _CAPTURE
        if self.mode != "calibration_collect" or capture.repair is not self:
            raise RuntimeError("Draft capture hook called outside the isolated collection runtime")
        if self.fused_proposal_commit or self.mapped_tree_scores or native_log_partition is not None:
            raise RuntimeError("Unexpected production scoring was enabled in calibration")
        width = 1 if step == 0 else 2
        if len(slots) != len(capture.requests) or ids.shape != (width * len(slots), 2):
            raise RuntimeError("Unexpected Draft parent layout in calibration")
        if ids.dtype != torch.long or h.dtype != torch.bfloat16 or p.shape != ids.shape:
            raise RuntimeError("Unexpected Draft proposal tensor types")
        active = valid.flatten() & real_request_mask.repeat_interleave(width)
        bad_ids = ((ids < 0) | (ids >= self.vocab)).any(1)
        bad_probs = (~torch.isfinite(p) | (p < 0) | (p > 1)).any(1)
        invalid = ((bad_ids | bad_probs) & active).any().to(torch.int64)
        self.round_errors.add_(invalid)
        safe_ids = ids.clamp(0, self.vocab - 1).contiguous()
        available = self.pool_enabled(slots, prefix_lengths) & real_request_mask
        enabled = active & available.repeat_interleave(width)
        actual = self.query(h, safe_ids, slots, active, remaining, width, prefix_lengths, real_request_mask)
        reference = _reference_readout(module, self, h, safe_ids, slots, enabled, remaining, width)
        _, cold, _, usable = actual
        selected = torch.zeros_like(usable)
        use = selected
        if step == 0:
            claims = [_probe_request(seed, rate, ctx["request_id"], ctx["visible_prefix_sha256"], ctx["root_token"])
                      for ctx in capture.requests]
            selected = torch.tensor(claims, dtype=torch.bool, device=p.device)
            horizon = self.draft_output_horizon(slots, prefix_lengths)
            terminal = self.candidate_terminal(cold[:, None], slots, width).flatten()
            use = selected & enabled & usable & (remaining >= 2) & (horizon >= 2) & ~terminal & (p[:, 1] > 0)
        chosen = torch.stack((safe_ids[:, 0], torch.where(use, cold, safe_ids[:, 1])), dim=1).contiguous()
        # Keep the original scores and archived node budget for collection.
        local = torch.cat((p.float(), chosen.float()), dim=1).contiguous()
        gathered = local.new_empty((self.group.world_size * len(p), local.shape[1]))
        self.group.all_gather_into_tensor(gathered, local)
        gathered = gathered.reshape(self.group.world_size, len(p), local.shape[1])
        mismatch = ((gathered != gathered[:1]) & active[None, :, None]).any()
        self.round_errors.add_(mismatch.long())
        self.audit[8].add_(real_request_mask.any().long())
        self.audit[9].add_(mismatch.long())
        module.decision_audit(p, chosen, use, active, usable, remaining, width, self.audit, fingerprint=False)
        capture.step = step
        failure = None
        if _writer(self.group):
            try:
                _record_step(capture, h, p, safe_ids, slots, active, remaining, step, selected, use,
                             actual, reference, raw_path)
                cells = _cpu(chosen.reshape(len(slots), width, 2))
                decisions = _cpu(use.reshape(len(slots), width))
                capture.offers = [
                    [[(a, "native1"), (b, "x" if swap else "native2")]
                     for (a, b), swap in zip(row, mask, strict=True)]
                    for row, mask in zip(cells, decisions, strict=True)
                ]
            except Exception as exc:
                failure = exc
        _sync_failure(self, failure, f"proposal step {step}", h.device)
        return p, chosen

    cls.update_known_context = update
    cls.propose = propose


def _install_frontier(module):
    original = module.select_frontier

    @wraps(original)
    def select(step, *args, **kwargs):
        result = original(step, *args, **kwargs)
        capture = _CAPTURE
        if capture.repair is None:
            return result
        ids, hidden, _, info, valid, _ = result
        failure = None
        if _writer(capture.repair.group):
            try:
                if step != capture.step:
                    raise RuntimeError("Draft selector is out of sync with captured proposals")
                capture.paths, capture.kinds = advance_paths(
                    step, capture.paths, _cpu(ids.reshape(-1, 2)), _cpu(valid), _cpu(info[2]), capture.offers,
                )
                if step < MAX_STEPS - 1:
                    capture.pending = (step + 1, ids.detach().clone(), hidden.detach().clone())
            except Exception as exc:
                failure = exc
        _sync_failure(capture.repair, failure, f"frontier step {step}", ids.device)
        return result

    module.select_frontier = select


def _install_runner(module):
    original = module.ModelRunner.forward

    @wraps(original)
    def forward(self, batch, *args, **kwargs):
        import torch

        rank_zero = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
        if not rank_zero or not getattr(self, "is_draft_worker", False):
            return original(self, batch, *args, **kwargs)
        capture = _CAPTURE
        pending = capture.pending if batch.forward_mode.is_decode() else None
        if pending is not None:
            step, token_ids, hidden_in = pending
            capture.pending = None
            expected_slots = [ctx["slot"] for ctx in capture.requests]
            inputs_match = (
                torch.equal(batch.input_ids, token_ids)
                and torch.equal(batch.spec_info.hidden_states, hidden_in)
                and _cpu(batch.req_pool_indices) == expected_slots
            )
        result = original(self, batch, *args, **kwargs)
        output = result.logits_output
        logits, hidden = output.next_token_logits, output.hidden_states
        if pending is None:
            capture.observe_seed(batch, logits, hidden)
        elif (inputs_match and logits is not None and hidden is not None
              and logits.shape == (len(token_ids), 32000) and len(hidden) == len(token_ids)):
            capture.observed = (step, hidden.detach().clone(), logits.detach().clone())
        return result

    module.ModelRunner.forward = forward


def install(module):
    """Called by the isolated calibration import hook in each scheduler process."""
    if os.environ.get("OOV_TP8_COLLECTION") != "1":
        return
    name = module.__name__.rsplit(".", 1)[-1]
    if name not in SUPPORTED_MODULES:
        raise ValueError(f"Unexpected Draft capture module {name}")
    if getattr(module, "_oov_tp8_capture", False):
        return
    if name == "context_repair":
        raw = Path(os.environ["OOV_TP8_DRAFT_JSONL"])
        rate = float(os.environ["OOV_TP8_PROBE_RATE"])
        seed = int(os.environ["OOV_TP8_PROBE_SEED"])
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Probe rate must be in [0,1]")
        _install_context(module, raw, seed, rate)
    elif name == "bound_frontier":
        _install_frontier(module)
    else:
        _install_runner(module)
    module._oov_tp8_capture = True
