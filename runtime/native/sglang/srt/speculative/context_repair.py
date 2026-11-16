"""Context-supported candidate repair for greedy Adaptive2, isolated candidate.

Query only verified-context tokens outside the Draft's reduced vocabulary.
Reuse resident Target *weights*, never current Target activations or answers.
Frozen alignment/calibration are inherited from an earlier positive prototype.
This module is not claimed novel or an established online acceptance gain.
"""
import json
from pathlib import Path
import torch
import triton as tr
import triton.language as tl
from sglang.srt.speculative.repair_io import pack_local,merge_global,decision_audit,normalized_features


def mapped_draft_probabilities(native_p, mapped_logit, native_log_partition, use):
    """Add only the chosen X to the complete native LM-head softmax."""
    log_ratio = torch.where(use, mapped_logit - native_log_partition, 0.0)
    updated = torch.stack(
        (
            native_p[:, 0].float() * torch.sigmoid(-log_ratio),
            torch.sigmoid(log_ratio),
        ),
        dim=1,
    )
    return torch.where(use[:, None], updated, native_p)


@tr.jit
def _query(X, TW, POOL, ROOT, HOT, SLOTS, VALID, REM, OUT,
           WIDTH:tl.constexpr, CAP:tl.constexpr,
           D:tl.constexpr, XS:tl.constexpr,
           TWS:tl.constexpr, TLO:tl.constexpr,
           THI:tl.constexpr,
           VOCAB:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0);c=tl.program_id(1)
    request=row//WIDTH
    slot=tl.load(SLOTS+request).to(tl.int64)
    live=tl.load(VALID+row)&(tl.load(REM+request)>0)
    k=tl.arange(0,BLOCK)
    if c < CAP:token=tl.load(POOL+slot*CAP+c)
    else:token=tl.load(ROOT+slot)
    safe=tl.maximum(0,tl.minimum(VOCAB-1,token))
    cold=tl.load(HOT+safe)<0
    owned=live & cold & (token>=TLO)&(token<THI)
    score=-float('inf')
    if owned:
        x=tl.load(X+row*XS+k,k<D,other=0).to(tl.float32)
        w=tl.load(TW+(token-TLO).to(tl.int64)*TWS+k,k<D,other=0).to(tl.float32)
        # Keep the BF16 output rounding used by the frozen readout.
        score=tl.sum(x*w,0).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT+row*(CAP+1)+c,score)


@tr.jit
def _native_pair(H, DW, IDS, HOT, O, D:tl.constexpr, HS:tl.constexpr,
                 WS:tl.constexpr, LO:tl.constexpr, HI:tl.constexpr,
                 BLOCK:tl.constexpr):
    row=tl.program_id(0);which=tl.program_id(1)
    token=tl.load(IDS+row*2+which)
    index=tl.load(HOT+token)
    result=0.
    if (index>=LO)&(index<HI):
        k=tl.arange(0,BLOCK)
        x=tl.load(H+row*HS+k,k<D,other=0).to(tl.float32)
        w=tl.load(DW+(index-LO).to(tl.int64)*WS+k,k<D,other=0).to(tl.float32)
        result=tl.sum(x*w,0).to(tl.bfloat16).to(tl.float32)
    tl.store(O+row*2+which,result)


class ContextRepair:
    """Correctness-first isolated implementation; experimental modes are explicit.

    Default deployment requires a validated unified-event asset. The historical
    classifier remains accessible ONLY through legacy_diagnostic to measure the
    effect of engineering fixes without silently relabelling its mixed scores.
    """
    def __init__(self, root, target_head, draft_head, norm, hot_ids,
                 group, max_requests, capacity=64, cached=False, req_pool=None,
                 target_model_path=None, draft_model_path=None):
        import os
        import hashlib
        from .repair_lifecycle import PrefixLedger, readout_shard_bounds
        from .repair_calibration.contracts import (
            CANDIDATE_PAIR_POLICY, DRAFT_KV_ANCESTRY, hot_map_digest,
            load_deployment_asset, sha256_file,
        )
        from .repair_calibration.candidates import (
            CandidateGenerator, ROOT_POLICY, TIE_POLICY, generator_sha256,
        )
        self.ledger = PrefixLedger()
        self.req_pool = req_pool
        if req_pool is None:
            raise ValueError('ContextRepair requires allocation-generation tracking')
        root = Path(root)
        self.mode = os.environ.get('EAGLE_CONTEXT_REPAIR_MODE', 'validated')
        if self.mode not in ('validated', 'legacy_diagnostic', 'trained_experiment'):
            raise ValueError('Unknown repair mode')
        self.backend = os.environ.get('EAGLE_CONTEXT_READOUT_BACKEND', 'reference')
        if self.backend not in ('reference', 'fused_cached_v1'):
            raise ValueError('Unknown readout backend')
        self.fused_packet_inputs = (
            group.world_size == 1 and target_head.weight.device.type == 'cuda'
            and self.mode == 'trained_experiment' and self.backend == 'fused_cached_v1'
            and os.environ.get('EAGLE_CONTEXT_TREE_SCORE_MODE') == 'legacy_mixed_v1'
            and not os.environ.get('CR_DRAFT_CAPTURE_DIR')
        )
        if self.backend == 'fused_cached_v1' and not cached:
            raise ValueError('fused_cached_v1 requires cached weight projection')
        default_policy = 'legacy_all_cold_or_disable_v1' if self.mode == 'legacy_diagnostic' else 'recent_distinct_v1'
        self.policy = os.environ.get('EAGLE_CONTEXT_POOL_POLICY', default_policy)
        if self.policy not in ('legacy_all_cold_or_disable_v1', 'recent_distinct_v1'):
            raise ValueError('Unknown candidate pool policy')
        self.device = target_head.weight.device
        self._trained_cpu_fast_scope = False
        if (self.device.type == 'cuda' and group.world_size == 1
                and self.mode == 'trained_experiment'
                and self.policy == 'recent_distinct_v1'
                and self.backend == 'fused_cached_v1'):
            from sglang.srt.runtime_context import get_schedule
            self._trained_cpu_fast_scope = bool(get_schedule().disable_overlap_schedule)
        if self._trained_cpu_fast_scope:
            from .repair_array_ledger import ArrayPrefixLedger
            self.ledger = ArrayPrefixLedger()
        self.target, self.draft = target_head.weight, draft_head.weight
        if self.draft.device != self.device:
            raise ValueError('Target and Draft output shards must reside on the same rank device')
        if self.target.dtype != torch.bfloat16 or self.draft.dtype != torch.bfloat16:
            raise ValueError('Only the validated BF16 model family is supported')
        if self.target.ndim != 2 or self.draft.ndim != 2 or self.target.shape[1] != 4096 or self.draft.shape[1] != 4096:
            raise ValueError('Unsupported hidden dimensions')
        if capacity != 64:
            raise ValueError('This experimental runtime requires candidate pool capacity 64')
        self.norm, self.group, self.capacity = norm, group, capacity
        self.vocab = 128256
        if target_head.org_vocab_size != self.vocab or target_head.num_embeddings != self.vocab:
            raise ValueError('Unsupported Target vocabulary or added-token head')
        if draft_head.org_vocab_size != 32000 or draft_head.num_embeddings != 32000:
            raise ValueError('Unsupported Draft output vocabulary or added-token head')
        self.target_low, self.target_high = readout_shard_bounds(target_head, group, 'Target')
        self.draft_low, self.draft_high = readout_shard_bounds(draft_head, group, 'Draft')
        self.hot_ids = hot_ids.to(self.device).long()
        hot_cpu_list = self.hot_ids.cpu().tolist()
        if len(hot_cpu_list) != 32000 or len(set(hot_cpu_list)) != 32000 or min(hot_cpu_list) < 0 or max(hot_cpu_list) >= self.vocab:
            raise ValueError('Invalid reduced vocabulary mapping')
        self.hot_cpu = set(hot_cpu_list)
        self.candidate_generator = CandidateGenerator(self.hot_cpu,capacity=capacity,
            policy=self.policy,target_vocab_size=self.vocab)
        self.incremental_recent_pool = None
        if (self.mode == 'trained_experiment' and group.world_size == 1
                and self.policy == 'recent_distinct_v1'):
            from .repair_recent_distinct_pool import IncrementalRecentDistinctPool
            self.incremental_recent_pool = IncrementalRecentDistinctPool(self.candidate_generator)
        self.stop_metadata_cache = None
        if self._trained_cpu_fast_scope:
            from .repair_stop_metadata import StopMetadataCache
            self.stop_metadata_cache = StopMetadataCache(self.vocab)
        self._ledger_epoch = req_pool.reset_epoch
        self.hot = torch.full((self.vocab,), -1, device=self.device, dtype=torch.long)
        self.hot[self.hot_ids] = torch.arange(len(self.hot_ids), device=self.device)
        alignment_cpu = torch.load(root/'alignment.pt', map_location='cpu', weights_only=True)['transform']
        if alignment_cpu.shape != (4096,4096) or not torch.isfinite(alignment_cpu).all():
            raise ValueError('Invalid alignment transform')
        self.A = alignment_cpu.to(self.device, dtype=torch.bfloat16)
        self.cached = bool(cached)
        self.query_weight = self.target
        if cached:
            self.query_weight = torch.empty_like(self.target)
            with torch.no_grad():
                for begin in range(0,len(self.target),1024):
                    self.query_weight[begin:begin+1024].copy_(self.target[begin:begin+1024]@self.A)
        self.numeric_version = 'cached_WA_bf16_v1' if cached else 'query_hAT_bf16_v1'
        self.raw_budget = 26
        if self.mode != 'legacy_diagnostic':
            from .repair_model_identity import model_identity,tokenizer_identity
            contract_path = os.environ.get('EAGLE_CONTEXT_RUNTIME_CONTRACT')
            if not contract_path:
                raise ValueError('Trained repair requires a runtime contract')
            expected = json.loads(Path(contract_path).read_text(encoding='utf8'))
            target_id,target_hash=model_identity(target_model_path)
            draft_id,draft_hash=model_identity(draft_model_path)
            actual = dict(candidate_generator_sha256=generator_sha256(),
                target_model_id=target_id,target_weights_sha256=target_hash,
                draft_model_id=draft_id,draft_weights_sha256=draft_hash,
                tokenizer_sha256=tokenizer_identity(target_model_path),
                candidate_policy=self.policy, capacity=capacity, readout_backend=self.backend,
                readout_numeric_version=self.numeric_version, hidden_size=4096,
                target_vocab_size=self.vocab, draft_vocab_size=32000, dtype='bfloat16',
                tp_size=group.world_size, max_steps=7, sampling_mode='greedy',
                alignment_sha256=sha256_file(root/'alignment.pt'),
                hot_map_sha256=hot_map_digest(hot_cpu_list),
                root_policy=ROOT_POLICY, tie_policy=TIE_POLICY,
                candidate_pair_policy=CANDIDATE_PAIR_POLICY,
                draft_kv_ancestry=DRAFT_KV_ANCESTRY)
            for k,v in actual.items():
                if expected.get(k) != v:
                    raise ValueError('Runtime contract differs from loaded implementation: '+k)
            if self.mode == 'trained_experiment':
                from .repair_calibration.experiment import load_experiment_asset
                asset = load_experiment_asset(root/'calibration_asset.json', expected,
                    os.environ.get('EAGLE_CONTEXT_EXPERIMENT_ASSET_SHA256'))
            else:
                asset = load_deployment_asset(root/'calibration_asset.json', expected)
            model = asset['parameters']
            self.score_mode = 'unified_target_event_v2'
            self.candidate_pair_policy = CANDIDATE_PAIR_POLICY
            self.asset_hash = sha256_file(root/'calibration_asset.json')
        else:
            # A historical comparison is allowed, but NEVER promoted to validated.
            model = json.loads((root/'models.json').read_text(encoding='utf8'))['joint']
            if model['classes'] != [0,1,2,3]:
                raise ValueError('Unexpected legacy class order')
            self.score_mode = 'legacy_mixed_v1_unvalidated'
            self.candidate_pair_policy = 'preserve_native1_v1'
            self.asset_hash = sha256_file(root/'models.json')
        tree_score_mode = os.environ.get('EAGLE_CONTEXT_TREE_SCORE_MODE', 'default')
        if tree_score_mode not in ('default', 'mapped_draft_v1', 'legacy_mixed_v1'):
            raise ValueError('Unknown ContextRepair tree score mode')
        self.mapped_tree_scores = tree_score_mode == 'mapped_draft_v1'
        self.legacy_mixed_tree_scores = tree_score_mode == 'legacy_mixed_v1'
        if self.legacy_mixed_tree_scores and self.mode != 'trained_experiment':
            raise ValueError('legacy_mixed_v1 requires trained_experiment mode')
        if self.mapped_tree_scores or self.legacy_mixed_tree_scores:
            self.score_mode = tree_score_mode
            self.candidate_pair_policy = 'preserve_native1_v1'
        self.fused_proposal_commit = (
            self.legacy_mixed_tree_scores and group.world_size == 1
            and self.device.type == 'cuda' and self.backend == 'fused_cached_v1'
            and not os.environ.get('CR_DRAFT_CAPTURE_DIR')
        )
        self.draft_kv_ancestry = DRAFT_KV_ANCESTRY
        dim = 11 if self.mode == 'legacy_diagnostic' else 12
        for key,shape in [('mean',(dim,)),('scale',(dim,)),('coef',(4,dim)),('intercept',(4,))]:
            value = torch.tensor(model[key],dtype=torch.float64)
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError('Invalid calibration parameter: '+key)
            if key == 'scale' and not (value > 0).all():
                raise ValueError('Calibration scale must be positive')
            setattr(self,key,value.to(self.device))
        self.pool = torch.full((max_requests,capacity),-1,device=self.device,dtype=torch.long)
        self.roots = torch.full((max_requests,),-1,device=self.device,dtype=torch.long)
        self.overflow = torch.ones((max_requests,),device=self.device,dtype=torch.bool)
        self.prompt_lengths = torch.zeros((max_requests,),device=self.device,dtype=torch.long)
        self.known_counts = torch.zeros((max_requests,),device=self.device,dtype=torch.long)
        self.output_limits = torch.zeros((max_requests,), device=self.device, dtype=torch.long)
        # Stop ids are only a conservative optimization: unsupported cases defer.
        self.stop_ids = torch.full((max_requests,32),-1,device=self.device,dtype=torch.long)
        self.packed_context_upload = None
        if self._trained_cpu_fast_scope:
            from .repair_packed_context import PackedContextUpload
            self.packed_context_upload = PackedContextUpload(self.device, max_requests)
        self.round_errors = torch.zeros((1,),device=self.device,dtype=torch.int64)
        self.tree_errors = torch.zeros((max_requests,),device=self.device,dtype=torch.int32)
        self.stats = dict(pool_updates=0,overflow_updates=0,stale_prefix_disabled=0,
                          terminal_pruning_deferred=0,allocation_checks=0,prefix_lag_request_rounds=0)
        self.audit_names=('live_parent_rows','usable_query_rows','repair_proposals',
            'admitted_oov_nodes','verify_request_rounds','accepted_draft_tokens',
            'accepted_oov_draft_tokens','disabled_request_query_calls',
            'tp_decision_checks','tp_decision_mismatch_calls','verified_tree_oov_nodes',
            'terminal_deferred_request_steps','terminal_filler_nodes')
        self.audit=torch.zeros((len(self.audit_names),),device=self.device,dtype=torch.int64)

    def update_known_context(self,batch):
        """Outside Graph, read only exact committed CPU prefixes and known root."""
        if not batch.sampling_info.is_all_greedy:
            raise RuntimeError('ContextRepair contract requires greedy sampling')
        uploader = self.packed_context_upload
        upload_stream = None if uploader is None else uploader.begin_update()
        if self._ledger_epoch != self.req_pool.reset_epoch:
            self.ledger.clear()
            if self.stop_metadata_cache is not None:
                self.stop_metadata_cache.clear()
            if self.incremental_recent_pool is not None:
                self.incremental_recent_pool.clear()
            self.overflow.fill_(True)
            self.roots.fill_(-1)
            self.stop_ids.fill_(-1)
            self._ledger_epoch = self.req_pool.reset_epoch
        slots=[];pools=[];disabled=[];counts=[];prompts=[];stops=[];output_limits=[]
        allocation_ids = getattr(batch,'context_repair_allocation_ids',None)
        if allocation_ids is None or len(allocation_ids) != len(batch.reqs):
            raise RuntimeError('Missing frozen batch allocation identities')
        stop_cache = self.stop_metadata_cache
        batch_tokenizers = {} if stop_cache is not None else None
        cpu_slots = cpu_lengths = None
        if uploader is not None and batch.reqs:
            slot_tensor, length_tensor = batch.req_pool_indices_cpu, batch.seq_lens_cpu
            expected_shape = (len(batch.reqs),)
            if all(type(t) is torch.Tensor and t.device.type == 'cpu'
                   and t.dtype == torch.int64 and t.layout == torch.strided
                   and t.shape == expected_shape for t in (slot_tensor, length_tensor)):
                cpu_slots, cpu_lengths = slot_tensor.tolist(), length_tensor.tolist()
        for row,req in enumerate(batch.reqs):
            slot = req.kv.req_pool_idx
            if slot is None:
                raise RuntimeError('Batch refers to a released request allocation')
            slot = int(slot)
            if cpu_slots is None:
                row_slot = None if batch.req_pool_indices_cpu is None else int(batch.req_pool_indices_cpu[row])
            else:
                row_slot = cpu_slots[row]
            if row_slot != slot:
                raise RuntimeError('CPU batch slot no longer matches its request allocation')
            live = (self.req_pool.reset_epoch, slot, int(self.req_pool.req_generation[slot]))
            expected = allocation_ids[row]
            if batch.seq_lens_cpu is None:
                # No guessed count based on the potentially newer output list.
                raise RuntimeError('Committed-prefix length must be available on CPU')
            row_length = int(batch.seq_lens_cpu[row]) if cpu_lengths is None else cpu_lengths[row]
            prefix_count = max(0, row_length - len(req.origin_input_ids))
            if prefix_count > len(req.output_ids):
                self.stats['prefix_lag_request_rounds'] += 1
            entry = self.ledger.observe(slot,expected,live,req,req.origin_input_ids,req.output_ids,prefix_count)
            self.stats['allocation_checks'] += 1
            if entry is None:
                # Do not mutate another iteration's candidate membership.
                self.overflow[slot] = True
                self.stop_ids[slot].fill_(-1)
                self.stats['stale_prefix_disabled'] += 1
                continue
            if self.incremental_recent_pool is None:
                pool = self.candidate_generator(entry.prompt,entry.committed)
            else:
                pool = self.incremental_recent_pool.observe(slot, entry, req.origin_input_ids, req.output_ids)
            slots.append(slot)
            pools.append(pool.pool_ids)
            disabled.append((pool.overflow and self.policy=='legacy_all_cold_or_disable_v1') or batch.spec_info.bonus_tokens is None)
            counts.append(len(entry.committed));prompts.append(len(entry.prompt))
            sp=req.sampling_params
            output_limits.append(len(entry.prompt) + sp.max_new_tokens)
            cached_stop = None if stop_cache is None else stop_cache.row(slot, entry, req, sp, batch_tokenizers)
            if cached_stop is None:
                stop = set()
                # Token-only stop handling agrees with Req.check_finished. String and
                # grammar stopping are still accounted for by emitted-output auditing.
                if not sp.ignore_eos and len(entry.committed) >= (sp.min_new_tokens or 0):
                    stop.update(sp.stop_token_ids or ())
                    stop.update(req.eos_token_ids or ())
                    if req.tokenizer is not None:
                        if req.tokenizer.eos_token_id is not None:stop.add(req.tokenizer.eos_token_id)
                        stop.update(getattr(req.tokenizer,'additional_stop_token_ids',None) or ())
                if len(stop)>32:
                    stop=set();self.stats['terminal_pruning_deferred']+=1
                stop=sorted(t for t in stop if 0<=t<self.vocab)
                stops.append(stop+[-1]*(32-len(stop)))
            else:
                stop_row, deferred = cached_stop
                stops.append(stop_row)
                if deferred:
                    self.stats['terminal_pruning_deferred'] += 1
            self.stats['overflow_updates']+=int(pool.overflow)
        if slots:
            packed = uploader is not None and uploader.upload(
                upload_stream, self, slots, pools, disabled, counts, prompts, stops, output_limits
            )
            if not packed:
                pools = [list(ids) + [-1] * (self.capacity - len(ids)) for ids in pools]
                ix=torch.tensor(slots,device=self.device,dtype=torch.long)
                for dst,values,dtype in [(self.pool,pools,torch.long),(self.overflow,disabled,torch.bool),
                    (self.known_counts,counts,torch.long),(self.prompt_lengths,prompts,torch.long),
                    (self.stop_ids,stops,torch.long), (self.output_limits,output_limits,torch.long)]:
                    dst.index_copy_(0,ix,torch.tensor(values,device=self.device,dtype=dtype))
            self.stats['pool_updates']+=len(slots)
        if batch.spec_info.bonus_tokens is not None:
            self.roots.index_copy_(0,batch.req_pool_indices.long(),batch.spec_info.bonus_tokens.long())

    def pool_enabled(self,slots,prefix_lengths):
        ix=slots.long()
        return (ix>0)&~self.overflow[ix]&(self.known_counts[ix]<=prefix_lengths-self.prompt_lengths[ix])

    def draft_output_horizon(self, slots, prefix_lengths):
        # seq_lens excludes the already sampled root, which occupies the
        # next output position before any of this round's draft candidates.
        return (self.output_limits[slots.long()] - prefix_lengths - 1).clamp_min(0)

    def begin_round(self):
        # Captured clear prevents stale errors leaking into the next Graph replay.
        self.round_errors.zero_()

    def query(self,h,ids,slots,valid,remaining,width,prefix_lengths,real,prepared=None):
        rows=h.shape[0]
        assert h.shape[1]==4096 and rows==len(slots)*width
        normalized=self.norm(h.clone())
        if isinstance(normalized,tuple):normalized=normalized[0]
        if prepared is None:
            available=self.pool_enabled(slots,prefix_lengths)&real
            enabled=valid&available.repeat_interleave(width)
        else:
            available, enabled = prepared
        if self.backend=='fused_cached_v1':
            from .repair_row_readout import packet as fused_packet
            packet=fused_packet(normalized,self.query_weight,self.draft,ids,self.pool,self.roots,
                self.hot,slots,enabled,remaining,width,self.target_low,self.target_high,
                self.draft_low,self.draft_high,self.vocab)
        else:
            transformed=normalized if self.cached else normalized@self.A.T
            scores=torch.empty((rows,self.capacity+1),device=self.device,dtype=torch.float32)
            _query[(rows,self.capacity+1)](transformed,self.query_weight,self.pool,self.roots,
                self.hot,slots,enabled,remaining,scores,WIDTH=width,CAP=self.capacity,D=4096,
                XS=transformed.stride(0),TWS=self.query_weight.stride(0),TLO=self.target_low,
                THI=self.target_high,VOCAB=self.vocab,BLOCK=4096,num_warps=4)
            native=torch.empty((rows,2),device=self.device,dtype=torch.float32)
            _native_pair[(rows,2)](normalized,self.draft,ids,self.hot,native,D=4096,
                HS=normalized.stride(0),WS=self.draft.stride(0),LO=self.draft_low,
                HI=self.draft_high,BLOCK=4096,num_warps=4)
            packet=pack_local(scores,self.pool,self.roots,slots,native,width)
        if self.group.world_size>1:
            gathered=torch.empty((self.group.world_size*rows,4),device=self.device,dtype=torch.float32)
            self.group.all_gather_into_tensor(gathered,packet)
            gathered=gathered.reshape(self.group.world_size,rows,4)
        else:
            if prepared is not None:
                from .repair_packet_inputs import finish_packet
                return finish_packet(packet,available,real,self.round_errors,self.audit,width)
            gathered=packet[None]
        packet_error=(torch.isnan(gathered)|torch.isposinf(gathered)).any(-1)
        self.round_errors.add_((packet_error&real.repeat_interleave(width)[None,:]).any().long())
        return merge_global(gathered,available,width,self.audit,real)

    def propose(self,h,p,ids,slots,valid,remaining,step,prefix_lengths,real_request_mask,
                native_log_partition=None):
        p, ids, _ = self._propose(h,p,ids,slots,valid,remaining,step,prefix_lengths,
                                 real_request_mask,native_log_partition)
        return p, ids

    def propose_with_terminal(self,h,p,ids,slots,valid,remaining,step,prefix_lengths,real_request_mask,
                              native_log_partition=None):
        return self._propose(h,p,ids,slots,valid,remaining,step,prefix_lengths,
                             real_request_mask,native_log_partition,want_terminal=True)

    def _propose(self,h,p,ids,slots,valid,remaining,step,prefix_lengths,real_request_mask,
                 native_log_partition=None,want_terminal=False):
        width=1 if step==0 else 2
        real=real_request_mask
        inputs = None
        if self.fused_packet_inputs:
            from .repair_packet_inputs import packet_inputs
            inputs = packet_inputs(p,ids,slots,valid,prefix_lengths,real,
                                   self.overflow,self.known_counts,self.prompt_lengths,
                                   self.round_errors,width,self.vocab)
        if inputs is None:
            active=valid.flatten()&real.repeat_interleave(width)
            bad_ids=((ids<0)|(ids>=self.vocab)).any(1)
            bad_p=(~torch.isfinite(p)|(p<0)|(p>1)).any(1)
            self.round_errors.add_(((bad_ids|bad_p)&active).any().long())
            # Keep later error handling safe: no invalid GPU index can be consumed.
            ids=ids.clamp(0,self.vocab-1)
            best,cold,native,usable=self.query(h,ids,slots,active,remaining,width,prefix_lengths,real)
        else:
            ids, slots, active, available, enabled = inputs
            best,cold,native,usable=self.query(h,ids,slots,active,remaining,width,
                                              prefix_lengths,real,prepared=(available,enabled))
        x=normalized_features(p,best,native,usable,self.mean[:11],self.scale[:11],step)
        if self.mode != 'legacy_diagnostic':
            has_x=(usable.double()-self.mean[11])/self.scale[11]
            x=torch.cat((x,has_x[:,None]),1)
        logits=x@self.coef.T+self.intercept
        if self.mode != 'legacy_diagnostic':
            logits[:,2]=torch.where(usable,logits[:,2],-torch.inf)
        posterior=logits.softmax(-1)
        if want_terminal and self.fused_proposal_commit:
            from .repair_proposal_commit import proposal_commit
            fused = proposal_commit(p,ids,posterior,cold,active,usable,remaining,slots,
                                    self.stop_ids,self.round_errors,self.audit,width)
            if fused is not None:
                return fused
        self.round_errors.add_((~torch.isfinite(posterior)&active[:,None]).any().long())
        eligible = usable & active & (remaining.repeat_interleave(width) > 0)
        if self.mapped_tree_scores:
            if (
                native_log_partition is None
                or native_log_partition.shape != best.shape
                or native_log_partition.dtype != torch.float32
            ):
                raise RuntimeError('Mapped Draft scoring requires the full native log partition')
            use = eligible & (posterior[:, 2] > posterior[:, 1])
            invalid = active & ~torch.isfinite(native_log_partition)
            self.round_errors.add_(invalid.any().long())
            use = use & ~invalid
            repaired_ids = torch.stack((ids[:, 0], torch.where(use, cold, ids[:, 1])), 1)
            repaired_p = mapped_draft_probabilities(p, best, native_log_partition, use)
        elif self.mode != 'legacy_diagnostic' and not self.legacy_mixed_tree_scores:
            # Stable descending order: native1, native2, then X on ties.
            scores = torch.cat((posterior[:, :2], torch.where(eligible, posterior[:, 2], -torch.inf)[:, None]), dim=1)
            order = scores.argsort(dim=1, descending=True, stable=True)[:, :2]
            choices = torch.cat((ids, cold[:, None]), dim=1)
            repaired_ids = torch.where(active[:, None], choices.gather(1, order), ids)
            repaired_p = torch.where(active[:, None], scores.gather(1, order).float(), p)
            use = active & (order == 2).any(dim=1)
        else:
            use=eligible&(posterior[:,2]>posterior[:,1])
            repaired_ids=torch.stack((ids[:,0],torch.where(use,cold,ids[:,1])),1)
            adjusted=torch.stack((posterior[:,0],posterior[:,2]),1).float()
            repaired_p=torch.where(use[:,None],adjusted,p)
        decision_audit(repaired_p,repaired_ids,use,active,usable,remaining,width,self.audit,
                       fingerprint=False)
        if self.group.world_size > 1:
            # The fixed-shape collective is captured with the draft graph.
            local = torch.cat((repaired_p, repaired_ids.float()), dim=1).contiguous()
            gathered = local.new_empty((self.group.world_size * len(p), 4))
            self.group.all_gather_into_tensor(gathered, local)
            gathered = gathered.reshape(self.group.world_size, len(p), 4)
            mismatch = ((gathered != gathered[:1]) & active[None, :, None]).any()
            self.round_errors.add_(mismatch.long())
            self.audit[8].add_(real.any().long())
            self.audit[9].add_(mismatch.long())
        terminal = self.candidate_terminal(repaired_ids,slots,width) if want_terminal else None
        return repaired_p,repaired_ids,terminal

    def candidate_terminal(self,tokens,slots,width):
        stop=self.stop_ids[slots.long()].repeat_interleave(width,dim=0)
        return ((tokens[:,:,None]==stop[:,None,:])&(stop[:,None,:]>=0)).any(-1).contiguous()

    def check_round(self,real_bs):
        # Must run BEFORE tree-mask construction, not just before Target forward.
        errors=self.round_errors.clone()+self.tree_errors[:real_bs].ne(0).sum()
        if self.group.world_size > 1:
            errors = self.group.all_reduce(errors)
        if int(errors.item()):
            flags = self.tree_errors[:real_bs].tolist()
            raise RuntimeError(
                'ContextRepair rejected invalid or TP-inconsistent draft tree '
                f'before Target verification (local round errors={self.round_errors.item()}, '
                f'local tree flags={flags})'
            )

    def record_admitted(self,scores,tokens,real):
        legal=torch.isfinite(scores).reshape(tokens.shape)&real[:,None]
        self.audit[3].add_((legal&(self.hot[tokens.long().clamp(0,self.vocab-1)]<0)).sum())

    def record_verified_tree(self,tokens,real,tree_state):
        self.audit[10].add_(((self.hot[tokens.long().clamp(0,self.vocab-1)]<0)&real[:,None]).sum())
        self.audit[11].add_((tree_state['deferred_count']*real).sum())
        self.audit[12].add_((tree_state['terminal_fill_count']*real).sum())

    def verified_sources(self,tokens,accept_lens):
        # Only this frozen model contract: native head cannot emit a cold token.
        bs=accept_lens.numel()
        if not bs:
            return torch.empty_like(tokens,dtype=torch.int8)
        values=tokens.reshape(bs,-1).long()
        mask=torch.arange(values.shape[1],device=self.device)[None,:]<(accept_lens.long()-1)[:,None]
        cold=self.hot[values.clamp(0,self.vocab-1)]<0
        return torch.where(mask,1+cold.to(torch.int8),0).to(torch.int8).reshape(-1)

    def record_verified(self,tokens,accept_lens):
        sources=self.verified_sources(tokens,accept_lens)
        if accept_lens.numel():
            self.audit[4].add_(accept_lens.numel())
            self.audit[5].add_((sources>0).sum());self.audit[6].add_((sources==2).sum())
        return sources

    def snapshot(self):
        import hashlib
        stop_cache = self.stop_metadata_cache
        stop_cache_stats = (None if stop_cache is None else
                            dict(hits=stop_cache.hits, misses=stop_cache.misses,
                                 inactive=stop_cache.inactive, fallbacks=stop_cache.fallbacks))
        return dict(stop_metadata_cache=stop_cache_stats, cpu=dict(self.stats),gpu=dict(zip(self.audit_names,self.audit.cpu().tolist())),
            definition='Cumulative real-request counters; subtract workload snapshots. Raw acceptance is not emitted acceptance.',
            tp_size=self.group.world_size,capacity=self.capacity,candidate_policy=self.policy,mode=self.mode,
            score_mode=self.score_mode,backend=self.backend,numeric_version=self.numeric_version,
            mapped_native_partition_source='native_full_logits' if self.mapped_tree_scores else None,
            candidate_pair_policy=self.candidate_pair_policy,draft_kv_ancestry=self.draft_kv_ancestry,
            module_file=str(Path(__file__).resolve()),module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            calibration_sha256=self.asset_hash,cached_weight_projection=self.cached,
            experimental_proposal_commit_eligible=self.fused_proposal_commit,
            array_prefix_ledger_enabled=self._trained_cpu_fast_scope,
            experimental_fused_packet_inputs_eligible=self.fused_packet_inputs,
            deployment_validated=self.mode=='validated',correctness_barrier_per_round=True,
            packed_context_upload_enabled=self.packed_context_upload is not None,
            packed_context_upload_calls=0 if self.packed_context_upload is None else self.packed_context_upload.packed_calls,
            packed_context_fallback_calls=0 if self.packed_context_upload is None else self.packed_context_upload.fallback_calls,
            packed_host_row_cache=(None if self.packed_context_upload is None else
                dict(hits=self.packed_context_upload.static_rows.hits,
                     rebuilt=self.packed_context_upload.static_rows.rebuilt)),
            raw_generation_budget=self.raw_budget,verification_draft_nodes=7,
            incremental_recent_pool_enabled=self.incremental_recent_pool is not None,
            incremental_recent_pool=(self.incremental_recent_pool.receipt()
                                     if self.incremental_recent_pool is not None else None))
