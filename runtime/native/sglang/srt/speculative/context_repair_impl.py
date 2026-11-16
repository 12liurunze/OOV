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
from sglang.srt.speculative.repair_io import pack_local,merge_global,decision_audit,agreement,normalized_features


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
    def __init__(self, root, target_head, draft_head, norm, hot_ids,
                 group, max_requests, capacity=64, cached=False):
        root=Path(root)
        self.device=target_head.weight.device
        self.target=target_head.weight
        self.draft=draft_head.weight
        assert self.target.dtype==self.draft.dtype==torch.bfloat16
        assert self.target.ndim==self.draft.ndim==2 and self.target.shape[1]==4096
        self.norm=norm
        self.group=group
        self.capacity=capacity
        self.hot_ids=hot_ids.to(self.device).long()
        self.vocab=128256
        self.hot=torch.full((self.vocab,),-1,device=self.device,dtype=torch.long)
        self.hot[self.hot_ids]=torch.arange(len(self.hot_ids),device=self.device)
        self.hot_cpu=set(self.hot_ids.cpu().tolist())
        self.target_low=target_head.shard_indices.org_vocab_start_index
        self.target_high=target_head.shard_indices.org_vocab_end_index
        self.draft_low=draft_head.shard_indices.org_vocab_start_index
        self.draft_high=draft_head.shard_indices.org_vocab_end_index
        self.A=torch.load(root/'alignment.pt',map_location='cpu',weights_only=True)['transform'].to(self.device,dtype=torch.bfloat16)
        self.cached=cached
        self.query_weight=self.target
        if cached:
            # A separate numerical variant: reassociate the static linear map
            # once at startup. BF16 rounding occurs on W*A rather than h*A.T;
            # component decision checks and model validation are required.
            self.query_weight=torch.empty_like(self.target)
            with torch.no_grad():
                for begin in range(0,len(self.target),1024):
                    self.query_weight[begin:begin+1024].copy_(self.target[begin:begin+1024]@self.A)
        self.raw_budget=2+4*(7-1)  # maximum visible children from the same six two-row forwards
        model=json.loads((root/'models.json').read_text())['joint']
        assert model['classes']==[0,1,2,3]
        self.mean=torch.tensor(model['mean'],device=self.device,dtype=torch.float64)
        self.scale=torch.tensor(model['scale'],device=self.device,dtype=torch.float64)
        self.coef=torch.tensor(model['coef'],device=self.device,dtype=torch.float64)
        self.intercept=torch.tensor(model['intercept'],device=self.device,dtype=torch.float64)
        self.pool=torch.full((max_requests,capacity),-1,device=self.device,dtype=torch.long)
        self.roots=torch.full((max_requests,),-1,device=self.device,dtype=torch.long)
        self.overflow=torch.zeros((max_requests,),device=self.device,dtype=torch.bool)
        self.prompt_lengths=torch.zeros((max_requests,),device=self.device,dtype=torch.long)
        self.known_counts=torch.zeros((max_requests,),device=self.device,dtype=torch.long)
        self.owners={}
        self.past_counts={}
        self.members={}
        self.stats=dict(pool_updates=0,overflow_updates=0,scanned_verified_tokens=0)
        self.audit_names=('live_parent_rows','usable_query_rows','repair_proposals',
            'admitted_oov_nodes','verify_request_rounds','accepted_draft_tokens',
            'accepted_oov_draft_tokens','disabled_request_query_calls',
            'tp_decision_checks','tp_decision_mismatch_calls','verified_tree_oov_nodes')
        self.audit=torch.zeros((len(self.audit_names),),device=self.device,dtype=torch.int64)

    def update_known_context(self,batch):
        """Outside Graph: only CPU-available, already verified tokens are read.

        Pending root is copied separately on GPU and never persisted in the CPU
        set. A reused request-pool slot always resets its ownership and contents.
        """
        if batch.spec_info.bonus_tokens is None:
            # No forced D2H synchronization or guessed committed-prefix count.
            self.overflow.index_fill_(0,batch.req_pool_indices.long(),True)
            return
        slots=[];values=[];overflows=[]
        flag_slots=[];flags=[];version_slots=[];version_counts=[];prompt_counts=[]
        for batch_row,req in enumerate(batch.reqs):
            slot=int(req.kv.req_pool_idx);owner=str(req.rid)
            if self.owners.get(slot)!=owner:
                self.owners[slot]=owner
                self.past_counts[slot]=0
                self.members[slot]=set(t for t in req.origin_input_ids if t not in self.hot_cpu)
                self.stats['scanned_verified_tokens']+=len(req.origin_input_ids)
                changed=True
            else:changed=False
            # CPU output lists can lag behind or outlive a captured batch view.
            # Limit updates to this batch's committed prefix, never its future.
            if batch.seq_lens_cpu is not None:
                permitted=max(0,int(batch.seq_lens_cpu[batch_row])-len(req.origin_input_ids))
                end=min(permitted,len(req.output_ids))
            else:
                # The last emitted token can still be this round's pending
                # root. Exclude it; a GPU version check below additionally
                # rejects pools newer than this round's committed prefix.
                end=max(0,len(req.output_ids)-1)
            old=self.past_counts[slot]
            assert len(req.output_ids)>=old,'Verified output count moved backwards'
            if old>end:
                flag_slots.append(slot);flags.append(True)
                continue  # no use of a pool newer than this round's prefix
            new=req.output_ids[old:end]
            self.stats['scanned_verified_tokens']+=len(new)
            size=len(self.members[slot])
            self.members[slot].update(t for t in new if t not in self.hot_cpu)
            self.past_counts[slot]=end
            version_slots.append(slot);version_counts.append(end);prompt_counts.append(len(req.origin_input_ids))
            changed=changed or len(self.members[slot])!=size
            flag_slots.append(slot);flags.append(len(self.members[slot])>self.capacity)
            if changed:
                ids=sorted(self.members[slot]);overflow=len(ids)>self.capacity
                # Never silently truncate and pretend the full set was queried.
                values.append((ids if not overflow else [])+[-1]*(self.capacity if overflow else self.capacity-len(ids)))
                slots.append(slot);overflows.append(overflow)
        if slots:
            ix=torch.tensor(slots,device=self.device,dtype=torch.long)
            self.pool.index_copy_(0,ix,torch.tensor(values,device=self.device,dtype=torch.long))
            self.overflow.index_copy_(0,ix,torch.tensor(overflows,device=self.device,dtype=torch.bool))
            self.stats['pool_updates']+=len(slots)
            self.stats['overflow_updates']+=sum(overflows)
        if flag_slots:
            self.overflow.index_copy_(0,torch.tensor(flag_slots,device=self.device,dtype=torch.long),
                torch.tensor(flags,device=self.device,dtype=torch.bool))
        if version_slots:
            ix=torch.tensor(version_slots,device=self.device,dtype=torch.long)
            self.known_counts.index_copy_(0,ix,torch.tensor(version_counts,device=self.device,dtype=torch.long))
            self.prompt_lengths.index_copy_(0,ix,torch.tensor(prompt_counts,device=self.device,dtype=torch.long))
        # The bonus token is already known before this round. No D2H readback.
        self.roots.index_copy_(0,batch.req_pool_indices.long(),batch.spec_info.bonus_tokens.long())

    def pool_enabled(self,slots,prefix_lengths):
        ix=slots.long()
        return ~self.overflow[ix] & (self.known_counts[ix]<=prefix_lengths-self.prompt_lengths[ix])

    def query(self,h,ids,slots,valid,remaining,width,prefix_lengths=None):
        rows=h.shape[0]
        assert h.shape[1]==4096 and rows==len(slots)*width
        normalized=self.norm(h.clone())
        if isinstance(normalized,tuple):normalized=normalized[0]
        if prefix_lengths is None:
            # Component callers with manually populated pools provide no
            # lifecycle metadata; production propose requires an explicit GPU
            # committed-prefix tensor.
            available=~self.overflow[slots.long()]
        else:available=self.pool_enabled(slots,prefix_lengths)
        enabled=valid & available.repeat_interleave(width)
        transformed=normalized if self.cached else normalized@self.A.T
        if self.cached:
            from sglang.srt.speculative.row_readout import packet as fused_row_packet
            packet=fused_row_packet(normalized,self.query_weight,self.draft,ids,self.pool,self.roots,
                self.hot,slots,enabled,remaining,width,self.target_low,self.target_high,
                self.draft_low,self.draft_high,self.vocab)
        else:
            scores=torch.empty((rows,self.capacity+1),device=self.device,dtype=torch.float32)
            _query[(rows,self.capacity+1)](transformed,self.query_weight,
                self.pool,self.roots,self.hot,slots,enabled,remaining,scores,
                WIDTH=width,CAP=self.capacity,D=4096,XS=transformed.stride(0),
                TWS=self.query_weight.stride(0),TLO=self.target_low,THI=self.target_high,
                VOCAB=self.vocab,BLOCK=4096,num_warps=4)
            hotlogit=torch.empty((rows,2),device=self.device,dtype=torch.float32)
            _native_pair[(rows,2)](normalized,self.draft,ids,self.hot,hotlogit,D=4096,
                HS=normalized.stride(0),WS=self.draft.stride(0),LO=self.draft_low,HI=self.draft_high,
                BLOCK=4096,num_warps=4)
            packet=pack_local(scores,self.pool,self.roots,slots,hotlogit,width)
        if self.group.world_size>1:
            gathered=torch.empty((self.group.world_size*rows,4),device=self.device,dtype=torch.float32)
            self.group.all_gather_into_tensor(gathered,packet)
            gathered=gathered.reshape(self.group.world_size,rows,4)
        else:gathered=packet[None]
        return merge_global(gathered,available,width,self.audit)

    def propose(self,h,p,ids,slots,valid,remaining,step,prefix_lengths):
        width=1 if step==0 else 2
        assert prefix_lengths is not None,'Production query needs GPU prefix versions'
        best,cold,native,usable=self.query(h,ids,slots,valid.flatten(),remaining,width,prefix_lengths)
        # Exact FP64 features, standardization, native GEMM and softmax.
        x=normalized_features(p,best,native,usable,self.mean,self.scale,step)
        posterior=(x@self.coef.T+self.intercept).softmax(-1)
        use=usable & valid.flatten() & (remaining.repeat_interleave(width)>0) & (posterior[:,2]>posterior[:,1])
        repaired_ids=torch.stack((ids[:,0],torch.where(use,cold,ids[:,1])),1)
        # Candidate-enabled rows use estimated Target outcome probabilities.
        # Original rows retain their probabilities exactly. This is greedy-only
        # selection scoring, NOT a normalized proposal for rejection sampling.
        adjusted=torch.stack((posterior[:,0],posterior[:,2]),1).float()
        repaired_p=torch.where(use[:,None],adjusted,p)
        fingerprint=decision_audit(repaired_p,repaired_ids,use,valid.flatten(),usable,remaining,width,self.audit)
        if self.group.world_size>1:
            gathered=torch.empty((self.group.world_size*2,),device=self.device,dtype=torch.int64)
            self.group.all_gather_into_tensor(gathered,fingerprint)
            agreement(gathered,self.audit)
        return repaired_p,repaired_ids

    def record_admitted(self,scores,tokens):
        legal=torch.isfinite(scores).reshape(tokens.shape)
        self.audit[3].add_((legal & (self.hot[tokens.long()]<0)).sum())

    def record_verified_tree(self,tokens):
        self.audit[10].add_((self.hot[tokens.long()]<0).sum())

    def record_verified(self,tokens,accept_lens):
        # run_eagle_verify finalizes the accepted path at the front of each
        # eight-token block. accept_lens includes its final Target bonus.
        bs=accept_lens.numel()
        if not bs:return
        values=tokens.reshape(bs,-1).long()
        count=accept_lens.long()-1
        mask=torch.arange(values.shape[1],device=self.device)[None,:]<count[:,None]
        safe=values.clamp(0,self.vocab-1)
        self.audit[4].add_(bs)
        self.audit[5].add_(count.sum())
        self.audit[6].add_((mask & (values>=0) & (values<self.vocab) & (self.hot[safe]<0)).sum())

    def snapshot(self):
        # Called by get_internal_state only outside the measured workload.
        return dict(cpu=dict(self.stats),gpu=dict(zip(self.audit_names,self.audit.cpu().tolist())),
            definition='Cumulative counts include warmup/capture; subtract before/after workload snapshots.',
            capacity=self.capacity,cached_weight_projection=self.cached,
            raw_generation_budget=self.raw_budget,verification_draft_nodes=7)
