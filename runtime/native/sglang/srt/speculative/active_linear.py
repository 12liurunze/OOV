"""Dependency-aware Draft projections for actualBS512 / TP2.

Keep both per-request QKV lanes: a currently dormant lane can become a valid
branch later and its Draft KV history is still read. Downstream query-only
projections need rows that can supply another candidate, not every physical
lane. No host counts, variable NCCL shapes, KV remapping or tree-rule changes.
Numerical and full-model gates remain mandatory; this is an isolated candidate.
"""
import contextlib
import torch
import triton as tr
import triton.language as tl
_plans=None
PHYSICAL_ROWS=1024

@tr.jit
def _pack_both(V,R,KI,QI,COUNT,M:tl.constexpr,B:tl.constexpr,HAS_REMAIN:tl.constexpr):
    i=tl.arange(0,B);row=i//2
    v=tl.load(V+i,i<M,other=0).to(tl.int32)
    a=tl.load(V+row*2,i<M,other=0);b=tl.load(V+row*2+1,i<M,other=0)
    kv=(a|b).to(tl.int32)
    if HAS_REMAIN:q=v & (tl.load(R+row,i<M,other=0)>0).to(tl.int32)
    else:q=v
    pk=tl.cumsum(kv,0);pq=tl.cumsum(q,0);nk=tl.sum(kv,0);nq=tl.sum(q,0)
    dk=tl.where(kv!=0,pk-1,nk+i-pk);dq=tl.where(q!=0,pq-1,nq+i-pq)
    tl.store(KI+dk,i,i<M);tl.store(QI+dq,i,i<M)
    tl.store(COUNT,nk);tl.store(COUNT+1,nq)

def make_plans(valid,remaining=None):
    m=valid.numel();ki=torch.empty((m,),device=valid.device,dtype=torch.int32);qi=torch.empty_like(ki)
    count=torch.empty((2,),device=valid.device,dtype=torch.int32)
    _pack_both[(1,)](valid,remaining if remaining is not None else valid,ki,qi,count,M=m,B=tr.next_power_of_2(m),HAS_REMAIN=remaining is not None,num_warps=4)
    return dict(kv=(ki,count[:1]),query=(qi,count[1:]))

@contextlib.contextmanager
def active_rows(valid,remaining=None):
    global _plans
    previous=_plans
    _plans=make_plans(valid,remaining) if valid is not None and valid.numel()==PHYSICAL_ROWS else None
    try:yield
    finally:_plans=previous

@tr.jit
def _mm(A,W,IDS,COUNT,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        AS0:tl.constexpr,AS1:tl.constexpr,WS0:tl.constexpr,WS1:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    mi=tl.program_id(0);ni=tl.program_id(1)
    rm=mi*BM+tl.arange(0,BM);rn=ni*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    rid=tl.load(IDS+rm,rm<M,other=0);active=tl.load(COUNT)
    acc=tl.full((BM,BN),0,tl.float32)
    if mi*BM<active:
        for block in range(tr.cdiv(K,BK)):
            kk=block*BK+rk
            x=tl.load(A+rid[:,None]*AS0+kk[None,:]*AS1,(rm[:,None]<active)&(kk[None,:]<K),other=0)
            w=tl.load(W+rn[None,:]*WS0+kk[:,None]*WS1,(rn[None,:]<N)&(kk[:,None]<K),other=0)
            acc=tl.dot(x,w,acc)
    tl.store(O+rid[:,None]*N+rn[None,:],acc,(rm[:,None]<M)&(rn[None,:]<N))

def matmul(x,w,plan,config=(64,128,64,4),out_dtype=None):
    bm,bn,bk,warps=config[:4];stages=config[4] if len(config)>4 else 3;m,k=x.shape;n=w.shape[0]
    out=torch.empty((m,n),device=x.device,dtype=out_dtype or x.dtype)
    _mm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](x,w,*plan,out,M=m,N=n,K=k,AS0=x.stride(0),AS1=x.stride(1),
        WS0=w.stride(0),WS1=w.stride(1),BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages)
    return out

def eligible(x,w,plan):
    return (plan is not None and x.is_cuda and x.dtype==torch.bfloat16 and w.dtype==x.dtype and x.ndim==w.ndim==2
        and x.shape[0]==PHYSICAL_ROWS and plan[0].numel()==PHYSICAL_ROWS and x.shape[1]==w.shape[1])

class ActiveQuantMethod:
    def __init__(self,original,config,kind):self.original=original;self.config=config;self.kind=kind
    def __getattr__(self,name):return getattr(self.original,name)
    def apply(self,layer,x,bias=None):
        plan=_plans[self.kind] if _plans is not None else None
        if bias is None and eligible(x,layer.weight,plan):return matmul(x,layer.weight,plan,self.config)
        return self.original.apply(layer,x,bias)

def install(model,configs):
    installed=[]
    for name,module in model.model.named_modules():
        if name not in configs:continue
        original=getattr(module,'quant_method',None)
        assert type(original).__name__=='UnquantizedLinearMethod',(name,type(original).__name__)
        # Before-attention QKV remains pair-live. Everything else selected here
        # is a query output and is not a source of this forward's stored KV.
        kind='kv' if name.endswith('qkv_proj') else 'query'
        module.quant_method=ActiveQuantMethod(original,tuple(configs[name]),kind);installed.append(name)
    if 'lm_head' in configs:
        original=model.logits_processor._compute_lm_head;cfg=tuple(configs['lm_head'])
        def head(hidden_states,lm_head,embedding_bias=None):
            plan=_plans['query'] if _plans is not None else None
            if embedding_bias is None and eligible(hidden_states,lm_head.weight,plan):
                dtype=torch.float32 if model.logits_processor.use_fp32_lm_head else hidden_states.dtype
                return matmul(hidden_states,lm_head.weight,plan,cfg,out_dtype=dtype)
            return original(hidden_states,lm_head,embedding_bias)
        model.logits_processor._compute_lm_head=head;installed.append('lm_head')
    assert set(installed)==set(configs),(installed,list(configs))
    return installed
