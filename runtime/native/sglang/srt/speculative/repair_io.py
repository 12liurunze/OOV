"""Exact data movement/reductions for context repair; scoring stays unchanged."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

@tr.jit
def _features(P,BEST,NATIVE,USABLE,MEAN,SCALE,OUT,ROWS:tl.constexpr,STEP:tl.constexpr,BS:tl.constexpr,NS:tl.constexpr,BR:tl.constexpr):
    r=tl.program_id(0)*BR+tl.arange(0,BR)
    f=tl.arange(0,16)
    p1=tl.load(P+r*2,r<ROWS,other=0).to(tl.float64)
    p2=tl.load(P+r*2+1,r<ROWS,other=0).to(tl.float64)
    lo=tl.full((),1e-7,tl.float64);hi=tl.full((),1.,tl.float64)
    l1=libdevice.log(tl.minimum(tl.maximum(p1,lo),hi))
    l2=libdevice.log(tl.minimum(tl.maximum(p2,lo),hi))
    rest=(hi-p1)-p2
    lr=libdevice.log(tl.minimum(tl.maximum(rest,lo),hi))
    depth=tl.full((),STEP/6.,tl.float64)
    m=tl.load(BEST+r*BS,r<ROWS,other=0).to(tl.float64)*2.
    m=m-tl.load(NATIVE+r*NS+1,r<ROWS,other=0).to(tl.float64)
    m=m-2.
    usable=tl.load(USABLE+r,r<ROWS,other=False)
    m=tl.where(usable,m,tl.full((),0.,tl.float64))
    value=tl.where(f[None,:]==0,l1[:,None],tl.full((BR,16),0.,tl.float64))
    value=tl.where(f[None,:]==1,l2[:,None],value)
    value=tl.where(f[None,:]==2,lr[:,None],value)
    value=tl.where(f[None,:]==3,depth,value)
    value=tl.where(f[None,:]==4,p1[:,None],value)
    value=tl.where(f[None,:]==5,p2[:,None],value)
    value=tl.where(f[None,:]==6,(p1-p2)[:,None],value)
    value=tl.where(f[None,:]==7,depth*depth,value)
    value=tl.where(f[None,:]==8,m[:,None],value)
    value=tl.where(f[None,:]==9,(m+l2)[:,None],value)
    value=tl.where(f[None,:]==10,(m*depth)[:,None],value)
    mean=tl.load(MEAN+f,f<11,other=0);scale=tl.load(SCALE+f,f<11,other=1)
    value=(value-mean[None,:])/scale[None,:]
    tl.store(OUT+r[:,None]*11+f[None,:],value,(r[:,None]<ROWS)&(f[None,:]<11))

def normalized_features(p,best,native,usable,mean,scale,step):
    assert p.is_contiguous() and native.stride(1)==1
    out=torch.empty((len(p),11),device=p.device,dtype=torch.float64)
    _features[(tr.cdiv(len(p),16),)](p,best,native,usable,mean,scale,out,ROWS=len(p),STEP=step,BS=best.stride(0),NS=native.stride(0),BR=16,num_warps=4,enable_fp_fusion=False)
    return out

@tr.jit
def _pack(S,POOL,ROOT,SLOTS,NATIVE,OUT,ROWS:tl.constexpr,CAP:tl.constexpr,WIDTH:tl.constexpr,BR:tl.constexpr,BC:tl.constexpr):
    r=tl.program_id(0)*BR+tl.arange(0,BR)
    c=tl.arange(0,BC)
    slot=tl.load(SLOTS+r//WIDTH,r<ROWS,other=0)
    score=tl.load(S+r[:,None]*(CAP+1)+c[None,:],(r[:,None]<ROWS)&(c[None,:]<=CAP),other=-float('inf'))
    best=tl.max(score,1)
    tok=tl.load(POOL+slot[:,None]*CAP+c[None,:],(r[:,None]<ROWS)&(c[None,:]<CAP),other=2**30)
    root=tl.load(ROOT+slot,r<ROWS,other=2**30)
    tok=tl.where(c[None,:]==CAP,root[:,None],tok)
    cold=tl.min(tl.where((c[None,:]<=CAP)&(score==best[:,None]),tok,2**30),1)
    h0=tl.load(NATIVE+r*2,r<ROWS,other=0)
    h1=tl.load(NATIVE+r*2+1,r<ROWS,other=0)
    tl.store(OUT+r*4,best,r<ROWS);tl.store(OUT+r*4+1,cold.to(tl.float32),r<ROWS)
    tl.store(OUT+r*4+2,h0,r<ROWS);tl.store(OUT+r*4+3,h1,r<ROWS)

def pack_local(scores,pool,roots,slots,native,width):
    rows=scores.shape[0];cap=pool.shape[1]
    out=torch.empty((rows,4),device=scores.device,dtype=torch.float32)
    _pack[(tr.cdiv(rows,4),)](scores,pool,roots,slots,native,out,ROWS=rows,CAP=cap,WIDTH=width,BR=4,BC=tr.next_power_of_2(cap+1))
    return out

@tr.jit
def _merge(G,AVAILABLE,REAL,OUT,COLD,USABLE,AUDIT,ROWS:tl.constexpr,WORLD:tl.constexpr,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    best=tl.full((BLOCK,),-float('inf'),tl.float32)
    for rank in tl.static_range(WORLD):
        s=tl.load(G+(rank*ROWS+r)*4,r<ROWS,other=-float('inf'))
        best=tl.maximum(best,s)
    cold=tl.full((BLOCK,),float(2**30),tl.float32)
    a=tl.full((BLOCK,),0.,tl.float32);b=tl.full((BLOCK,),0.,tl.float32)
    for rank in tl.static_range(WORLD):
        s=tl.load(G+(rank*ROWS+r)*4,r<ROWS,other=-float('inf'))
        t=tl.load(G+(rank*ROWS+r)*4+1,r<ROWS,other=float(2**30))
        cold=tl.minimum(cold,tl.where(s==best,t,float(2**30)))
        a=a+tl.load(G+(rank*ROWS+r)*4+2,r<ROWS,other=0)
        b=b+tl.load(G+(rank*ROWS+r)*4+3,r<ROWS,other=0)
    tl.store(OUT+r*4,best,r<ROWS);tl.store(OUT+r*4+2,a,r<ROWS);tl.store(OUT+r*4+3,b,r<ROWS)
    tl.store(COLD+r,cold.to(tl.int64),r<ROWS)
    tl.store(USABLE+r,(best!=float('inf'))&(best!=-float('inf'))&(best==best),r<ROWS)
    av=tl.load(AVAILABLE+r//WIDTH,r<ROWS,other=True)
    real=tl.load(REAL+r//WIDTH,r<ROWS,other=False)
    n=tl.sum(((r<ROWS)&real&(r%WIDTH==0)&~av).to(tl.int64),0)
    tl.atomic_add(AUDIT+7,n,sem='relaxed')

def merge_global(gathered,available,width,audit,real_request_mask=None):
    world,rows,_=gathered.shape
    if real_request_mask is None:
        real_request_mask=torch.ones((rows//width,),device=gathered.device,dtype=torch.bool)
    # Each native logit has exactly one owning vocabulary shard. Every other
    # shard contributes zero, so adding more shards preserves the logit.
    out=torch.empty((rows,4),device=gathered.device,dtype=torch.float32)
    cold=torch.empty((rows,),device=gathered.device,dtype=torch.long)
    usable=torch.empty((rows,),device=gathered.device,dtype=torch.bool)
    _merge[(tr.cdiv(rows,256),)](gathered,available,real_request_mask,out,cold,usable,audit,ROWS=rows,WORLD=world,WIDTH=width,BLOCK=256,enable_fp_fusion=False)
    return out[:,0],cold,out[:,2:4],usable

@tr.jit
def _decision(P,IDS,USE,VALID,USABLE,REM,AUDIT,FP,ROWS:tl.constexpr,WIDTH:tl.constexpr,BLOCK:tl.constexpr,FINGERPRINT:tl.constexpr):
    r=tl.arange(0,BLOCK);ok=r<ROWS
    if FINGERPRINT:
        a=tl.load(IDS+r*2,ok,other=0).to(tl.int64)
        b=tl.load(IDS+r*2+1,ok,other=0).to(tl.int64)
        p0=tl.load(P+r*2,ok,other=0).to(tl.int32,bitcast=True).to(tl.int64)
        p1=tl.load(P+r*2+1,ok,other=0).to(tl.int32,bitcast=True).to(tl.int64)
        row=(r+1).to(tl.int64)
        u=a*131+b+(p0%16777213)*5+(p1%16777213)*7
        v=(a+17)*(a+17)+3*(b+17)*(b+17)+(p0%16777199)*11+(p1%16777199)*13
        x=tl.sum(tl.where(ok,u*row,0),0)%16777213
        y=tl.sum(tl.where(ok,v*row,0),0)%16777199
        tl.store(FP,x);tl.store(FP+1,y)
    active=tl.load(VALID+r,ok,other=False)&(tl.load(REM+r//WIDTH,ok,other=0)>0)
    usable=tl.load(USABLE+r,ok,other=False)
    use=tl.load(USE+r,ok,other=False)
    tl.atomic_add(AUDIT,tl.sum((ok&active).to(tl.int64),0),sem='relaxed')
    tl.atomic_add(AUDIT+1,tl.sum((ok&active&usable).to(tl.int64),0),sem='relaxed')
    tl.atomic_add(AUDIT+2,tl.sum((ok&use).to(tl.int64),0),sem='relaxed')

def decision_audit(p, ids, use, valid, usable, remaining, width, audit, *, fingerprint=True):
    rows = len(ids)
    fp = torch.empty((2,), device=p.device, dtype=torch.long) if fingerprint else None
    _decision[(1,)](p, ids, use, valid, usable, remaining, audit,
                    fp if fingerprint else audit, ROWS=rows, WIDTH=width,
                    BLOCK=tr.next_power_of_2(rows), FINGERPRINT=fingerprint, num_warps=4)
    return fp

@tr.jit
def _agreement(G,AUDIT,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK)
    value=tl.load(G+i,i<N,other=0)
    base=tl.load(G+i%2)
    mismatch=tl.sum(((i<N)&(value!=base)).to(tl.int32),0)>0
    tl.atomic_add(AUDIT+8,1,sem='relaxed');tl.atomic_add(AUDIT+9,mismatch.to(tl.int64),sem='relaxed')

def agreement(gathered,audit):
    _agreement[(1,)](gathered,audit,N=gathered.numel(),BLOCK=tr.next_power_of_2(gathered.numel()))
