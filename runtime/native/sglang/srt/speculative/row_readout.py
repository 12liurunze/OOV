"""Fuse cached-context readout, native-pair readout and local packing.

One CTA per hidden row. Iterate only owned, nonpadding context candidates.
No tree/scoring/calibration change, no CPU count read, no CUDA conditional node.
"""
import torch,triton as tr,triton.language as tl
@tr.jit
def _packet(X,TW,DW,IDS,POOL,ROOT,HOT,SLOTS,VALID,REM,O,
 WIDTH:tl.constexpr,CAP:tl.constexpr,D:tl.constexpr,XS:tl.constexpr,TWS:tl.constexpr,DWS:tl.constexpr,
 TLO:tl.constexpr,THI:tl.constexpr,DLO:tl.constexpr,DHI:tl.constexpr,VOCAB:tl.constexpr,B:tl.constexpr,BC:tl.constexpr):
 row=tl.program_id(0);req=row//WIDTH;slot=tl.load(SLOTS+req).to(tl.int64)
 live=tl.load(VALID+row)&(tl.load(REM+req)>0)
 c=tl.arange(0,BC);t=tl.load(POOL+slot*CAP+c,c<CAP,other=2**30)
 root=tl.load(ROOT+slot);cold=tl.minimum(tl.min(t,0),root)
 safe=tl.maximum(0,tl.minimum(VOCAB-1,t));hot=tl.load(HOT+safe)
 owned=live&(c<CAP)&(hot<0)&(t>=TLO)&(t<THI)
 order=tl.cumsum(owned.to(tl.int32),0);count=tl.sum(owned.to(tl.int32),0)
 k=tl.arange(0,B);x=tl.load(X+row*XS+k,k<D,0).to(tl.float32)
 best=-float('inf');nan_seen=False
 for i in range(count):
  token=tl.sum(tl.where(owned&(order==i+1),t,0).to(tl.int64),0)
  w=tl.load(TW+(token-TLO)*TWS+k,k<D,0).to(tl.float32)
  score=tl.sum(x*w,0).to(tl.bfloat16).to(tl.float32)
  nan_seen=nan_seen|(score!=score)
  greater=score>best;equal=score==best
  cold=tl.where(greater,token,tl.where(equal,tl.minimum(cold,token),cold))
  best=tl.maximum(best,score)
 safe_root=tl.maximum(0,tl.minimum(VOCAB-1,root));root_cold=tl.load(HOT+safe_root)<0
 if live&root_cold&(root>=TLO)&(root<THI):
  w=tl.load(TW+(root-TLO).to(tl.int64)*TWS+k,k<D,0).to(tl.float32)
  score=tl.sum(x*w,0).to(tl.bfloat16).to(tl.float32)
  nan_seen=nan_seen|(score!=score)
  greater=score>best;equal=score==best
  cold=tl.where(greater,root,tl.where(equal,tl.minimum(cold,root),cold));best=tl.maximum(best,score)
 best=tl.where(nan_seen,float('nan'),best);cold=tl.where(nan_seen,2**30,cold)
 tl.store(O+row*4,best);tl.store(O+row*4+1,cold.to(tl.float32))
 for which in tl.static_range(2):
  token=tl.load(IDS+row*2+which);index=tl.load(HOT+token);score=0.
  if (index>=DLO)&(index<DHI):
   w=tl.load(DW+(index-DLO).to(tl.int64)*DWS+k,k<D,0).to(tl.float32)
   score=tl.sum(x*w,0).to(tl.bfloat16).to(tl.float32)
  tl.store(O+row*4+2+which,score)

def packet(x,tw,dw,ids,pool,roots,hot,slots,enabled,remaining,width,target_low,target_high,draft_low,draft_high,vocab=128256):
 rows,d=x.shape;cap=pool.shape[1];assert d==4096 and x.dtype==tw.dtype==dw.dtype==torch.bfloat16
 out=torch.empty((rows,4),device=x.device,dtype=torch.float32)
 _packet[(rows,)](x,tw,dw,ids,pool,roots,hot,slots,enabled,remaining,out,
  WIDTH=width,CAP=cap,D=d,XS=x.stride(0),TWS=tw.stride(0),DWS=dw.stride(0),
  TLO=target_low,THI=target_high,DLO=draft_low,DHI=draft_high,VOCAB=vocab,B=4096,BC=tr.next_power_of_2(cap),num_warps=4)
 return out
