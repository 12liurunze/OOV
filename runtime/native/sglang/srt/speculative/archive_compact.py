"""Stable prefix-closed top7, optionally preferring useful generated nodes.

No token is synthesized. A malformed archive sets a fatal error flag that the
worker must agree across TP ranks BEFORE Target verification.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _finish(A,T0,T1,T2,T3,T4,T5,T6,P0,P1,P2,P3,P4,P5,OP,OI,OT,
            USEFUL,ERR,FILL,REAL,HARDEN:tl.constexpr,HAS_REAL:tl.constexpr,VOCAB:tl.constexpr):
    b=tl.program_id(0);j=tl.arange(0,32)
    raw=tl.load(A+b*26+j,j<26,other=-float('inf'))
    finite=(raw==raw)&(tl.abs(raw)!=float('inf'))&(j<26)
    s=tl.where(finite,raw,-float('inf'))
    useful=tl.full((32,),False,tl.int1)
    if HARDEN:
        useful=tl.load(USEFUL+b*26+j,j<26,other=False)&finite
    x=s[:,None];y=s[None,:];ix=j[:,None];iy=j[None,:]
    before=(y>x)|((x==y)&(iy<ix))
    if HARDEN:
        ux=useful[:,None];uy=useful[None,:]
        before=(uy&~ux)|((uy==ux)&before)
    rank=tl.sum(before.to(tl.int32),1)
    keep=(j<26)&finite&(rank<7)
    pos=tl.cumsum(keep.to(tl.int32),0)-1
    t=tl.load(T0+b*2+j,j<2,0)
    t+=tl.load(T1+b*4+j-2,(j>=2)&(j<6),0)
    t+=tl.load(T2+b*4+j-6,(j>=6)&(j<10),0)
    t+=tl.load(T3+b*4+j-10,(j>=10)&(j<14),0)
    t+=tl.load(T4+b*4+j-14,(j>=14)&(j<18),0)
    t+=tl.load(T5+b*4+j-18,(j>=18)&(j<22),0)
    t+=tl.load(T6+b*4+j-22,(j>=22)&(j<26),0)
    p=tl.load(P0+b*3+j,j<3,0)
    p+=tl.load(P1+b*2+j-3,(j>=3)&(j<5),0)
    p+=tl.load(P2+b*2+j-5,(j>=5)&(j<7),0)
    p+=tl.load(P3+b*2+j-7,(j>=7)&(j<9),0)
    p+=tl.load(P4+b*2+j-9,(j>=9)&(j<11),0)
    p+=tl.load(P5+b*2+j-11,(j>=11)&(j<13),0)
    if HARDEN:
        real=True
        if HAS_REAL:
            real=tl.load(REAL+b)
        parent=tl.gather(p,j//2,axis=0)
        parent_present=tl.sum(((parent[:,None]==j[None,:])&keep[None,:]).to(tl.int32),1)>0
        root=(j<2)&(parent==-1)
        ancestor=(j>=2)&(parent>=0)&(parent<j)&parent_present
        bad_closure=keep&~(root|ancestor)
        parent_safe=tl.maximum(0,tl.minimum(parent,31))
        parent_score=tl.gather(s,parent_safe,axis=0)
        parent_useful=tl.gather(useful,parent_safe,axis=0)
        bad_source=finite&((s<0.)|(s>1.)|((j>=2)&((parent<0)|(parent>=j)|(s>parent_score)))|((j<2)&(parent!=-1)))
        bad_useful=useful&(j>=2)&~parent_useful
        invalid_score=(j<26)&((raw!=raw)|(raw==float('inf')))
        invalid_token=finite&(t<0)
        sibling_token=tl.gather(t,j ^ 1,axis=0)
        sibling_finite=tl.gather(finite,j ^ 1,axis=0)
        duplicate=finite&sibling_finite&((j%2)==0)&(t==sibling_token)
        if VOCAB>0:
            invalid_token=invalid_token|(finite&(t>=VOCAB))
        flags=(tl.sum(keep.to(tl.int32),0)!=7).to(tl.int32)*16
        flags|=(tl.sum(bad_closure.to(tl.int32),0)>0).to(tl.int32)*32
        flags|=(tl.sum((invalid_score|bad_source|bad_useful).to(tl.int32),0)>0).to(tl.int32)*64
        flags|=(tl.sum(invalid_token.to(tl.int32),0)>0).to(tl.int32)*8
        flags|=(tl.sum(duplicate.to(tl.int32),0)>0).to(tl.int32)*128
        tl.atomic_or(ERR+b,tl.where(real,flags,0))
        tl.store(FILL+b,tl.where(real,tl.sum((keep&~useful).to(tl.int32),0),0))
    tl.store(OI+b*7+pos,j,keep)
    tl.store(OT+b*7+pos,t,keep)
    tl.store(OP+b*13+j,p,j<13)


def organize_archive(score_list,token_list,parents_list,num_draft_tokens,archive_scores,
                     *,tree_state=None,real_request_mask=None):
    assert num_draft_tokens==8 and len(score_list)==len(token_list)==len(parents_list)==7
    bs=archive_scores.shape[0];assert archive_scores.shape==(bs,26)
    for i in range(7):
        assert token_list[i].shape==(bs,2 if i==0 else 4)
        assert parents_list[i].shape==(bs,3 if i==0 else 2)
    if tree_state is not None:
        assert tree_state['archive_useful'].shape==(bs,26)
        assert tree_state['error_flags'].shape==(bs,)
    if real_request_mask is not None:
        assert real_request_mask.shape==(bs,) and real_request_mask.dtype==torch.bool
    parents=torch.empty((bs,13),device=archive_scores.device,dtype=parents_list[0].dtype)
    # Sentinels are error-only outputs: never valid verification candidates.
    indices=torch.full((bs,7),-1,device=archive_scores.device,dtype=torch.int64)
    tokens=torch.full((bs,7),-1,device=archive_scores.device,dtype=token_list[0].dtype)
    if bs:
        _finish[(bs,)](archive_scores,*token_list,*parents_list[:-1],parents,indices,tokens,
            tree_state['archive_useful'] if tree_state is not None else archive_scores,
            tree_state['error_flags'] if tree_state is not None else archive_scores,
            tree_state['terminal_fill_count'] if tree_state is not None else archive_scores,
            real_request_mask if real_request_mask is not None else archive_scores,
            HARDEN=tree_state is not None,HAS_REAL=real_request_mask is not None,
            VOCAB=tree_state['target_vocab_size'] if tree_state is not None else 0,num_warps=4)
    return parents,indices,tokens
