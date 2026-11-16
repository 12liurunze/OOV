"""Verification-eligibility bound with fixed CUDA Graph shapes.

At most26 computed children; final7 verification nodes. Let tau be the seventh
largest score currently archived. For a prefix of score s<=tau, every descendant
has score <=s, so no later descendant can enter stable top7. Its continuation
can be disabled. This proves a score-selection property, NOT Target acceptance.
Context-supported candidates and their frozen calibrator remain unchanged.
Branch-and-bound itself is not claimed novel or independent of prior tree work.

Optional hardened state preserves seven *real generated* verification nodes.
Terminal pruning is deferred until that scaffold exists.  Deferred work is
counted; these fixed-shape compatibility nodes are not useful output gains.
The caller MUST check error_flags across TP ranks before Target verification.
"""
import torch
import triton as _select_triton
import triton.language as _stl

@_select_triton.jit
def _bound_select_hidden(P, T, S, V, R, H, IDS, NS, RS, NV, NR, PAR, OH, ARCH,
                           FT, NFT, CT, AU, ERR, DEFER, HORIZON, REAL,
                           FIRST: _stl.constexpr, STEP: _stl.constexpr,
                           C: _stl.constexpr, HD: _stl.constexpr,
                           HS0: _stl.constexpr, HS1: _stl.constexpr,
                           BLOCK: _stl.constexpr, HARDEN: _stl.constexpr,
                           HAS_TERMINAL: _stl.constexpr, HAS_HORIZON: _stl.constexpr,
                           HAS_REAL: _stl.constexpr, VOCAB: _stl.constexpr):
    b = _stl.program_id(0)
    hb = _stl.program_id(1)
    j = _stl.arange(0, C)
    parent = j // 2
    width: _stl.constexpr = C // 2
    p = _stl.load(P + b*C + j).to(_stl.float32)
    valid = _stl.load(V + b*width + parent)
    rem = _stl.load(R + b)
    if FIRST:
        ps = _stl.full((C,), 1., _stl.float32)
    else:
        ps = _stl.load(S + b*2 + parent).to(_stl.float32)
    real = True
    if HAS_REAL:
        real = _stl.load(REAL+b)
    parent_terminal = _stl.full((C,), False, _stl.int1)
    terminal = _stl.full((C,), False, _stl.int1)
    horizon = 2147483647
    if HARDEN:
        parent_terminal = _stl.load(FT+b*width+parent)
        if HAS_TERMINAL:
            terminal = _stl.load(CT+b*C+j)
        if HAS_HORIZON:
            horizon = _stl.load(HORIZON+b)
        # Keep capture shapes defined even on bad inputs; errors are fatal at
        # the worker boundary, never silently interpreted as repaired scores.
        p_finite = (p == p) & (_stl.abs(p) != float('inf'))
        ps_finite = (ps == ps) & (_stl.abs(ps) != float('inf'))
        bad_p = valid & (rem > 0) & ~p_finite
        bad_range = valid & (rem > 0) & ((p < 0.) | (p > 1.))
        bad_ps = valid & (rem > 0) & (~ps_finite | (ps < 0.) | (ps > 1.))
        tok = _stl.load(T+b*C+j)
        sibling_tok = _stl.load(T+b*C+(j ^ 1))
        bad_token = valid & (rem > 0) & (tok < 0)
        if VOCAB > 0:
            bad_token = bad_token | (valid & (rem > 0) & (tok >= VOCAB))
        bits = (_stl.sum(bad_p.to(_stl.int32),0)>0).to(_stl.int32)
        bits |= (_stl.sum(bad_range.to(_stl.int32),0)>0).to(_stl.int32)*2
        bits |= (_stl.sum(bad_ps.to(_stl.int32),0)>0).to(_stl.int32)*4
        bits |= (_stl.sum(bad_token.to(_stl.int32),0)>0).to(_stl.int32)*8
        duplicate = valid & (rem > 0) & ((j % 2)==0) & (tok==sibling_tok)
        bits |= (_stl.sum(duplicate.to(_stl.int32),0)>0).to(_stl.int32)*128
        if hb == 0:
            _stl.atomic_or(ERR+b, _stl.where(real,bits,0))
        p = _stl.minimum(_stl.maximum(_stl.where(p_finite,p,0.),0.),1.)
        ps = _stl.minimum(_stl.maximum(_stl.where(ps_finite,ps,0.),0.),1.)
    # Candidate eligibility is not inferred from p1 alone.
    allowed = valid & (rem > 0)
    score = ps * p
    # Children of an already computed parent are free to retain. Confidence
    # controls continuation, not whether its available second child survives.
    archive_rank = _stl.cumsum(valid.to(_stl.int32), 0) - 1
    admitted = valid & (archive_rank < rem)
    if hb == 0:
        _stl.store(RS + b*C + j, _stl.where(admitted, score, -float('inf')))
        _stl.store(NR + b, rem - _stl.sum(admitted.to(_stl.int64), 0))
    # Archive threshold includes the children just computed by this call.
    # Only earlier cells are read, so hb0 writes cannot race with other blocks.
    offset: _stl.constexpr = 0 if FIRST else 2 + (STEP-1)*4
    a = _stl.arange(0, 32)
    previous = _stl.load(ARCH+b*26+a, a<offset, other=-float('inf'))
    match = a[:,None] == offset+j[None,:]
    fresh = _stl.sum(_stl.where(match, _stl.where(admitted,score,-float('inf'))[None,:],0.),1)
    current = (a>=offset)&(a<offset+C)
    archived = _stl.where(current,fresh,previous)
    useful = admitted & ~parent_terminal & (STEP+1 <= horizon)
    continue_allowed = allowed
    ready = False
    if HARDEN:
        prior_useful = _stl.load(AU+b*26+a,a<offset,other=False)
        new_useful = _stl.sum(_stl.where(match,useful[None,:].to(_stl.int32),0),1)>0
        archived_useful = _stl.where(current,new_useful,prior_useful)
        finite_count = _stl.sum(((archived == archived) & (_stl.abs(archived) != float('inf'))).to(_stl.int32),0)
        ready = finite_count >= 7
        # Before seven generated nodes exist, preserve the legacy legal
        # scaffold. Afterwards never extend a terminal / out-of-horizon path.
        continue_allowed = allowed & (~ready | (~parent_terminal & ~terminal & (STEP+1 < horizon)))
        ranked_archive = _stl.where(archived_useful,archived,-float('inf'))
        if hb == 0:
            _stl.store(AU+b*26+a,archived_useful,current)
    else:
        ranked_archive = archived
    ordered = _stl.sort(ranked_archive,descending=True)
    cutoff = _stl.sum(_stl.where(a==6,ordered,0.),0)
    if hb == 0:
        _stl.store(ARCH+b*26+a,archived,current)
    priority = _stl.where(continue_allowed, score, -float('inf'))
    x, y = priority[:, None], priority[None, :]
    ix, iy = j[:, None], j[None, :]
    xn, yn = x != x, y != y
    before = ((yn & ~xn) | (~yn & ~xn & (y > x))
              | (((x == y) | (xn & yn)) & (iy < ix)))
    rank = _stl.sum(before.to(_stl.int32), 1)
    hcol = hb*BLOCK + _stl.arange(0, BLOCK)
    for slot in _stl.static_range(2):
        if FIRST:
            selected = slot
        else:
            selected = _stl.min(_stl.where(rank == slot, j, C), 0)
        ok = _stl.sum(((j == selected) & admitted & continue_allowed & (score > cutoff)).to(_stl.int32), 0) > 0
        safe = _stl.where(ok, selected, 0)
        if FIRST:
            row = b
            ss = _stl.sum(_stl.where(j==safe,score,0.),0)
        else:
            row = b*2 + safe//2
            ss = _stl.sum(_stl.where(j==safe,score,0.),0)
        if hb == 0:
            token = _stl.load(T + b*C + safe)
            if HARDEN:
                # Invalid source IDs already set a fatal flag. Keep later
                # captured Draft embedding loads in bounds until the host/TP
                # abort boundary. The archive retains the original bad ID;
                # this transient safety value MUST NOT reach verification.
                token = _stl.maximum(token,0)
                if VOCAB > 0:
                    token = _stl.minimum(token,VOCAB-1)
            _stl.store(IDS + b*2 + slot, token)
            _stl.store(NS + b*2 + slot, _stl.where(ok, ss, 0.))
            _stl.store(NV + b*2 + slot, ok)
            if HARDEN:
                stopped = _stl.sum(((j==safe)&(parent_terminal|terminal)).to(_stl.int32),0)>0
                _stl.store(NFT+b*2+slot,stopped & ok)
                deferred = ok & ~ready & (stopped | (STEP+1 >= horizon)) & real
                _stl.atomic_add(DEFER+b,deferred.to(_stl.int32))
            if not FIRST:
                _stl.store(PAR + b*2 + slot, safe + 2 + (STEP-1)*4)
        if HD > 0:
            value = _stl.load(H + row*HS0 + hcol*HS1, hcol < HD, other=0)
            _stl.store(OH + (b*2+slot)*HD + hcol, value, hcol < HD)
    if FIRST:
        if hb == 0:
            k = _stl.arange(0, 4)
            _stl.store(PAR + b*3 + k, k-1, mask=k<3)


def create_tree_state(bs, device, *, error_flags=None, target_vocab_size=0):
    """Call inside each captured/eager round so reset operations are replayed.

    error_flags can be a stable external buffer read after Graph replay.
    Keeping that external reference is the caller's responsibility.
    """
    if error_flags is None:
        error_flags = torch.zeros((bs,), dtype=torch.int32, device=device)
    else:
        assert error_flags.shape == (bs,) and error_flags.dtype == torch.int32
        error_flags.zero_()
    return dict(
        frontier_terminal=torch.zeros((bs,1),dtype=torch.bool,device=device),
        archive_useful=torch.zeros((bs,26),dtype=torch.bool,device=device),
        error_flags=error_flags,
        deferred_count=torch.zeros((bs,),dtype=torch.int32,device=device),
        terminal_fill_count=torch.zeros((bs,),dtype=torch.int32,device=device),
        target_vocab_size=int(target_vocab_size),
    )


def select_frontier(i, topk_p, topk_index, hidden_states, scores, topk,
                    frontier_valid, remaining, *, archive_scores,
                    tree_state=None, candidate_terminal=None,
                    output_horizon=None, real_request_mask=None):
    bs = remaining.shape[0]
    assert archive_scores is not None and archive_scores.shape==(bs,26)
    width = 1 if i == 0 else 2
    assert topk == 2 and 0 <= i < 7
    assert frontier_valid.shape == (bs,width)
    if tree_state is not None:
        assert tree_state['frontier_terminal'].shape == (bs,width)
        assert tree_state['archive_useful'].shape == (bs,26)
    if candidate_terminal is not None:
        assert tree_state is not None and candidate_terminal.numel() == bs*width*2
        assert candidate_terminal.dtype == torch.bool and candidate_terminal.is_contiguous()
    if output_horizon is not None:
        assert tree_state is not None and output_horizon.shape == (bs,)
    if real_request_mask is not None:
        assert real_request_mask.shape == (bs,) and real_request_mask.dtype == torch.bool
    dev = topk_p.device
    ids = torch.empty((bs*2,), device=dev, dtype=topk_index.dtype)
    ns = torch.empty((bs, 2), device=dev, dtype=torch.float32)
    rs = torch.empty((bs, width, 2), device=dev, dtype=torch.float32)
    nv = torch.empty((bs, 2), device=dev, dtype=torch.bool)
    nr = torch.empty_like(remaining)
    next_terminal = torch.empty((bs,2),dtype=torch.bool,device=dev) if tree_state is not None else ns
    parents = torch.empty((bs, 3 if i == 0 else 2), device=dev, dtype=torch.int64)
    has_hidden = hidden_states is not None and hidden_states.numel() > 0
    hd = hidden_states.shape[1] if has_hidden else 0
    h = torch.empty((bs*2, hd), device=dev, dtype=hidden_states.dtype) if has_hidden else hidden_states
    block = _select_triton.next_power_of_2(hd) if 0 < hd <= 4096 else 256
    warps = 8 if block >= 2048 else 4
    if bs:
        _bound_select_hidden[(bs, max(1, _select_triton.cdiv(hd, block)))](
            topk_p, topk_index, topk_p if i == 0 else scores,
            frontier_valid, remaining, hidden_states if has_hidden else topk_p,
            ids, ns, rs, nv, nr, parents, h if has_hidden else ns, archive_scores,
            tree_state['frontier_terminal'] if tree_state is not None else ns,
            next_terminal, candidate_terminal if candidate_terminal is not None else ns,
            tree_state['archive_useful'] if tree_state is not None else ns,
            tree_state['error_flags'] if tree_state is not None else ns,
            tree_state['deferred_count'] if tree_state is not None else ns,
            output_horizon if output_horizon is not None else ns,
            real_request_mask if real_request_mask is not None else ns,
            FIRST=i == 0, STEP=i, C=width*2, HD=hd,
            HS0=hidden_states.stride(0) if has_hidden else 0,
            HS1=hidden_states.stride(1) if has_hidden else 0,
            BLOCK=block,HARDEN=tree_state is not None,
            HAS_TERMINAL=candidate_terminal is not None,HAS_HORIZON=output_horizon is not None,
            HAS_REAL=real_request_mask is not None,
            VOCAB=tree_state['target_vocab_size'] if tree_state is not None else 0,
            num_warps=warps, enable_fp_fusion=False)
    if tree_state is not None:
        tree_state['frontier_terminal'] = next_terminal
    return ids, h, ns, (rs, topk_index.reshape(bs, width*2), parents), nv, nr


def organize_archive(score_list,token_list,parents_list,num_draft_tokens):
    assert num_draft_tokens==8
    scores=torch.cat(score_list,dim=1).flatten(1)
    tokens=torch.cat(token_list,dim=1)
    # Conditional scores <= 1 make every child no larger than its parent.
    # Stable ties retain earlier (ancestor) nodes before later descendants.
    chosen=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:7]
    chosen=torch.sort(chosen,dim=1).values
    parents=torch.cat(parents_list[:-1],dim=1)
    return parents,chosen,torch.gather(tokens,1,chosen)

