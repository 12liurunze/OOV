"""Confidence-controlled expansion with retained terminal insurance leaves.

Maximum raw pool: 2+6*4=26, from the same six physical two-row forwards.
Final verification budget: 7 Draft nodes. No hand-enumerated tree shapes.
Kernel derived from frozen SGLang Adaptive selector; Apache-2.0.
"""
import torch
import triton as _select_triton
import triton.language as _stl

@_select_triton.jit
def _archive_select_hidden(P, T, S, V, R, H, IDS, NS, RS, NV, NR, PAR, OH,
                           FIRST: _stl.constexpr, STEP: _stl.constexpr,
                           MARGIN: _stl.constexpr, THRESHOLD: _stl.constexpr,
                           C: _stl.constexpr, HD: _stl.constexpr,
                           HS0: _stl.constexpr, HS1: _stl.constexpr,
                           BLOCK: _stl.constexpr):
    b = _stl.program_id(0)
    hb = _stl.program_id(1)
    j = _stl.arange(0, C)
    parent = j // 2
    width: _stl.constexpr = C // 2
    p = _stl.load(P + b*C + j).to(_stl.float32)
    p1 = _stl.load(P + b*C + parent*2).to(_stl.float32)
    p2 = _stl.load(P + b*C + parent*2 + 1).to(_stl.float32)
    valid = _stl.load(V + b*width + parent)
    rem = _stl.load(R + b)
    if FIRST:
        ps = _stl.full((C,), 1., _stl.float32)
    else:
        ps = _stl.load(S + b*2 + parent).to(_stl.float32)
    confidence = p1 - p2 if MARGIN else p1
    threshold = _stl.full((), THRESHOLD, _stl.float32)
    branch = valid & (confidence < threshold) & (rem > 0)
    allowed = valid & ((j % 2 == 0) | branch)
    score = ps * p
    priority = _stl.where(allowed, score, -float('inf'))
    x, y = priority[:, None], priority[None, :]
    ix, iy = j[:, None], j[None, :]
    xn, yn = x != x, y != y
    before = ((yn & ~xn) | (~yn & ~xn & (y > x))
              | (((x == y) | (xn & yn)) & (iy < ix)))
    rank = _stl.sum(before.to(_stl.int32), 1)
    # Children of an already computed parent are free to retain. Confidence
    # controls continuation, not whether its available second child survives.
    archive_rank = _stl.cumsum(valid.to(_stl.int32), 0) - 1
    admitted = valid & (archive_rank < rem)
    if hb == 0:
        _stl.store(RS + b*C + j, _stl.where(admitted, score, -float('inf')))
        _stl.store(NR + b, rem - _stl.sum(admitted.to(_stl.int64), 0))
    hcol = hb*BLOCK + _stl.arange(0, BLOCK)
    for slot in _stl.static_range(2):
        if FIRST:
            selected = slot
        else:
            selected = _stl.min(_stl.where(rank == slot, j, C), 0)
        ok = _stl.sum(((j == selected) & admitted & allowed).to(_stl.int32), 0) > 0
        safe = _stl.where(ok, selected, 0)
        if FIRST:
            row = b
            ss = _stl.load(P + b*C + safe).to(_stl.float32)
        else:
            row = b*2 + safe//2
            ss = (_stl.load(P + b*C + safe).to(_stl.float32)
                  * _stl.load(S + b*2 + safe//2).to(_stl.float32))
        if hb == 0:
            token = _stl.load(T + b*C + safe)
            _stl.store(IDS + b*2 + slot, token)
            _stl.store(NS + b*2 + slot, _stl.where(ok, ss, 0.))
            _stl.store(NV + b*2 + slot, ok)
            if not FIRST:
                _stl.store(PAR + b*2 + slot, safe + 2 + (STEP-1)*4)
        if HD > 0:
            value = _stl.load(H + row*HS0 + hcol*HS1, hcol < HD, other=0)
            _stl.store(OH + (b*2+slot)*HD + hcol, value, hcol < HD)
    if FIRST:
        if hb == 0:
            k = _stl.arange(0, 4)
            _stl.store(PAR + b*3 + k, k-1, mask=k<3)


def select_frontier(i, topk_p, topk_index, hidden_states,
                                     scores, topk, frontier_valid, remaining,
                                     threshold, confidence_type='margin'):
    bs = remaining.shape[0]
    width = 1 if i == 0 else 2
    dev = topk_p.device
    ids = torch.empty((bs*2,), device=dev, dtype=topk_index.dtype)
    ns = torch.empty((bs, 2), device=dev, dtype=torch.float32)
    rs = torch.empty((bs, width, 2), device=dev, dtype=torch.float32)
    nv = torch.empty((bs, 2), device=dev, dtype=torch.bool)
    nr = torch.empty_like(remaining)
    parents = torch.empty((bs, 3 if i == 0 else 2), device=dev, dtype=torch.int64)
    has_hidden = hidden_states is not None and hidden_states.numel() > 0
    hd = hidden_states.shape[1] if has_hidden else 0
    h = torch.empty((bs*2, hd), device=dev, dtype=hidden_states.dtype) if has_hidden else hidden_states
    if bs:
        _archive_select_hidden[(bs, max(1, _select_triton.cdiv(hd, 256)))](
            topk_p, topk_index, topk_p if i == 0 else scores,
            frontier_valid, remaining, hidden_states if has_hidden else topk_p,
            ids, ns, rs, nv, nr, parents, h if has_hidden else ns,
            FIRST=i == 0, STEP=i, MARGIN=confidence_type == 'margin',
            THRESHOLD=threshold, C=width*2, HD=hd,
            HS0=hidden_states.stride(0) if has_hidden else 0,
            HS1=hidden_states.stride(1) if has_hidden else 0,
            BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return ids, h, ns, (rs, topk_index.reshape(bs, width*2), parents), nv, nr


def reference_frontier(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
    frontier_valid: torch.Tensor,
    remaining: torch.Tensor,
    threshold: float,
    confidence_type: str="margin",
):
    batch_size = remaining.shape[0]
    parent_width = 1 if i==0 else topk
    probs = topk_p.reshape(batch_size, parent_width, topk).float()
    tokens = topk_index.reshape(batch_size, parent_width * topk)
    if i == 0:
        parent_scores = torch.ones_like(probs[:,:,0])
    else:
        parent_scores = scores.reshape(batch_size, parent_width)
    if confidence_type == "margin":
        confidence = probs[:,:,0] - probs[:,:,1]
    elif confidence_type == "top1":
        confidence = probs[:,:,0]
    else:
        raise ValueError(f"Unknown confidence_type: {confidence_type}")
    # Decide independently for each valid parent at this step.
    # A previous split does not force its descendants to split again.
    branch_here = (
        frontier_valid
        & (confidence < threshold)
        & (remaining[:, None] > 0)
    )
    child_rank = torch.arange(topk, device=topk_p.device)
    allowed = frontier_valid.unsqueeze(-1) & (
        (child_rank[None, None, :] == 0) | branch_here.unsqueeze(-1)
    )
    path_scores = parent_scores.unsqueeze(-1) * probs
    flat_scores = path_scores.flatten(1)
    flat_allowed = allowed.flatten(1)
    priority = torch.where(flat_allowed, flat_scores, float("-inf"))
    order = torch.argsort(priority, dim=1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(
        1,
        order,
        torch.arange(order.shape[1], device=order.device)
        .unsqueeze(0)
        .expand_as(order),
    )
    archive_allowed = frontier_valid.unsqueeze(-1).expand_as(probs).flatten(1)
    archive_rank = archive_allowed.long().cumsum(1) - 1
    admitted = archive_allowed & (archive_rank < remaining.unsqueeze(1))
    remaining = remaining - admitted.sum(dim=1)
    recorded_scores = torch.where(admitted, flat_scores, float("-inf"))

    # Initial slots must keep native numbering: child j occupies slot j.
    if i == 0:
        selected = child_rank.unsqueeze(0).expand(batch_size, -1)
    else:
        selected = order[:, :topk]
    next_valid = (admitted & flat_allowed).gather(1, selected)

    # Inactive rows still execute a dense forward. Give them legal indices.
    # Their descendants remain invalid because next_valid travels with them.
    safe_selected = torch.where(next_valid, selected, 0)
    input_ids = tokens.gather(1, safe_selected).reshape(-1)
    next_scores = torch.where(
        next_valid, flat_scores.gather(1, safe_selected), 0.0
    )

    selected_hidden = hidden_states
    if hidden_states is not None and hidden_states.numel() > 0:
        if i == 0:
            selected_hidden = hidden_states.repeat_interleave(topk, dim=0)
        else:
            offsets = torch.arange(batch_size, device=probs.device)[:, None] * topk
            source_rows = safe_selected // topk + offsets
            selected_hidden = hidden_states[source_rows.reshape(-1)]

    if i == 0:
        parents = torch.arange(-1, topk, device=probs.device, dtype=torch.long)
        parents = parents.unsqueeze(0).expand(batch_size, -1)
    else:
        pool_offset = topk + (i - 1) * topk * topk
        parents = safe_selected + pool_offset

    tree_info = (
        recorded_scores.reshape(batch_size, parent_width, topk),
        tokens,
        parents,
    )
    return (
        input_ids,
        selected_hidden,
        next_scores,
        tree_info,
        next_valid,
        remaining,
    )


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
