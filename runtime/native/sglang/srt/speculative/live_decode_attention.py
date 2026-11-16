# Copyright 2023-2024 SGLang Team; Apache-2.0.
# Derived from the frozen decode_attention.py; see SOURCE.json and patch.diff.
"""Live-query Draft GQA. No KV cache layout, writes, or attention math changes."""
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.decode_attention import (
    _extract_kv_strides, _grouped_head_tiles, _GROUPED_BLOCK_H,
    _MIN_BLOCK_KV, tanh, unpack_aux_tensors,
)
_is_hip = False
_is_gfx1250 = False

@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    # Page-aware strides (used when PAGE_SIZE > 1).
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    HAS_MLA: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
    IS_GFX1250: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 1,
    SCORE_MOD: tl.constexpr = None,
    Aux0=None,
    aux0_stride_t=0,
    aux0_stride_h=0,
    aux0_len=0,
    forced_kv_splits=0,
    USE_FORCED: tl.constexpr = False,
    LIVE_IDS=None,
    LIVE_COUNT=None,
):
    # int64 to avoid overflow of flat offsets into Mid_O when
    # batch * num_head * max_kv_splits * head_dim exceeds 2**31.
    packed_row = tl.program_id(0).to(tl.int64)
    cur_batch = tl.load(LIVE_IDS + packed_row).to(tl.int64)
    live_count = tl.load(LIVE_COUNT)
    if packed_row < live_count:
        cur_head_id = tl.program_id(1)
        cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
        split_kv_id = tl.program_id(2)

        if BLOCK_H < kv_group_num:
            VALID_BLOCK_H: tl.constexpr = BLOCK_H
        else:
            VALID_BLOCK_H: tl.constexpr = kv_group_num
        cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
        mask_h = mask_h & (cur_head < q_head_num)

        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_dv = tl.arange(0, BLOCK_DV)
        mask_d = offs_d < Lk
        mask_dv = offs_dv < Lv

        cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
        cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
        # runtime, not constexpr: it only feeds the kv_len_per_split arithmetic below, so
        # a constexpr buys nothing and costs one stage-1 variant per cuda-graph ladder
        # rung (stage-2 does need it at compile time). Any count covers any length since
        # kv_len_per_split rounds cdiv(L, S) up; short sequences leave the tail empty.
        if USE_FORCED:
            kv_splits = forced_kv_splits
        else:
            kv_splits = tl.load(num_kv_splits + cur_batch)

        if xai_temperature_len > 0:
            offs_qidx = cur_batch_seq_len - 1
            xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
            _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
            xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

        offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

        if BLOCK_DPE > 0:
            offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
            mask_dpe = offs_dpe < Lk
            off_qpe = (
                cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
            )

        kv_len_per_split = (
            tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
        )
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

        # Hoist loop-invariant base offsets
        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        if BLOCK_DPE > 0:
            base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]
        if not HAS_MLA:
            base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

        if split_kv_end > split_kv_start:
            q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
            # gfx1250: triton tl.dot(fp8, fp8) returns garbage (~1e34+) for contraction
            # dim K>=128 (verified K=64 ok, K>=128 broken; bf16 fine at all K). The MLA
            # nope QK dot has K=512, so an fp8 KV cache MUST NOT be consumed as an fp8 dot
            # here: keep q in bf16 and upcast the fp8 K to bf16 for the dot. No-op for a
            # bf16 cache. (Do NOT "optimize" this back to q.to(fp8) on gfx1250.)
            # On all other platforms keep the original downcast of q to the KV dtype.
            # TODO: remove this branch once the gfx1250 fp8 tl.dot issue is resolved.
            if IS_GFX1250:
                q_k = q
            else:
                q_k = q.to(K_Buffer.dtype.element_ty)
            if BLOCK_DPE > 0:
                qpe = tl.load(
                    Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
                )
            for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                kv_loc = tl.load(
                    kv_indices + cur_batch_kv_start_idx + offs_n,
                    mask=offs_n < split_kv_end,
                    other=0,
                )
                # Page-aware KV address math (see _fwd_kernel_stage1).
                if PAGE_SIZE == 1:
                    offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
                else:
                    page_id = kv_loc // PAGE_SIZE
                    tok_in_p = kv_loc % PAGE_SIZE
                    offs_buf_k = (
                        page_id[None, :] * stride_buf_kpage
                        + tok_in_p[None, :] * stride_buf_ktok
                        + base_offs_k
                    )
                k = tl.load(
                    K_Buffer + offs_buf_k,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                    other=0.0,
                )
                if IS_GFX1250:
                    qk = tl.dot(q_k, k.to(q_k.dtype))
                else:
                    qk = tl.dot(q_k, k)
                if BLOCK_DPE > 0:
                    if PAGE_SIZE == 1:
                        offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe
                    else:
                        offs_buf_kpe = (
                            page_id[None, :] * stride_buf_kpage
                            + tok_in_p[None, :] * stride_buf_ktok
                            + base_offs_kpe
                        )
                    kpe = tl.load(
                        K_Buffer + offs_buf_kpe,
                        mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                        other=0.0,
                    )
                    qk += tl.dot(qpe, kpe.to(qpe.dtype))
                qk *= sm_scale_withk

                if logit_cap > 0:
                    qk = logit_cap * tanh(qk / logit_cap)

                if xai_temperature_len > 0:
                    qk *= xai_temperature_reg[:, None]

                if SCORE_MOD is not None:
                    qk = SCORE_MOD(
                        qk,
                        cur_batch_seq_len - 1,
                        offs_n[None, :],
                        cur_batch,
                        cur_head[:, None],
                        mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                        Aux0,
                        aux0_stride_t,
                        aux0_stride_h,
                        aux0_len,
                    )

                qk = tl.where(
                    mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
                )
                if HAS_MLA:
                    v = tl.trans(k)
                else:
                    if PAGE_SIZE == 1:
                        offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
                    else:
                        offs_buf_v = (
                            page_id[:, None] * stride_buf_vpage
                            + tok_in_p[:, None] * stride_buf_vtok
                            + base_offs_v
                        )
                    v = tl.load(
                        V_Buffer + offs_buf_v,
                        mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                        other=0.0,
                    )

                n_e_max = tl.maximum(tl.max(qk, 1), e_max)
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
                acc *= re_scale[:, None]
                # Keep the softmax weights p in fp32 for the P·V dot (do NOT downcast p to
                # bf16) on gfx1250. The bf16 downcast of p was the accuracy loss vs a torch
                # fp32 SDPA reference (recovers gfx1250 R1 GSM8K ~0.82 -> ~0.92 with
                # attention idealized). On other platforms restore the p.to(v.dtype) cast.
                # TODO: remove this branch once the gfx1250 bf16 P·V issue is resolved.
                if IS_GFX1250:
                    acc += tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)
                else:
                    acc += tl.dot(p.to(v.dtype), v)

                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max

            offs_mid_o = (
                cur_batch * stride_mid_ob
                + cur_head[:, None] * stride_mid_oh
                + split_kv_id * stride_mid_os
                + offs_dv[None, :]
            )

            tl.store(
                Att_Out + offs_mid_o,
                acc / e_sum[:, None],
                mask=(mask_h[:, None]) & (mask_dv[None, :]),
            )

            offs_mid_o_1 = (
                cur_batch * stride_mid_ob
                + cur_head * stride_mid_oh
                + split_kv_id * stride_mid_os
            ) // Lv

            tl.store(
                Att_Lse + offs_mid_o_1,
                e_max + tl.log(e_sum),
                mask=mask_h,
            )

        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    logit_cap,
    xai_temperature_len=-1,
    has_mla=False,
    use_pdl=False,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
    tune_mla: bool = False,
    forced_kv_splits: int = 0,
    live_plan=None,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    if _is_hip and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    # 4-D view exposes head_num at dim 2; legacy 3-D exposes
    # it at dim 1.
    kv_head_num = k_buffer.shape[-2]
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // kv_head_num

    BLOCK_H = _GROUPED_BLOCK_H
    MAX_KV_SPLITS = max_kv_splits
    head_tiles = _grouped_head_tiles(head_num, kv_group_num)

    extra_kargs = {}
    num_stages = 2
    num_warps = 4
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    if tune_mla:
        # num_warps reorders the fp32 accumulation, so whoever declined the batch-wide
        # count gets a batch-independent geometry too
        bucket = _mla_bucket(batch) if forced_kv_splits else _MLA_BUCKET_BATCH_FREE
        BLOCK, num_warps, num_stages = (
            _MLA_BLOCK_N,
            bucket.num_warps,
            bucket.num_stages,
        )

    # Blocks at or above the split count return immediately, so the grid shrinks too.
    grid = (batch, head_tiles, forced_kv_splits or MAX_KV_SPLITS)

    k_slot_stride, k_head_stride, k_page_stride, k_tok_stride = _extract_kv_strides(
        k_buffer, page_size
    )
    v_slot_stride, v_head_stride, v_page_stride, v_tok_stride = _extract_kv_strides(
        v_buffer, page_size
    )

    aux0, aux0_stride_t, aux0_stride_h, aux0_len = unpack_aux_tensors(
        score_mod, aux_tensors
    )

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_slot_stride,
        k_head_stride,
        v_slot_stride,
        v_head_stride,
        k_page_stride,
        k_tok_stride,
        v_page_stride,
        v_tok_stride,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        LIVE_IDS=live_plan[0],
        LIVE_COUNT=live_plan[1],
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        HAS_MLA=has_mla,
        USE_PDL=use_pdl,
        IS_GFX1250=_is_gfx1250,
        PAGE_SIZE=page_size,
        SCORE_MOD=score_mod,
        Aux0=aux0,
        aux0_stride_t=aux0_stride_t,
        aux0_stride_h=aux0_stride_h,
        aux0_len=aux0_len,
        forced_kv_splits=forced_kv_splits,
        USE_FORCED=forced_kv_splits > 0,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    v_scale,
    kv_indptr,
    num_kv_splits,
    sink_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    HAS_SINK: tl.constexpr,
    USE_PDL: tl.constexpr = False,
    FORCED_KV_SPLITS: tl.constexpr = 0,
    LIVE_IDS=None,
    LIVE_COUNT=None,
):
    # int64 to avoid overflow of flat offsets into Mid_O when
    # batch * num_head * max_kv_splits * head_dim exceeds 2**31.
    packed_row = tl.program_id(0).to(tl.int64)
    cur_batch = tl.load(LIVE_IDS + packed_row).to(tl.int64)
    live_count = tl.load(LIVE_COUNT)
    if packed_row < live_count:
        cur_head = tl.program_id(1)

        if USE_PDL:
            tl.extra.cuda.gdc_wait()

        cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
            kv_indptr + cur_batch
        )
        # Same count stage-1 used, or the two disagree about where split i starts. SPLIT_END
        # is a constexpr in both branches: a dynamic bound would merge the same partials
        # (stage-1 leaves the surplus splits masked out) but stops the unrolling, and
        # reassociating the fp32 reduction moves the result a few ULP off stock.
        if FORCED_KV_SPLITS > 0:
            kv_splits = FORCED_KV_SPLITS
            SPLIT_END: tl.constexpr = FORCED_KV_SPLITS
        else:
            kv_splits = tl.load(num_kv_splits + cur_batch)
            SPLIT_END: tl.constexpr = MAX_KV_SPLITS

        offs_d = tl.arange(0, BLOCK_DV)
        mask_d = offs_d < Lv

        e_sum = 0.0
        e_max = -float("inf")
        acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

        offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
        offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
        kv_len_per_split = (
            tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
        )

        for split_kv_id in tl.range(0, SPLIT_END, num_stages=2):
            split_kv_start = kv_len_per_split * split_kv_id
            split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

            if split_kv_end > split_kv_start:
                tv = tl.load(
                    Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
                )
                tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
                n_e_max = tl.maximum(tlogic, e_max)

                old_scale = tl.exp(e_max - n_e_max)
                acc *= old_scale
                exp_logic = tl.exp(tlogic - n_e_max)
                acc += exp_logic * tv

                e_sum = e_sum * old_scale + exp_logic
                e_max = n_e_max

        if HAS_SINK:
            cur_sink = tl.load(sink_ptr + cur_head)
            e_sum += tl.exp(cur_sink - e_max)

        tl.store(
            O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
            acc / e_sum * v_scale,
            mask=mask_d,
        )

    else:
        cur_head = tl.program_id(1)
        col = tl.arange(0, BLOCK_DV)
        tl.store(O + cur_batch*stride_obs + cur_head*stride_oh + col, 0., col < Lv)


def _decode_softmax_reducev_fwd(
    logits,
    lse,
    q,
    o,
    v_scale,
    v_buffer,
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
    sinks=None,
    use_pdl=False,
    forced_kv_splits: int = 0,
    live_plan=None,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits
    HAS_SINK = sinks is not None

    extra_kargs = {}
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        v_scale,
        kv_indptr,
        num_kv_splits,
        sinks,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        HAS_SINK=HAS_SINK,
        USE_PDL=use_pdl,
        FORCED_KV_SPLITS=forced_kv_splits,
        num_warps=4,
        LIVE_IDS=live_plan[0],
        LIVE_COUNT=live_plan[1],
        num_stages=2,
        **({"launch_pdl": True} if use_pdl else {}),
        **extra_kargs,
    )


def forward(q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits,
            attn_lse, num_kv_splits, max_kv_splits, sm_scale, k_scale, v_scale,
            logit_cap=0.0, sinks=None, xai_temperature_len=-1, has_mla=False,
            use_pdl=False, page_size=1, score_mod=None, aux_tensors=None,
            enable_lean=None, lean_Mp=None, lean_Lp=None, lean_Op=None,
            lean_locks=None, live_plan=None):
    assert live_plan is not None
    assert not has_mla and not use_pdl and page_size == 1
    assert sinks is None and score_mod is None and xai_temperature_len < 0
    assert q.dtype == k_buffer.dtype == v_buffer.dtype == torch.bfloat16
    assert q.shape[0] == live_plan[0].numel() == 1024
    _decode_grouped_att_m_fwd(q,k_buffer,v_buffer,attn_logits,attn_lse,
        kv_indptr,kv_indices,num_kv_splits,max_kv_splits,sm_scale*k_scale,
        logit_cap,page_size=page_size,live_plan=live_plan)
    _decode_softmax_reducev_fwd(attn_logits,attn_lse,q,o,v_scale,v_buffer,
        kv_indptr,num_kv_splits,max_kv_splits,live_plan=live_plan)
