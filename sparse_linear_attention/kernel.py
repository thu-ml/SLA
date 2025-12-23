""" 
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention}, 
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_preprocess(
    CK, V, S, Z,
    L: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    CD: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    CK += idx_bh * L * CD
    V += idx_bh * L * D
    S += (idx_bh * N_BLOCKS + idx_n) * CD * D
    Z += (idx_bh * N_BLOCKS + idx_n) * CD

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_cd = tl.arange(0, CD)

    c_k = tl.load(CK + offs_n[None, :] * CD + offs_cd[:, None], mask=offs_n[None, :] < L, other=0)
    v = tl.load(V + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < L)
    s = tl.dot(c_k, v).to(S.type.element_ty)
    z = tl.sum(c_k, axis=1).to(Z.type.element_ty)
    tl.store(S + offs_cd[:, None] * D + offs_d[None, :], s)
    tl.store(Z + offs_cd, z)


@triton.jit
def _attn_fwd(
    Q, K, V, CQ,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT, LSE, S, Z, SSUM, ZSUM,
    SQ, ZQ, DENOM, OS, OL,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    CD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_offset = idx_bh * L * D
    cqk_offset = idx_bh * L * CD
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    s_offset = idx_bh * N_BLOCKS * CD * D
    z_offset = idx_bh * N_BLOCKS * CD
    ssum_offset = idx_bh * CD * D
    zsum_offset = idx_bh * CD
    sq_offset = (idx_bh * M_BLOCKS + idx_m) * CD * D
    zq_offset = (idx_bh * M_BLOCKS + idx_m) * CD
    lse_offset = idx_bh * L
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_cd = tl.arange(0, CD)

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[None, :] * D + offs_d[:, None]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    OS_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    OL_ptrs = OL + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    CQ_ptrs = CQ + cqk_offset + offs_m[:, None] * CD + offs_cd[None, :]
    LUT_ptr = LUT + lut_offset
    S_ptrs = S + s_offset + offs_cd[:, None] * D + offs_d[None, :]
    Z_ptrs = Z + z_offset + offs_cd
    SSUM_ptrs = SSUM + ssum_offset + offs_cd[:, None] * D + offs_d[None, :]
    ZSUM_ptrs = ZSUM + zsum_offset + offs_cd
    SQ_ptrs = SQ + sq_offset + offs_cd[:, None] * D + offs_d[None, :]
    ZQ_ptrs = ZQ + zq_offset + offs_cd
    LSE_ptrs = LSE + lse_offset + offs_m
    DENOM_ptrs = DENOM + lse_offset + offs_m
    
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    s_q = tl.load(SSUM_ptrs)
    z_q = tl.load(ZSUM_ptrs)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N
        
        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        if L - idx_n * BLOCK_N < BLOCK_N:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        z_q -= tl.load(Z_ptrs + idx_n * CD)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        s_q -= tl.load(S_ptrs + idx_n * CD * D)
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m
    
    tl.store(SQ_ptrs, s_q)
    tl.store(ZQ_ptrs, z_q)

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)
    
    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < L)
    
    c_q = tl.load(CQ_ptrs, mask=offs_m[:, None] < L)
    if topk < N_BLOCKS:
        denom = tl.sum(c_q * z_q[None, :], axis=1, dtype=tl.float32)
    else:
        denom = tl.full([BLOCK_M], float("inf"), dtype=tl.float32)
    o_l = tl.dot(c_q, s_q.to(c_q.dtype)) / denom[:, None]
    
    tl.store(OL_ptrs, o_l.to(OL.type.element_ty), mask=offs_m[:, None] < L)
    tl.store(DENOM_ptrs, denom, mask=offs_m < L)


@triton.jit
def _attn_bwd_preprocess(
    CQ, OS, OL, DOS, DOL,
    QO, QD, DENOM, DELTAS, DELTAL,
    L, M_BLOCKS,
    D: tl.constexpr,
    CD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    CQ += idx_bh * L * CD
    OS += idx_bh * L * D
    OL += idx_bh * L * D
    DOS += idx_bh * L * D
    DOL += idx_bh * L * D
    QO += (idx_bh * M_BLOCKS + idx_m) * CD * D
    QD += (idx_bh * M_BLOCKS + idx_m) * CD
    DENOM += idx_bh * L
    DELTAS += idx_bh * L
    DELTAL += idx_bh * L

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_cd = tl.arange(0, CD)

    c_q = tl.load(CQ + offs_m[None, :] * CD + offs_cd[:, None], mask=offs_m[None, :] < L)
    o_s = tl.load(OS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    o_l = tl.load(OL + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    do_s = tl.load(DOS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    do_l = tl.load(DOL + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    denom = tl.load(DENOM + offs_m, mask=offs_m < L, other=float('inf'))
    
    delta_s = tl.sum(o_s * do_s, axis=1).to(DELTAS.type.element_ty)
    delta_l = tl.sum(o_l * do_l, axis=1).to(DELTAL.type.element_ty)
    c_q = c_q / denom[None, :]
    qo = tl.dot(c_q.to(do_l.dtype), do_l).to(QO.type.element_ty)
    qd = tl.sum(c_q * delta_l[None, :], axis=1)
    
    tl.store(DELTAS + offs_m, delta_s, mask=offs_m < L)
    tl.store(DELTAL + offs_m, delta_l, mask=offs_m < L)
    tl.store(QO + offs_cd[:, None] * D + offs_d[None, :], qo)
    tl.store(QD + offs_cd, qd)


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    Q, K, V, SQ, ZQ, LSE, DELTAS, DELTAL,
    DENOM, DOS, DOL, DQ, DCQ, LUT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    CD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_cd = tl.arange(0, CD)

    qkv_offset = idx_bh * L * D
    cqk_offset = idx_bh * L * CD
    sq_offset = idx_bh * M_BLOCKS * CD * D
    zq_offset = idx_bh * M_BLOCKS * CD
    lse_offset = idx_bh * L
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DQ_ptrs = DQ + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DOS_ptrs = DOS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DOL_ptrs = DOL + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DCQ_ptrs = DCQ + cqk_offset + offs_m[:, None] * CD + offs_cd[None, :]
    SQ_ptrs = SQ + sq_offset + idx_m * CD * D + offs_cd[None, :] * D + offs_d[:, None]
    ZQ_ptrs = ZQ + zq_offset + idx_m * CD + offs_cd
    LSE_ptrs = LSE + lse_offset + offs_m
    DENOM_ptrs = DENOM + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    DELTAL_ptrs = DELTAL + lse_offset + offs_m
    LUT_ptr = LUT + lut_offset

    # load Q, DOS, DOL, LSE, DELTA, S: they stay in SRAM throughout the inner loop.
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
    do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < L)
    delta_s = tl.load(DELTAS_ptrs, mask=offs_m < L)
    lse = tl.load(LSE_ptrs, mask=offs_m < L, other=float("inf"))
    
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for block_idx in tl.range(topk, num_stages=2):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N
        
        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(n_mask[None, :], p, 0.0)
        
        # Compute dP and dS.
        dp = tl.dot(do_s, v.T).to(tl.float32)
        ds = p * (dp - delta_s[:, None])
        # Compute dQ.
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < L)
    
    s_q = tl.load(SQ_ptrs)
    z_q = tl.load(ZQ_ptrs)
    denom = tl.load(DENOM_ptrs, mask=offs_m < L, other=float('inf'))
    delta_l = tl.load(DELTAL_ptrs, mask=offs_m < L)
    do_l = tl.load(DOL_ptrs, mask=offs_m[:, None] < L)
    if topk < N_BLOCKS:
        dc_q = (tl.dot(do_l, s_q.to(do_l.dtype)) - delta_l[:, None] * z_q[None, :]) / denom[:, None]
    else:
        dc_q = tl.zeros([BLOCK_M, CD], dtype=tl.float32)

    # Write back dQ and dCQ
    tl.store(DCQ_ptrs, dc_q, mask=offs_m[:, None] < L)


@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, CK, QO, QD, QOSUM, QDSUM,
    DOS, DK, DV, DCK,
    qk_scale, KBID, LSE, DELTAS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    CD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SLICE_FACTOR: tl.constexpr,
):
    BLOCK_M2: tl.constexpr = BLOCK_M // BLOCK_SLICE_FACTOR

    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M2)
    offs_d = tl.arange(0, D)
    offs_cd = tl.arange(0, CD)

    qkv_offset = idx_bh * L * D
    cqk_offset = idx_bh * L * CD
    kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS
    sq_offset = idx_bh * M_BLOCKS * CD * D
    zq_offset = idx_bh * M_BLOCKS * CD
    qosum_offset = idx_bh * CD * D
    qdsum_offset = idx_bh * CD
    lse_offset = idx_bh * L

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    CK_ptrs = CK + cqk_offset + offs_n[:, None] * CD + offs_cd[None, :]
    DOS_ptrs = DOS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DK_ptrs = DK + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DV_ptrs = DV + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DCK_ptrs = DCK + cqk_offset + offs_n[:, None] * CD + offs_cd[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    QOSUM_ptrs = QOSUM + qosum_offset + offs_cd[:, None] * D + offs_d[None, :]
    QDSUM_ptrs = QDSUM + qdsum_offset + offs_cd
    QO_ptrs = QO + sq_offset + offs_cd[:, None] * D + offs_d[None, :]
    QD_ptrs = QD + zq_offset + offs_cd
    KBID_ptr = KBID + kbid_offset + idx_n

    # load K, V and CK: they stay in SRAM throughout the inner loop.
    k = tl.load(K_ptrs, mask=offs_n[:, None] < L)
    v = tl.load(V_ptrs, mask=offs_n[:, None] < L)
    c_k = tl.load(CK_ptrs, mask=offs_n[:, None] < L)
        
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dkv = tl.load(QOSUM_ptrs)
    dkd = tl.load(QDSUM_ptrs)
    for idx_m in tl.range(0, L, BLOCK_M2):
        kbid = tl.load(KBID_ptr)
        if kbid == 1:
            m_mask = offs_m < L - idx_m
            q = tl.load(Q_ptrs, mask=m_mask[:, None])
            lse = tl.load(LSE_ptrs, mask=m_mask, other=float("inf"))
            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
            pT = tl.math.exp2(qkT - lse[None, :])
            pT = tl.where(offs_n[:, None] < L, pT, 0.0)

            do = tl.load(DOS_ptrs, mask=m_mask[:, None])
            # Compute dV.
            dv += tl.dot(pT.to(do.dtype), do)
            delta = tl.load(DELTAS_ptrs, mask=m_mask)
            # Compute dP and dS.
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)
            
            if idx_m % BLOCK_M == 0:
                dkv -= tl.load(QO_ptrs)
                dkd -= tl.load(QD_ptrs)
        
        # Increment pointers
        Q_ptrs += BLOCK_M2 * D
        DOS_ptrs += BLOCK_M2 * D
        LSE_ptrs += BLOCK_M2
        DELTAS_ptrs += BLOCK_M2
        if (idx_m + BLOCK_M2) % BLOCK_M == 0:
            QO_ptrs += CD * D
            QD_ptrs += CD
            KBID_ptr += N_BLOCKS

    dc_k = tl.dot(v, tl.trans(dkv).to(v.dtype)) - dkd[None, :]
    dv += tl.dot(c_k, dkv.to(c_k.dtype))

    # Write back dK, dV and dCK
    tl.store(DK_ptrs, dk * qk_scale, mask=offs_n[:, None] < L)
    tl.store(DV_ptrs, dv, mask=offs_n[:, None] < L)
    tl.store(DCK_ptrs, dc_k, mask=offs_n[:, None] < L)
    

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, c_q, c_k, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert c_q.is_contiguous() and c_k.is_contiguous()
        assert k_block_id.is_contiguous() and lut.is_contiguous()

        # We recommend the following two settings
        assert BLOCK_M == 64 or BLOCK_M == 128
        assert BLOCK_N == 64

        B, H, L, D = q.shape
        CD = c_q.shape[-1]
        if qk_scale is None:
            qk_scale = D**-0.5

        M_BLOCKS = triton.cdiv(L, BLOCK_M)
        N_BLOCKS = triton.cdiv(L, BLOCK_N)

        o_s = torch.empty_like(v)
        o_l = torch.empty_like(v)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        s = torch.empty((B, H, N_BLOCKS, CD, D), device=q.device, dtype=q.dtype)
        z = torch.empty((B, H, N_BLOCKS, CD), device=q.device, dtype=q.dtype)
        s_q = torch.empty((B, H, M_BLOCKS, CD, D), device=q.device, dtype=q.dtype)
        z_q = torch.empty((B, H, M_BLOCKS, CD), device=q.device, dtype=q.dtype)
        denom = torch.empty_like(lse)

        grid = (N_BLOCKS, B * H)
        _attn_fwd_preprocess[grid](
            c_k, v, s, z,
            L, N_BLOCKS,
            D, CD, BLOCK_N
        )
        s_sum = torch.sum(s, axis=2, dtype=torch.float32)
        z_sum = torch.sum(z, axis=2, dtype=torch.float32)

        grid = (M_BLOCKS, B * H)
        _attn_fwd[grid](
            q, k, v, c_q, qk_scale, topk,
            lut, lse, s, z, s_sum, z_sum,
            s_q, z_q, denom, o_s, o_l,
            L, M_BLOCKS, N_BLOCKS,
            D, CD, BLOCK_M, BLOCK_N,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=3
        )
        
        ctx.save_for_backward(q, k, v, c_q, c_k, k_block_id, lut, lse, s_q, z_q, denom, o_s, o_l)
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        return o_s, o_l

    @staticmethod
    def backward(ctx, do_s, do_l):
        q, k, v, c_q, c_k, k_block_id, lut, lse, s_q, z_q, denom, o_s, o_l = ctx.saved_tensors
        do_s = do_s.contiguous()
        do_l = do_l.contiguous()

        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        B, H, L, D = q.shape
        CD = c_q.shape[-1]

        M_BLOCKS = triton.cdiv(L, BLOCK_M)
        N_BLOCKS = triton.cdiv(L, BLOCK_N)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dc_q = torch.empty_like(c_q)
        dc_k = torch.empty_like(c_k)
        qo = torch.empty((B, H, M_BLOCKS, CD, D), device=q.device, dtype=q.dtype)
        qd = torch.empty((B, H, M_BLOCKS, CD), device=q.device, dtype=q.dtype)
        delta_s = torch.empty_like(lse)
        delta_l = torch.empty_like(lse)

        grid = (M_BLOCKS, B * H)
        _attn_bwd_preprocess[grid](
            c_q, o_s, o_l, do_s, do_l,
            qo, qd, denom, delta_s, delta_l,
            L, M_BLOCKS,
            D, CD, BLOCK_M,
        )
        qo_sum = torch.sum(qo, axis=2, dtype=torch.float32).contiguous()
        qd_sum = torch.sum(qd, axis=2, dtype=torch.float32).contiguous()

        grid = (M_BLOCKS, B * H)
        _attn_bwd_dq[grid](
            q, k, v, s_q, z_q, lse, delta_s, delta_l,
            denom, do_s, do_l, dq, dc_q, lut,
            ctx.qk_scale, ctx.topk,
            L, M_BLOCKS, N_BLOCKS,
            D, CD, BLOCK_M, BLOCK_N,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=4 if q.shape[-1] == 64 else 5
        )

        grid = (N_BLOCKS, B * H)
        _attn_bwd_dkdv[grid](
            q, k, v, c_k, qo, qd, qo_sum, qd_sum,
            do_s, dk, dv, dc_k,
            ctx.qk_scale, k_block_id, lse, delta_s,
            L, M_BLOCKS, N_BLOCKS,
            D, CD, BLOCK_M, BLOCK_N,
            BLOCK_SLICE_FACTOR=BLOCK_M // 64,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=4 if q.shape[-1] == 64 else 5
        )

        return dq, dk, dv, dc_q, dc_k, None, None, None, None, None, None
