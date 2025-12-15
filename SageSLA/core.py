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
import torch.nn as nn
import torch.nn.functional as F

import spas_sage_attn._qattn as qattn
import spas_sage_attn._fused as fused
from spas_sage_attn.utils import get_vanilla_qk_quant, block_map_lut_triton

from .utils import get_block_map


class SageSparseLinearAttention(nn.Module):
    def __init__(self, head_dim, topk, feature_map='softmax', BLKQ=64, BLKK=64, use_bf16=True, tie_feature_map_qk=True, timestep_adaptive_topk=True):
        R'''
        Args:
            head_dim: dimension of each head.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            BLKQ: block size for query.
            BLKK: block size for key.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
            timestep_adaptive_topk: whether to adaptively adjust topk during diffusion.
        '''
        assert BLKQ == 128 and BLKK == 64, "For SageSLA, BLKQ must be 128 and BLKK must be 64."

        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.timestep_adaptive_topk = timestep_adaptive_topk
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)
        
    def forward(self, q, k, v, return_sparsity=False, timestep=None, total_timesteps=None):
        R'''
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
            return_sparsity: whether to return the actual sparsity.
            timestep: current timestep for diffusion models.
            total_timesteps: total timesteps for diffusion models.
        '''
        
        dtype = q.dtype
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        current_topk = self.topk
        if self.timestep_adaptive_topk and self.topk is not None and timestep is not None and total_timesteps is not None:
            early_timestep_threshold = 0.3 * total_timesteps
            if timestep < early_timestep_threshold:
                current_topk = self.topk * 2.0
        
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=current_topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)
        c_q = self.feature_map_q(q).contiguous().to(self.dtype)
        c_k = self.feature_map_k(k).contiguous().to(self.dtype)

        ########## SPARGE BEGIN ##########

        km = k.mean(dim=-2, keepdim=True)
        headdim = q.size(-1)
        
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km)
        lut, valid_block_num = block_map_lut_triton(sparse_map)
        scale = 1.0 / (headdim ** 0.5)

        assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

        ## quant v
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

        o_s = torch.empty_like(q)
        pvthreshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
        qattn.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            q_int8, k_int8, v_fp8, o_s, lut, valid_block_num, pvthreshold, q_scale, k_scale, v_scale, 1, False, 1, scale, 0
        )

        ########## SPARGE END ##########

        def calc_linear(q, k, v):
            kvsum = k.transpose(-1, -2) @ v
            ksum = torch.sum(k, dim=-2, keepdim=True)
            return (q @ kvsum) / (q * ksum).sum(dim=-1, keepdim=True)
        o_l = calc_linear(c_q, c_k, v)

        with torch.amp.autocast('cuda', dtype=self.dtype):
            o_proj = self.proj_l(o_l)
        o = (o_s + o_proj).to(dtype)

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        else:
            return o