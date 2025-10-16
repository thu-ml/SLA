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

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

from sparse_linear_attention.utils import get_block_map
from sparse_linear_attention.kernel import _attention


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, N_HEADS, HEAD_DIM = 2, 16, 128
# vary seq length for fixed head and batch
configs = []
for mode in ["fwd", "bwd"]:  # 
    for causal in [False]:
        if mode == "bwd" and causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[32760],
                line_arg="provider",
                line_vals=["sla"] + (["flash"] if HAS_FLASH else []),
                line_names=["SLA"] + (["Flash Attention"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-"), ("black", "-")],
                ylabel="TFLOPS",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device=DEVICE, sparsity=0.5):
    assert mode in ["fwd", "bwd"]
    dtype = torch.bfloat16
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True).contiguous()
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True).contiguous()
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True).contiguous()
    do = torch.randn_like(v)

    if provider == "flash":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        do = do.transpose(1, 2).contiguous()

        fn = lambda: flash_attn_func(q, k, v)
        if mode == "bwd":
            o = fn()
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    else:
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=1 - sparsity, BLKQ=128, BLKK=64)
            
        c_q = (torch.nn.functional.elu(q) + 1).detach()
        c_k = (torch.nn.functional.elu(k) + 1).detach()
        fn = lambda: _attention.apply(q, k, v, c_q, c_k, sparse_map, lut, real_topk, 128, 64)
        
        if mode == "bwd":
            o_s, o_l = fn()
            o = o_s + o_l
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    
    flops_per_matmul = 2.0 * q.numel() * N_CTX
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    torch.manual_seed(42)
    bench_attention.run(print_data=True, sparsity=0.95)
