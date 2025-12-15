# SageSLA

SageSLA is a fast quantized implementation of the forward pass of SLA. To use SageSLA, install [SpargeAttn](https://github.com/thu-ml/SpargeAttn) first:
```bash
pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation
```

Then use it the same way as SLA:
```python
import torch
from SageSLA import SageSparseLinearAttention

attn = SageSparseLinearAttention(
    head_dim=128,
    topk=0.2,                 # = 1 - sparsity
    feature_map="softmax",    # options: elu, relu, softmax
    BLKQ=128,                 # SageSLA only supports BLKQ=128 and BLKK=64
    BLKK=64,
).cuda()

B, H, L, D = 2, 4, 4096, 128
q = torch.randn((B, H, L, D), dtype=torch.bfloat16, device='cuda')
k = torch.randn((B, H, L, D), dtype=torch.bfloat16, device='cuda')
v = torch.randn((B, H, L, D), dtype=torch.bfloat16, device='cuda')

o = attn(q, k, v)
```