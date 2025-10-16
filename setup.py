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

from setuptools import setup, find_packages

setup(
    name='sparse_linear_attention',
    version='0.1.0',
    description='Sparse Linear Attention',
    author='Jintao Zhang, Haoxu Wang',
    author_email='jtzhang6@gmail.com',
    url='https://github.com/thu-ml/SLA',
    packages=find_packages(),
    python_requires='>=3.12',
    install_requires=[
        'torch>=2.6.0',
        'triton>=3.4.0',
        'flash-linear-attention',
    ],
    extras_require={
        'benchmark': ['flash-attn']
    }
)
