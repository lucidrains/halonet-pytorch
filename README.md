<img src="./halonet.png" width="500px"></img>

## HaloNet - Pytorch

Implementation of the Attention layer from the paper, <a href="https://arxiv.org/abs/2103.12731">Scaling Local Self-Attention For Parameter Efficient Visual Backbones</a>. This repository will only house the attention layer and not much more.


## Install

```bash
$ pip install halonet-pytorch
```

## Usage

```python
import torch
from halonet_pytorch import HaloAttention

attn = HaloAttention(
    dim = 512,
    fmap_size = 32,
    block_size = 8,
    halo_size = 4
).cuda()

fmap = torch.randn(1, 512, 32, 32).cuda()
attn(fmap) # (1, 512, 32, 32)
```

## Citations

```bibtex
@misc{vaswani2021scaling,
    title   = {Scaling Local Self-Attention For Parameter Efficient Visual Backbones}, 
    author  = {Ashish Vaswani and Prajit Ramachandran and Aravind Srinivas and Niki Parmar and Blake Hechtman and Jonathon Shlens},
    year    = {2021},
    eprint  = {2103.12731},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
