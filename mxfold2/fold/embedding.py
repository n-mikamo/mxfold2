from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.core.fromnumeric import reshape, transpose
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms.functional as TF

seq = ['AUGUCUAGUCUAGUCUG']

class OneHotEmbedding(nn.Module):
    def __init__(self, ksize: int = 0) -> None: 
        super(OneHotEmbedding, self).__init__()
        self.n_out = 4
        self.ksize = ksize
        eye = np.identity(4, dtype=np.float32)
        zero = np.zeros(4, dtype=np.float32)
        self.onehot: defaultdict[str, np.ndarray] = defaultdict(
            lambda: np.ones(4, dtype=np.float32)/4, 
            {'a': eye[0], 'c': eye[1], 'g': eye[2], 't': eye[3], 'u': eye[3], '0': zero} )

    def encode(self, seq) -> np.ndarray:
        seq = [ self.onehot[s] for s in seq.lower() ]
        seq = np.vstack(seq)
        return seq.transpose()

    def pad_all(self, seq, pad_size: int) -> list[str]:
        pad = 'n' * pad_size
        seq = [ pad + s + pad for s in seq ]
        l = max([len(s) for s in seq])
        seq = [ s + '0' * (l-len(s)) for s in seq ]
        return seq

    def forward(self, seq: str) -> torch.Tensor:
        seq2 = self.pad_all(seq, self.ksize//2)
        seq3 = [ self.encode(s) for s in seq2 ]
        return torch.from_numpy(np.stack(seq3)) # pylint: disable=no-member

class SparseEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super(SparseEmbedding, self).__init__()
        self.n_out = dim
        #以下の6は系列長(vocb), dimは何次元のベクトルにするか(今回は64としている)
        #padding_idx=0は0番目の要素を無視するという意味
        self.embedding = nn.Embedding(6, dim, padding_idx=0)
        self.vocb = defaultdict(lambda: 5,
            {'0': 0, 'a': 1, 'c': 2, 'g': 3, 't': 4, 'u': 4})

    def forward(self, seq: str) -> torch.Tensor:
        seq2 = torch.LongTensor([[self.vocb[c] for c in s.lower()] for s in seq])
        seq3 = seq2.to(self.embedding.weight.device)
        return self.embedding(seq3).transpose(1, 2)

S = SparseEmbedding(1)
print('Sparse = ', S.forward(seq).shape)




A_fp_array = np.loadtxt('A_np_txt')
A_fp = torch.from_numpy(A_fp_array)
C_fp_array = np.loadtxt('C_np_txt')
C_fp = torch.from_numpy(C_fp_array)
G_fp_array = np.loadtxt('G_np_txt')
G_fp = torch.from_numpy(G_fp_array)
U_fp_array = np.loadtxt('U_np_txt')
U_fp = torch.from_numpy(U_fp_array)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, seq):
        super().__init__()  # 基底クラスの初期化
        self.in_features = in_features
        self.out_features = out_features
        self.seq = seq

    def forward(self, seq):
        seq
        pass

class Fingerprint(nn.Module):
    def __init__(self, dim: int) -> None:
        super(Fingerprint, self).__init__()
        self.n_out = dim
        self.embedding = nn.Embedding(6, dim, padding_idx=0)
        self.ecpf = defaultdict(lambda: 5,
            {'0': 0, 'a': A_fp, 'c': C_fp, 'g': G_fp, 't': U_fp, 'u': U_fp})

    def encode(self, seq) -> np.ndarray:
        seq2 = [[self.ecpf[c] for c in s.lower()] for s in seq]
        seq3 = np.vstack(seq2)
        return seq3

    def forward(self, seq: str) -> torch.Tensor:
        seq = self.encode(seq)
        L = Linear(in_features=1024, out_features=64, seq=seq)
        seq5 = L.seq
        return seq5#, transpose(1, 2)

F = Fingerprint(1)
print('Finger = ', F.forward(seq).shape)
