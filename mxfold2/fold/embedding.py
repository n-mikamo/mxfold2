from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.core.fromnumeric import reshape, transpose
import torch
from torch._C import ParameterDict
import torch.nn as nn
import torch.nn.functional as F

#seq = ['AUGUCUAGUCUAGUCUG']

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
        #print(seq2)
        seq3 = seq2.to(self.embedding.weight.device)
        return self.embedding(seq3).transpose(1, 2)

#S = SparseEmbedding(64)
#print('Sparse = ', S.forward(seq).shape)


A_fp_array = np.loadtxt('A_np_txt')
A_fp = torch.from_numpy(A_fp_array).to(torch.float)
C_fp_array = np.loadtxt('C_np_txt')
C_fp = torch.from_numpy(C_fp_array).to(torch.float)
G_fp_array = np.loadtxt('G_np_txt')
G_fp = torch.from_numpy(G_fp_array).to(torch.float)
U_fp_array = np.loadtxt('U_np_txt')
U_fp = torch.from_numpy(U_fp_array).to(torch.float)
I_fp_array = np.loadtxt('I_np_txt')
I_fp = torch.from_numpy(I_fp_array).to(torch.float)
P_fp_array = np.loadtxt('P_np_txt')
P_fp = torch.from_numpy(P_fp_array).to(torch.float)
M_fp_array = np.loadtxt('M_np_txt')
M_fp = torch.from_numpy(M_fp_array).to(torch.float)

class FingerprintEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super(FingerprintEmbedding, self).__init__()
        self.n_out = dim
        self.linear = nn.Linear(1024, dim)
        #self.ecfp = nn.ParameterDict({'0': torch.zeros(1024), 'a': A_fp, 'c': C_fp, 'g': G_fp, 't': U_fp, 'u': U_fp})
        self.A_fp = nn.Parameter(A_fp)
        self.C_fp = nn.Parameter(C_fp)
        self.G_fp = nn.Parameter(G_fp)
        self.U_fp = nn.Parameter(U_fp)
        self.I_fp = nn.Parameter(I_fp)
        self.P_fp = nn.Parameter(P_fp)
        self.M_fp = nn.Parameter(M_fp)
        self.Z_fp = nn.Parameter(torch.zeros(1024))
        self.N_fp = nn.Parameter(torch.zeros(1024))
        self.ecfp = defaultdict(lambda: self.O_fp,
            {'0': self.Z_fp, 'a': self.A_fp, 'c': self.C_fp, 'g': self.G_fp, 't': self.U_fp, 'u': self.U_fp, 'i': self.I_fp, 'p': self.P_fp, 'm': self.M_fp, 'n': self.N_fp})

    
    def forward(self, seq: str) -> torch.Tensor:
        #seq2 = [[self.linear(self.ecfp[c].to(device=self.linear.weight.device)) for c in s.lower()] for s in seq] 
        seq2 = [[self.linear(self.ecfp[c]) for c in s.lower()] for s in seq] 
        seq2 = [torch.vstack(s).transpose(0, 1) for s in seq2]
        seq2 = torch.vstack(seq2)
        return seq2.reshape(-1, seq2.shape[0], seq2.shape[1])

#FE = FingerprintEmbedding(64)
#print('FingerprintEmbedding = ', FE.forward(seq).shape)
#print ('mxfold2')