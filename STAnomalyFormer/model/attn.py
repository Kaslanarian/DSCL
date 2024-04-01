import torch
from torch import nn
import math


class MultiheadAttention(nn.Module):
    '''For the shape (B, L, D)'''

    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, dim_k * n_heads)
        self.k = nn.Linear(d_model, dim_k * n_heads)
        self.v = nn.Linear(d_model, dim_v * n_heads)
        self.o = nn.Linear(dim_v * n_heads, d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def attention(self, Q, K, V):
        B, L = Q.shape[:2]
        scores = torch.einsum("blhe,bshe->bhls", Q, K) * self.norm_fact
        scores = scores.softmax(dim=-1)
        output = torch.einsum("bhls,bshd->blhd", scores, V).reshape(B, L, -1)
        return output, scores

    def forward(self, x, y):
        '''x : (B, L, D)'''
        B, L, _ = x.shape
        Q = self.q(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        K = self.k(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        V = self.v(y).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        output, scores = self.attention(Q, K, V)
        return self.o(output), scores
