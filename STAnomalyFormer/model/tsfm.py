import torch
from torch import nn
from .attn import MultiheadAttention


class TemporalTransformer(nn.Module):

    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
        dropout: float = 0.1,
        half: bool = False,
        return_attn: bool = False,
    ) -> None:
        super().__init__()
        self.half_ = half
        self.d_model = d_model
        self.attn = MultiheadAttention(d_model, dim_k, dim_v, n_heads)
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=dim_fc,
            kernel_size=1,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if not half:
            self.conv2 = nn.Conv1d(
                in_channels=dim_fc,
                out_channels=d_model,
                kernel_size=1,
            )
            self.norm2 = nn.LayerNorm(d_model)
        self.return_attn = return_attn

    def forward(self, x):
        x_, attn = self.attn(x, x)
        y = x = self.norm1(self.dropout(x + x_))
        y = self.dropout(torch.relu(self.conv1(y.transpose(-1, 1))))
        if self.half_:
            if self.return_attn:
                return y, attn
            return y
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)
        if self.return_attn:
            return output, attn
        return output
