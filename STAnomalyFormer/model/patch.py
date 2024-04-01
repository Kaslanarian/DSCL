import torch
from torch import nn
import math

from .tsfm import TemporalTransformer


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(
        bs,
        L,
        nvars,
        device=xb.device,
    )  # noise in [0, 1], bs x L x nvars

    # sort noise for each sample
    ids_shuffle = torch.argsort(
        noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle,
                                dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :
                           len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D),
    )  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(
        bs,
        L - len_keep,
        nvars,
        D,
        device=xb.device,
    )  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat(
        [x_kept, x_removed],
        dim=1,
    )  # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(
        x_,
        dim=1,
        index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D),
    )  # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones(
        [bs, L, nvars],
        device=x.device,
    )  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(
        mask,
        dim=1,
        index=ids_restore,
    )  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty(
            (q_len, d_model)
        )  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class Patch(nn.Module):

    def __init__(self, seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        tgt_len = patch_len + stride * (self.num_patch - 1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride,
        )  # xb: [bs x num_patch x n_vars x patch_len]
        return x


class PatchEncoder(nn.Module):

    def __init__(
        self,
        c_in,
        num_patch,
        patch_len,
        d_model=128,
        n_heads=16,
        d_ff=256,
        shared_embedding=True,
        attn_dropout=0.,
        dropout=0.,
        half: bool = False,
        pe='zeros',
        learn_pe=True,
    ):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.n_heads = n_heads

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars):
                self.W_P.append(nn.Linear(
                    patch_len,
                    d_model,
                ))
        else:
            self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(
            pe,
            learn_pe,
            num_patch,
            d_model,
        )

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.encoder = TemporalTransformer(
            d_model=d_model,
            dim_k=d_model // n_heads,
            dim_v=d_model // n_heads,
            n_heads=n_heads,
            dim_fc=d_ff,
            dropout=attn_dropout,
            half=half,
            return_attn=False,
        )

    def forward(self, x):
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:, :, i, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)  # x: [bs x num_patch x nvars x d_model]
        x = x.transpose(1, 2)
        # x: [bs x nvars x num_patch x d_model]
        u = torch.reshape(
            x, (bs * n_vars, num_patch,
                self.d_model))  # u: [bs * nvars x num_patch x d_model]
        u = self.dropout(u +
                         self.W_pos)  # u: [bs * nvars x num_patch x d_model]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x num_patch x d_model]
        return z.reshape((-1, n_vars, num_patch, self.d_model))
        # z: [bs x nvars x d_model x num_patch]


class RandomMasking(nn.Module):

    def __init__(self, mask_ratio: float = 0.4) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, xb):
        if self.training:
            x_masked, _, mask, _ = random_masking(xb, self.mask_ratio)
            self.mask = mask.bool()
        else:
            x_masked = xb
        return x_masked
