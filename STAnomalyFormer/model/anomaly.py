import torch
from torch import nn

from .embed import TemporalEmbedding
from .module import (
    SingleGCN,
    MultipleGCN,
    MultiheadAttention,
)
from .revin import RevIN


class STAnomalyFormer_v1(nn.Module):

    def __init__(
        self,
        dist_mat,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        n_gcn: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.embed = TemporalEmbedding(d_in, d_model)
        self.temporal_attn = MultiheadAttention(
            d_model,
            dim_k,
            dim_v,
            n_heads,
            batch_size=batch_size,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        self.spatial_attn = MultiheadAttention(d_model, dim_k, dim_v, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.gcn = SingleGCN(
            d_model,
            d_model,
            dist_mat=dist_mat,
            n_layers=n_gcn,
        )

        self.proj1 = nn.Linear(d_model, d_in)
        self.proj2 = nn.Linear(d_model, d_in)

    def forward(self, x):
        # temporal part
        x = self.embed(x)
        # x = self.auto_corr(x, x)[0]
        x = self.norm1(self.dropout(x + self.temporal_attn(x, x)[0]))

        x = x.swapaxes(0, 1)
        # dynamic spatial part
        output_dy, score_dy = self.spatial_attn(x, x)
        output_dy = self.norm2(self.dropout(x + output_dy))

        # static spatial part
        output_st, score_st = self.gcn(x)

        # fusion
        output_dy, output_st = self.proj1(output_dy), self.proj2(output_st)
        g = torch.sigmoid(output_dy + output_st)
        output = g * output_dy + (1 - g) * output_st
        # output = output_dy + output_st
        output = output.swapaxes(0, 1)
        score_dy = score_dy.mean(0)

        return output, score_dy, score_st


class STAnomalyFormer_v2(STAnomalyFormer_v1):

    def __init__(
        self,
        dist_mat,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        n_gcn: int,
        batch_size: int,
    ) -> None:
        super().__init__(dist_mat, d_in, d_model, dim_k, dim_v, n_heads, n_gcn,
                         batch_size)
        self.embed_gcn = SingleGCN(
            d_in,
            d_model,
            dist_mat,
            n_layers=1,
            activation=lambda x: x,
        )
        self.revin = RevIN(d_in)

    def forward(self, x):
        x = self.revin(x, 'norm')
        x1 = self.embed(x)
        branch1 = x1
        branch2 = x1 + self.embed_gcn(x.swapaxes(0, 1))[0].swapaxes(0, 1)

        branch1_, attn1 = self.temporal_attn(branch1, branch1)
        branch2_, attn2 = self.temporal_attn(branch2, branch2)

        branch1_ = self.norm1(self.dropout(branch1 + branch1_))
        branch2_ = self.norm1(self.dropout(branch2 + branch2_))

        branch1_ = branch1_.swapaxes(0, 1)
        branch2_ = branch2_.swapaxes(0, 1)

        output_dy, score_dy = self.spatial_attn(branch1_, branch1_)
        output_dy = self.norm2(self.dropout(branch1_ + output_dy))

        output_st, score_st = self.gcn(branch2_)

        output_dy, output_st = self.proj1(output_dy), self.proj2(output_st)
        g = torch.sigmoid(output_dy + output_st)
        output = g * output_dy + (1 - g) * output_st
        output = output.swapaxes(0, 1)
        score_dy = score_dy.mean(0)

        return (
            self.revin(output, 'denorm'),
            # output,
            attn1.mean(0),
            attn2.mean(0),
            score_dy,
            score_st,
        )


from .patch import Patch, PatchEncoder, RandomMasking
from .tsfm import TemporalTransformer


class STPatchFormer(nn.Module):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        dist_mat,
        n_gcn: int = 3,
        temporal_half: bool = False,
        spatial_half: bool = False,
        static_only: bool = False,
        dynamic_only: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.temporal_half = temporal_half
        self.spatial_half = spatial_half
        self.n_gcn = n_gcn
        self.static_only = static_only
        self.dynamic_only = dynamic_only

        self.revin = RevIN(d_in)
        self.patch = Patch(seq_len, patch_len, stride)
        self.patch_tsfm = PatchEncoder(
            d_in,
            self.patch.num_patch,
            patch_len,
            d_model,
            n_heads,
            256,
            False,
            0.1,
            0.1,
            temporal_half,
        )
        self.spatial_tsfm = TemporalTransformer(
            d_model,
            d_model // n_heads,
            d_model // n_heads,
            n_heads,
            d_model,
            0.1,
            spatial_half,
            True,
        )
        self.da_gcn = SingleGCN(
            d_model,
            d_model,
            dist_mat,
            n_layers=n_gcn,
        )
        self.proj_dy = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )
        self.proj_st = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )

    def forward(self, x, return_recon=True):
        # x : (N, T, d)
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        z = self.patch_tsfm(x)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        dy_z, attn = self.spatial_tsfm(z)  # (VAR * NP, N, D)
        st_z, graph = self.da_gcn(z)  # (VAR * NP, N, D)
        if not return_recon:
            return attn, graph

        if self.dynamic_only:
            z = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
        elif self.static_only:
            z = self.proj_st(st_z)
        else:
            dy_out = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
            st_out = self.proj_st(st_z)  # (VAR * NP, N, PL)
            g = torch.sigmoid(dy_out + st_out)
            z = g * dy_out + (1 - g) * st_out
        z = z.reshape(
            self.d_in,
            -1,
            x.shape[0],
            self.patch_len,  # (VAR, NP, N, PL)
        ).permute(2, 1, 3, 0)  # (N, NP, PL, VAR)
        z = self.revin(z, 'denorm')
        return (patch_x.transpose(2, 3), z), (attn, graph)


class STPatchMaskFormer(STPatchFormer):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        dist_mat,
        n_gcn: int = 3,
        temporal_half: bool = False,
        spatial_half: bool = False,
        static_only: bool = False,
        dynamic_only: bool = False,
        mask_ratio: float = 0.4,
    ) -> None:
        super().__init__(seq_len, patch_len, stride, d_in, d_model, n_heads,
                         dist_mat, n_gcn, temporal_half, spatial_half,
                         static_only, dynamic_only)
        self.mask_ratio = mask_ratio
        self.random_mask = RandomMasking(self.mask_ratio)

    def forward(self, x):
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        mask_x = self.random_mask(x)
        z = self.patch_tsfm(mask_x)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        dy_z, attn = self.spatial_tsfm(z)  # (VAR * NP, N, D)
        st_z, graph = self.da_gcn(z)  # (VAR * NP, N, D)

        dy_out = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
        st_out = self.proj_st(st_z)  # (VAR * NP, N, PL)
        g = torch.sigmoid(dy_out + st_out)
        z = g * dy_out + (1 - g) * st_out
        z = z.reshape(
            self.d_in,
            -1,
            x.shape[0],
            self.patch_len,  # (VAR, NP, N, PL)
        ).permute(2, 1, 3, 0)  # (N, NP, PL, VAR)
        z = self.revin(z, 'denorm')
        return (patch_x.transpose(2, 3), z), (attn, graph)


class STPatch_MGCNFormer(nn.Module):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        dist_mats,
        n_gcn: int = 3,
        temporal_half: bool = False,
        spatial_half: bool = False,
        static_only: bool = False,
        dynamic_only: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.temporal_half = temporal_half
        self.spatial_half = spatial_half
        self.n_gcn = n_gcn
        self.static_only = static_only
        self.dynamic_only = dynamic_only

        self.revin = RevIN(d_in)
        self.patch = Patch(seq_len, patch_len, stride)
        self.patch_tsfm = PatchEncoder(
            d_in,
            self.patch.num_patch,
            patch_len,
            d_model,
            n_heads,
            256,
            False,
            0.1,
            0.1,
            temporal_half,
        )
        self.spatial_tsfm = TemporalTransformer(
            d_model,
            d_model // n_heads,
            d_model // n_heads,
            n_heads,
            d_model,
            0.1,
            spatial_half,
            True,
        )
        self.da_gcn = MultipleGCN(
            d_model,
            d_model,
            dist_mats,
            n_layers=n_gcn,
            bias=True,
        )
        self.proj_dy = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )
        self.proj_st = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )

    def forward(self, x, return_recon=True):
        # x : (N, T, d)
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        z = self.patch_tsfm(x)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        dy_z, attn = self.spatial_tsfm(z)  # (VAR * NP, N, D)
        st_z, graph = self.da_gcn(z)  # (VAR * NP, N, D)
        if not return_recon:
            return attn, graph

        if self.dynamic_only:
            z = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
        elif self.static_only:
            z = self.proj_st(st_z)
        else:
            dy_out = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
            st_out = self.proj_st(st_z)  # (VAR * NP, N, PL)
            g = torch.sigmoid(dy_out + st_out)
            z = g * dy_out + (1 - g) * st_out
        z = z.reshape(
            self.d_in,
            -1,
            x.shape[0],
            self.patch_len,  # (VAR, NP, N, PL)
        ).permute(2, 1, 3, 0)  # (N, NP, PL, VAR)
        z = self.revin(z, 'denorm')
        return (patch_x.transpose(2, 3), z), (attn, graph)
