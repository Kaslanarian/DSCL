import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from .module import GCN, MLP


class DOMINANT(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_layers: int,
        act,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.act = act

        self.enc = GCN(n_layers, n_input, n_hidden, n_hidden, act)
        self.attr_dec = GCN(1, n_hidden, ..., n_input, act)

    def forward(self, x, edge_index):
        z = self.enc(x, edge_index)
        return z @ z.T, self.attr_dec(z, edge_index)


class AnomalyDAE(nn.Module):
    def __init__(
        self,
        node_num: int,
        n_dim: int,
        n_hidden: int,
        embed_dim: int,
        act,
    ) -> None:
        super().__init__()
        self.node_num = node_num
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.embed_dim = embed_dim
        self.act = act

        self.stru_enc_fc = MLP(1, n_dim, None, n_hidden, act)
        self.stru_enc_gat = GCN(
            1,
            n_hidden,
            None,
            embed_dim,
            act,
            conv_layer=GATConv,
        )
        self.attr_enc = MLP(2, node_num, n_hidden, embed_dim, act, False)

    def forward(self, x, edge_index):
        Zv = self.stru_enc_gat(self.stru_enc_fc(x), edge_index)
        Za = self.attr_enc(x.T)
        return torch.mm(Zv, Zv.T), torch.mm(Zv, Za.T)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, seq_len):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_len = seq_len

        self.lstm_enc = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                dropout=dropout,
                                batch_first=True)

    def forward(self, x):
        out, (last_h_state, last_c_state) = self.lstm_enc(x)
        x_enc = last_h_state.squeeze(dim=0)
        x_enc = x_enc.unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_enc, out


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, seq_len, use_act):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_len = seq_len
        self.use_act = use_act  # Parameter to control the last sigmoid activation - depends on the normalization used.
        self.act = nn.Sigmoid()

        self.lstm_dec = nn.LSTM(input_size=hidden_size,
                                hidden_size=hidden_size,
                                dropout=dropout,
                                batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, z):
        # z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, (hidden_state, cell_state) = self.lstm_dec(z)
        dec_out = self.fc(dec_out)
        if self.use_act:
            dec_out = self.act(dec_out)

        return dec_out, hidden_state


# LSTM Auto-Encoder Class
class LSTMAE(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 dropout_ratio,
                 seq_len,
                 use_act=True):
        super(LSTMAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.seq_len = seq_len

        self.encoder = Encoder(input_size=input_size,
                               hidden_size=hidden_size,
                               dropout=dropout_ratio,
                               seq_len=seq_len)
        self.decoder = Decoder(input_size=input_size,
                               hidden_size=hidden_size,
                               dropout=dropout_ratio,
                               seq_len=seq_len,
                               use_act=use_act)

    def forward(self, x, return_last_h=False, return_enc_out=False):
        x_enc, enc_out = self.encoder(x)
        x_dec, last_h = self.decoder(x_enc)

        if return_last_h:
            return x_dec, last_h
        elif return_enc_out:
            return x_dec, enc_out
        return x_dec
