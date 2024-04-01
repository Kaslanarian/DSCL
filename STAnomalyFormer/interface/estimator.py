from pyod.models.base import BaseDetector
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score
from typing import Union, List, Tuple, Literal
from copy import deepcopy
from tqdm import tqdm

from ..model.module import SpatialTSFM, TemporalTSFM
from ..model.baseline import DOMINANT, AnomalyDAE, GCN, LSTMAE
from ..model.anomaly import (
    STAnomalyFormer_v1,
    STAnomalyFormer_v2,
    STPatchFormer,
    STPatchMaskFormer,
    STPatch_MGCNFormer,
)
from .utils import predict_by_score, EarlyStopping


def kl_loss(p, q):
    res = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
    return torch.mean(torch.mean(res, dim=(0, 1)), dim=1)


def sym_kl_loss(p, q):
    return (kl_loss(p, q) + kl_loss(q, p)) / 2


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, x):
        super(SimpleDataset, self).__init__()
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]


class TemporalTSFMDetector(BaseDetector):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_heads: int,
        dim_fc_expand: int = 1,
        n_layers: int = 1,
        device: str = 'cpu',
        epoch: int = 10,
        lr: float = 1e-4,
        batch_size: int = -1,
        contamination=0.1,
        verbose=False,
        **kwargs,
    ):
        super().__init__(contamination)
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_fc_expand = dim_fc_expand
        self.n_layers = n_layers
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.tsfm_args = {
            "d_in": d_in,
            "d_model": d_model,
            "dim_k": d_model // n_heads,
            "dim_v": d_model // n_heads,
            "n_heads": n_heads,
            "dim_fc": n_heads * self.d_model,
            "n_layers": n_layers,
        }

    def fit(self, x, test_x, y=None):
        x_ = torch.FloatTensor(x)
        self.model = TemporalTSFM(**self.tsfm_args).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        batch_size = self.batch_size if self.batch_size != -1 else x.shape[0]
        dataset = SimpleDataset(x_)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.model.train()

        if y is not None:
            max_auc = 0.

        for epoch in range(self.epoch):
            for data in train_loader:
                data = data.to(self.device)
                output = self.model(data)
                score = torch.square(output - data).mean((1, 2))
                loss = score.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    score = self.decision_function(test_x)
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    def presict(self, x):
        score = self.decision_function(x)
        return predict_by_score(score, self.contamination)

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output = self.model(x)
        recon = torch.square(output - x)
        score = recon.mean((1, 2))
        return score.cpu().numpy()


class SpatialTSFMDetector(BaseDetector):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc_expand: int,
        n_layers: int = 1,
        device: str = 'cpu',
        epoch: int = 10,
        lr: float = 1e-4,
        batch_size: int = -1,
        contamination=0.1,
        verbose=False,
    ):
        super().__init__(contamination)
        self.d_in = d_in
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.dim_fc_expand = dim_fc_expand
        self.n_layers = n_layers
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.tsfm_args = {
            "d_in": d_in,
            "d_model": d_model,
            "dim_k": dim_k,
            "dim_v": dim_v,
            "n_heads": n_heads,
            "dim_fc": n_heads * self.d_model,
            "n_layers": n_layers,
        }

    def fit(self, x, y=None):
        x_ = torch.FloatTensor(x)
        self.model = SpatialTSFM(**self.tsfm_args).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        batch_size = self.batch_size if self.batch_size != -1 else x.shape[1]
        dataset = SimpleDataset(x_.swapaxes(0, 1))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.model.train()

        if y is not None:
            max_auc = 0.

        for epoch in range(self.epoch):
            for data in dataloader:
                data = data.to(self.device).swapaxes(0, 1)
                output = self.model(data)
                score = torch.square(output - data).mean((1, 2))
                loss = score.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                with torch.no_grad():
                    cuda_x = x_.to(self.device)
                    output = self.model(cuda_x)
                    score = torch.square(output - cuda_x).mean(
                        (1, 2)).cpu().numpy()
                loss = score.mean()
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss)
                if y is not None:
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    def presict(self, x):
        score = self.decision_function(x)
        return predict_by_score(score, self.contamination)

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output = self.model(x)
        score = torch.square(output - x).mean((1, 2))
        return score.cpu().numpy()


class DOMINANTDetector(BaseDetector):

    def __init__(
        self,
        n_hidden: Union[List[int], Tuple[int], int] = 64,
        n_layers: int = 3,
        act=nn.ReLU,
        alpha: float = 0.5,
        lr: float = 0.005,
        weight_decay: float = 0.,
        device: str = 'cpu',
        epoch: int = 5,
        verbose: bool = False,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination)
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.act = act
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, test_G: Data, y=None):
        G = G.to(self.device)
        test_G = test_G.to(self.device)
        self.model = DOMINANT(
            G.num_node_features,
            self.n_hidden,
            self.n_layers,
            self.act,
        ).to(self.device)
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.model.train()

        if y is not None:
            max_auc = 0.

        for epoch in range(1, self.epoch + 1):
            stru_recon, attr_recon = self.model(test_G.x, test_G.edge_index)

            stru_score = torch.square(stru_recon - A).sum(1)
            attr_score = torch.square(attr_recon - G.x).sum(1)
            score = self.alpha * stru_score + (1 - self.alpha) * attr_score
            loss = score.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    score = self.decision_function(test_G)
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        G = G.to(self.device)
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        stru_recon, attr_recon = self.model(G.x, G.edge_index)
        stru_score = torch.square(stru_recon - A).sum(1).sqrt()
        attr_score = torch.square(attr_recon - G.x).sum(1).sqrt()
        score = self.alpha * stru_score + (1 - self.alpha) * attr_score
        return score.cpu().numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)


class AnomalyDAEDetector(BaseDetector):

    def __init__(
        self,
        embed_dim: int = 8,
        n_hidden: int = 64,
        act=nn.ReLU,
        alpha: float = 0.5,
        theta: float = 1.1,
        eta: float = 1.1,
        lr: float = 0.005,
        weight_decay: float = 0.,
        device: str = 'cpu',
        epoch: int = 100,
        verbose: bool = False,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination)
        self.embed_dim = embed_dim
        self.n_hidden = n_hidden
        self.act = act
        self.alpha = alpha
        self.theta = theta
        self.eta = eta
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, test_G: Data, y=None):
        G = G.to(self.device)
        test_G = test_G.to(self.device)
        self.model = AnomalyDAE(
            G.num_nodes,
            G.num_node_features,
            self.n_hidden,
            self.embed_dim,
            self.act,
        ).to(self.device)

        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        Theta = torch.full_like(A, self.theta).to(self.device)
        Theta[G.edge_index[0], G.edge_index[1]] = 1.
        Eta = torch.ones_like(G.x).to(self.device)
        Eta[G.x != 0] = self.eta

        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.model.train()

        if y is not None:
            max_auc = 0.

        for epoch in range(1, self.epoch + 1):
            stru_recon, attr_recon = self.model(G.x, G.edge_index)
            stru_error = torch.square((stru_recon - A) * Theta)
            attr_error = torch.square((attr_recon - G.x) * Eta)
            stru_score = stru_error.sum(1)
            attr_score = attr_error.sum(1)
            score = self.alpha * stru_score + (1 - self.alpha) * attr_score
            loss = score.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    score = self.decision_function(test_G)
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        G = G.to(self.device)
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        Theta = torch.full_like(A, self.theta).to(self.device)
        Theta[G.edge_index[0], G.edge_index[1]] = 1.
        Eta = torch.ones_like(G.x).to(self.device)
        Eta[G.x != 0] = self.eta

        stru_recon, attr_recon = self.model(G.x, G.edge_index)
        stru_score = torch.square((stru_recon - A) * Theta).sum(1)
        attr_score = torch.square((attr_recon - G.x) * Eta).sum(1)
        score = self.alpha * stru_score + (1 - self.alpha) * attr_score
        return score.cpu().numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)


class OCGNNDetector(BaseDetector):

    def __init__(
        self,
        n_hidden: Union[List[int], Tuple[int], int] = 64,
        n_layers: int = 4,
        act=nn.ReLU,
        beta: float = 0.1,
        phi: int = 10,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = 'cpu',
        epoch: int = 100,
        verbose: bool = False,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.act = act
        self.beta = beta
        self.phi = phi
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, test_G: Data, y=None):
        G = G.to(self.device)
        test_G = test_G.to(self.device)
        self.model = GCN(
            self.n_layers,
            G.num_features,
            self.n_hidden,
            self.n_hidden,
            self.act,
            last_act=False,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        with torch.no_grad():
            r = 0.
            c = self.model(G.x, G.edge_index).mean(0)  # cuda

        self.model.train()

        if y is not None:
            max_auc = 0.

        for epoch in range(1, self.epoch + 1):
            dV = (self.model(G.x, G.edge_index) - c).square().sum(1)
            loss = torch.relu(dV - r**2).mean() / self.beta

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    with torch.no_grad():
                        score = (self.model(test_G.x, test_G.edge_index) -
                                 c).square().sum(1).cpu()
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

            if epoch % self.phi == 0:
                with torch.no_grad():
                    r = torch.quantile(dV, 1 - self.beta).item()
                    c = self.model(G.x, G.edge_index).mean(0)

        with torch.no_grad():
            self.r = torch.quantile(dV, 1 - self.beta).item()
            self.c = self.model(G.x, G.edge_index).mean(0)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        G = G.to(self.device)
        score = (self.model(G.x, G.edge_index) - self.c).square().sum(1)
        return score.cpu().numpy() - self.r**2

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)


class LSTM_AEDetector(BaseDetector):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        dropout: float = 0.,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = 'cpu',
        batch_size: int = -1,
        epoch: int = 100,
        verbose: bool = False,
        contamination: float = 0.1,
    ):
        super().__init__(contamination)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, test_G: Data, y=None):
        x = G.x.reshape(G.x.shape[0], -1, self.input_size).to(self.device)
        test_x = test_G.x.reshape(G.x.shape[0], -1,
                                  self.input_size).to(self.device)
        self.model = LSTMAE(
            self.input_size,
            self.hidden_size,
            self.dropout,
            x.shape[-2],
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if y is not None:
            max_auc = 0.

        batch_size = self.batch_size if self.batch_size != -1 else G.x.shape[0]
        dataset = SimpleDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        for epoch in range(1, self.epoch + 1):
            self.model.train()

            for data in loader:
                output = self.model(data)
                loss = F.mse_loss(output, data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                self.model.eval()
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    with torch.no_grad():
                        output = self.model(test_x)
                        score = torch.square(test_x - output).sum(
                            (-1, -2)).cpu().numpy()
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        x = G.x.reshape(G.x.shape[0], -1, self.input_size).to(self.device)
        return torch.square(x - self.model(x)).sum((-1, -2)).cpu().numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)


class STAnomalyFormerDetector_v1(BaseDetector):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        n_gcn: int,
        alpha: float = 0.,
        device: str = 'cpu',
        epoch: int = 10,
        batch_size: int = -1,
        lr: float = 1e-4,
        contamination=0.1,
        verbose: bool = False,
        log_interval: int = 1,
        **kwargs,
    ):
        super().__init__(contamination)
        self.d_in = d_in
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose
        self.tsfm_args = {
            "d_in": d_in,
            "d_model": d_model,
            "dim_k": dim_k,
            "dim_v": dim_v,
            "n_heads": n_heads,
            "batch_size": batch_size,
        }
        self.alpha = alpha
        self.n_gcn = n_gcn
        self.log_interval = log_interval

    def fit(self, x, mat, y=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        self.model = STAnomalyFormer_v1(
            mat.to(self.device),
            n_gcn=self.n_gcn,
            **self.tsfm_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if y is not None:
            max_auc = 0.

        process = range(self.epoch) if self.verbose else tqdm(range(
            self.epoch))
        for epoch in process:
            self.model.train()
            output, score_dy, score_st = self.model(x_)

            recon = torch.square(output - x_).mean((1, 2))
            discrepancy = sym_kl_loss(score_dy, score_st)
            score = recon + self.alpha * discrepancy

            loss = score.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                output, score_dy, score_st = self.model(x_)

                recon = torch.square(output - x_).mean((1, 2))
                discrepancy = sym_kl_loss(score_dy, score_st)
                score = recon + self.alpha * discrepancy
                loss = score.mean()

            log = "Epoch {:3d}, loss={:5.6f}".format(
                epoch,
                loss.item(),
            )
            if y is not None:
                auc = roc_auc_score(y, score.cpu().numpy())
                log += ", AUC={:6f}".format(auc)
                if auc >= max_auc:
                    max_auc = auc
                    self.model_copy = deepcopy(self.model)

            if self.verbose:
                if (epoch + 1) % self.log_interval == 0:
                    print(log)
            elif y is not None:
                process.set_postfix(
                    max_auc="max: {:.4f}, current: {:.4f}".format(
                        max_auc, auc),
                    refresh=True,
                )

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )

        return self

    def presict(self, x):
        score = self.decision_function(x)
        return predict_by_score(score, self.contamination)

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output, score_dy, score_st = self.model(x)
        recon = torch.square(output - x).mean((1, 2))
        discrepancy = sym_kl_loss(score_st, score_dy)
        score = recon + self.alpha * discrepancy
        return score.cpu().numpy()


class STAnomalyFormerDetector_v2(STAnomalyFormerDetector_v1):

    def fit(self, x, mat, y=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        self.model = STAnomalyFormer_v1(
            mat.to(self.device),
            n_gcn=self.n_gcn,
            **self.tsfm_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if y is not None:
            max_auc = 0.

        process = range(self.epoch) if self.verbose else tqdm(range(
            self.epoch))
        for epoch in process:
            self.model.train()
            output, score_dy, score_st = self.model(x_)

            recon = torch.square(output - x_).mean((1, 2))
            discrepancy = sym_kl_loss(
                score_dy,
                score_st.detach(),
            ) - sym_kl_loss(
                score_dy.detach(),
                score_st,
            )
            score = recon + self.alpha * discrepancy

            loss = score.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                output, score_dy, score_st = self.model(x_)

                recon = torch.square(output - x_).mean((1, 2))
                discrepancy = sym_kl_loss(score_dy, score_st)
                score = recon + self.alpha * discrepancy
                loss = score.mean()

            log = "Epoch {:3d}, loss={:5.6f}".format(
                epoch,
                loss.item(),
            )
            if y is not None:
                auc = roc_auc_score(y, score.cpu().numpy())
                log += ", AUC={:6f}".format(auc)
                if auc >= max_auc:
                    max_auc = auc
                    self.model_copy = deepcopy(self.model)

            if self.verbose:
                if (epoch + 1) % self.log_interval == 0:
                    print(log)
            elif y is not None:
                process.set_postfix(
                    max_auc="max: {:.4f}, current: {:.4f}".format(
                        max_auc, auc),
                    refresh=True,
                )

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )

        return self


class STAnomalyFormerDetector_v3(STAnomalyFormerDetector_v2):

    def __init__(self,
                 d_in: int,
                 d_model: int,
                 dim_k: int,
                 dim_v: int,
                 n_heads: int,
                 n_gcn: int,
                 alpha: float = 0,
                 beta: float = 0,
                 device: str = 'cpu',
                 epoch: int = 10,
                 batch_size: int = -1,
                 lr: float = 0.0001,
                 contamination=0.1,
                 verbose: bool = False,
                 log_interval: int = 1,
                 **kwargs):
        super().__init__(d_in, d_model, dim_k, dim_v, n_heads, n_gcn, alpha,
                         device, epoch, batch_size, lr, contamination, verbose,
                         log_interval, **kwargs)
        self.beta = beta

    def fit(self, x, mat, y=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        self.model = STAnomalyFormer_v2(
            mat.to(self.device),
            n_gcn=self.n_gcn,
            **self.tsfm_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if y is not None:
            max_auc = 0.

        process = range(self.epoch) if not self.verbose else tqdm(
            range(self.epoch))
        for epoch in process:
            self.model.train()
            output, td1, td2, score_dy, score_st = self.model(x_)

            recon = torch.square(output - x_).mean((1, 2))
            # recon = region_wise_smooth_l1_loss(output, x_)
            discrepancy1 = sym_kl_loss(td1, td2.detach()) - sym_kl_loss(
                td1.detach(), td2)
            discrepancy2 = sym_kl_loss(
                score_dy,
                score_st.detach(),
            ) - sym_kl_loss(score_dy.detach(), score_st)
            score = recon + self.alpha * discrepancy2
            loss = score.mean() + self.beta * discrepancy1.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                output, _, _, score_dy, score_st = self.model(x_)

                recon = torch.square(output - x_).mean((1, 2))
                # recon = region_wise_smooth_l1_loss(output, x_)
                discrepancy = sym_kl_loss(score_dy, score_st)
                score = recon + self.alpha * discrepancy
                loss = score.mean()

            # log = "Epoch {:3d}, loss={:5.6f}".format(
            #     epoch,
            #     loss.item(),
            # )
            if y is not None:
                auc = roc_auc_score(y, score.cpu().numpy())
                # log += ", AUC={:6f}".format(auc)
                if auc >= max_auc:
                    max_auc = auc
                    self.model_copy = deepcopy(self.model)

            if self.verbose:
                # if (epoch + 1) % self.log_interval == 0:
                #     print(log)
                if y is not None:
                    process.set_postfix(
                        stat="max: {:.4f}, current: {:.4f}".format(
                            max_auc, auc),
                        refresh=True,
                    )

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )

        return self

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output, _, _, score_dy, score_st = self.model(x)
        recon = torch.square(output - x).mean((1, 2))
        # recon = region_wise_smooth_l1_loss(output, x)
        discrepancy = sym_kl_loss(score_dy, score_st)
        score = recon + self.alpha * discrepancy
        return score.cpu().numpy()


class STAnomalyFormerDetector_v4(STAnomalyFormerDetector_v3):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        n_gcn: int,
        device: str = 'cpu',
        epoch: int = 10,
        batch_size: int = -1,
        lr: float = 0.0001,
        contamination=0.1,
        verbose: bool = False,
        log_interval: int = 1,
        **kwargs,
    ):
        super().__init__(
            d_in,
            d_model,
            dim_k,
            dim_v,
            n_heads,
            n_gcn,
            1.,
            1.,
            device,
            epoch,
            batch_size,
            lr,
            contamination,
            verbose,
            log_interval,
            **kwargs,
        )
        self.loss_weight = torch.ones(3) / 3

    def fit(self, x, mat, evaluate=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        self.early_stopping = EarlyStopping(50, trace_func=tqdm.write)
        self.model = STAnomalyFormer_v2(
            mat.to(self.device),
            n_gcn=self.n_gcn,
            **self.tsfm_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        process = range(self.epoch) if not self.verbose else tqdm(
            range(self.epoch))

        for epoch in process:
            self.model.train()
            output, td1, td2, score_dy, score_st = self.model(x_)

            recon = torch.square(output - x_).mean((1, 2))
            discrepancy1 = sym_kl_loss(td1, td2.detach()) - sym_kl_loss(
                td1.detach(), td2)
            discrepancy2 = sym_kl_loss(
                score_dy,
                score_st.detach(),
            ) - sym_kl_loss(score_dy.detach(), score_st)
            score = (loss1 := self.loss_weight[0] *
                     recon) + self.loss_weight[1] * discrepancy2

            with torch.no_grad():
                loss1 = recon.mean()
                loss2 = discrepancy2.mean()
            loss3 = discrepancy1.mean()

            loss = score.mean() + self.loss_weight[2] * loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.update_weight(loss1.item(), loss2.item(), loss3.item())

            if evaluate is not None:
                test_X, test_y = evaluate
                test_x = torch.FloatTensor(test_X).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    output, _, _, score_dy, score_st = self.model(test_x)

                    recon = torch.square(output - test_x).mean((1, 2))
                    discrepancy = sym_kl_loss(score_dy, score_st)
                    score = self.loss_weight[0] * recon + \
                        self.loss_weight[1] * discrepancy
                    loss = score.mean()

                auc = roc_auc_score(test_y, score.cpu().numpy())
                self.early_stopping(auc, self.model)

                if self.early_stopping.early_stop:
                    break

            if self.verbose:
                if evaluate is not None:
                    process.set_postfix(
                        max_auc="max: {:.4f}, current: {:.4f}".format(
                            self.early_stopping.val_score_max, auc),
                        refresh=True,
                    )
                else:
                    process.set_postfix(
                        max_auc="loss: {:.5f}".format(loss.item()),
                        refresh=True,
                    )

        self.model.load_state_dict(torch.load(self.early_stopping.path))
        self.model.eval()
        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )

        return self

    @torch.no_grad()
    def update_weight(self, loss1, loss2, loss3):
        exp = torch.tensor([-loss1, -loss2, -loss3]).exp()
        new_weight = self.loss_weight * exp
        self.loss_weight = new_weight / torch.sum(new_weight)

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output, _, _, score_dy, score_st = self.model(x)
        recon = torch.square(output - x).mean((1, 2))
        discrepancy = sym_kl_loss(score_dy, score_st)
        score = self.loss_weight[0] * recon + self.loss_weight[1] * discrepancy
        return score.cpu().numpy()


class STPatchFormerDetector(BaseDetector):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        temporal_half: bool = False,
        spatial_half: bool = False,
        n_gcn: int = 3,
        device: str = 'cuda',
        epoch: int = 50,
        lr: float = 0.001,
        early_stopping: bool = False,
        use_recon: bool = True,
        use_const: bool = True,
        diff_const: bool = True,
        static_only=False,
        dynamic_only=False,
        contamination: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__(contamination)
        self.model_args = {
            "seq_len": seq_len,
            "patch_len": patch_len,
            "stride": stride,
            "d_in": d_in,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_gcn": n_gcn,
            "temporal_half": temporal_half,
            "spatial_half": spatial_half,
            "static_only": static_only,
            "dynamic_only": dynamic_only,
        }
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.is_early_stopping = early_stopping
        self.verbose = verbose
        self.use_recon = use_recon
        self.use_const = use_const
        self.diff_const = diff_const
        assert self.use_recon or self.use_const

        self.loss_weight = torch.ones(2) / 2

    def fit(self, x, mat, evaluate=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        self.early_stopping = EarlyStopping(
            100,
            trace_func=tqdm.write,
            delta=0.01,
        )
        self.model = STPatchFormer(
            dist_mat=mat.to(self.device),
            **self.model_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        process = range(self.epoch) if not self.verbose else tqdm(
            range(self.epoch))

        for epoch in process:
            self.model.train()
            (patch_x, patch_recon), (score_dy, score_st) = self.model(x_)
            if self.use_recon:
                recon = torch.abs(patch_x - patch_recon).mean((1, 2, 3))
                loss1 = self.loss_weight[0] * recon.mean()
            if self.use_const:
                if self.diff_const:
                    discrepancy = sym_kl_loss(score_dy,
                                              score_st.detach()) - sym_kl_loss(
                                                  score_dy.detach(), score_st)
                else:
                    discrepancy = sym_kl_loss(score_dy, score_st)
                loss2 = self.loss_weight[1] * discrepancy.mean()

            optimizer.zero_grad()
            if self.use_recon and not self.use_const:
                loss = loss1
            elif self.use_const and not self.use_recon:
                loss = loss2
            else:
                loss = loss1 + loss2
                self.update_weight(loss1.item(), loss2.item())

            loss.backward()
            optimizer.step()

            if evaluate is not None:
                auc = roc_auc_score(
                    evaluate[1],
                    self.decision_function(evaluate[0]),
                )
                self.early_stopping(auc, self.model)

                if self.is_early_stopping and self.early_stopping.early_stop:
                    # print("Early stopping")
                    break

            if self.verbose:
                if evaluate is not None:
                    process.set_postfix(
                        max_auc="AUC: {:.3f}/{:.3f}, weight : {:.3f}".format(
                            auc, self.early_stopping.best_score,
                            self.loss_weight[0]),
                        refresh=True,
                    )
                else:
                    process.set_postfix(
                        max_auc="loss: {:.5f}".format(loss.item()),
                        refresh=True,
                    )
        self.model.load_state_dict(torch.load(self.early_stopping.path))
        self.model.eval()
        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def update_weight(self, loss1, loss2):
        exp = torch.tensor([-loss1, -loss2]).exp()
        new_weight = self.loss_weight * exp
        self.loss_weight = new_weight / torch.sum(new_weight)

    @torch.no_grad()
    def decision_function(self, x):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        (patch_x, patch_recon), (score_dy, score_st) = self.model(x)
        if self.use_recon:
            recon = torch.abs(patch_x - patch_recon).mean((1, 2, 3))
            score1 = self.loss_weight[0] * recon
        if self.use_const:
            discrepancy = sym_kl_loss(score_dy, score_st)
            score2 = self.loss_weight[1] * discrepancy

        if self.use_recon and not self.use_const:
            score = score1
        elif self.use_const and not self.use_recon:
            score = score2
        else:
            score = score1 + score2
        return score.cpu().numpy()

    def predict(self, x):
        score = self.decision_function(x)
        return predict_by_score(score, 1 - self.contamination)


class STPMFormerDector(STPatchFormerDetector):

    def __init__(self,
                 seq_len: int,
                 patch_len: int,
                 stride: int,
                 d_in: int,
                 d_model: int,
                 n_heads: int,
                 n_gcn: int = 3,
                 temporal_half: bool = False,
                 spatial_half: bool = False,
                 mask_ratio: float = 0.4,
                 device: str = 'cuda',
                 epoch: int = 50,
                 lr: float = 0.001,
                 contamination: float = 0.1,
                 verbose: bool = False):
        super().__init__(seq_len, patch_len, stride, d_in, d_model, n_heads,
                         temporal_half, spatial_half, n_gcn, device, epoch, lr,
                         contamination, verbose)
        self.model_args['mask_ratio'] = mask_ratio

    def fit(self, x, mat, evaluate=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        self.early_stopping = EarlyStopping(100, trace_func=tqdm.write)
        self.model = STPatchMaskFormer(
            dist_mat=mat.to(self.device),
            **self.model_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        process = range(self.epoch) if not self.verbose else tqdm(
            range(self.epoch))

        for epoch in process:
            self.model.train()
            (patch_x, patch_recon), (score_dy, score_st) = self.model(x_)
            mask = self.model.random_mask.mask
            recon_loss = torch.abs(patch_x - patch_recon).mean(-2)
            loss1 = (recon_loss * mask).sum() / mask.sum()
            loss2 = sym_kl_loss(score_dy, score_st).sum()
            loss = self.loss_weight[0] * loss1 + self.loss_weight[1] * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.update_weight(loss1.item(), loss2.item())

            if evaluate is not None:
                auc = roc_auc_score(
                    evaluate[1],
                    self.decision_function(evaluate[0]),
                )
                self.early_stopping(auc, self.model)

                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

            if self.verbose:
                if evaluate is not None:
                    process.set_postfix(
                        max_auc="max: {:.4f}, current: {:.4f}".format(
                            self.early_stopping.val_score_max, auc),
                        refresh=True,
                    )
                else:
                    process.set_postfix(
                        max_auc="loss: {:.5f}".format(loss.item()),
                        refresh=True,
                    )
        self.model.load_state_dict(torch.load(self.early_stopping.path))
        self.model.eval()
        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, x):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        (patch_x, patch_recon), (score_dy, score_st) = self.model(x)
        recon = torch.abs(patch_x - patch_recon).mean(-2).sum((-1, -2))
        discrepancy = sym_kl_loss(score_dy, score_st)
        score = self.loss_weight[0] * recon + self.loss_weight[1] * discrepancy
        return score.cpu().numpy()


class STPatch_MGCNDetector(STPatchFormerDetector):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        temporal_half: bool = False,
        spatial_half: bool = False,
        n_gcn: int = 3,
        device: str = 'cuda',
        epoch: int = 50,
        lr: float = 0.001,
        early_stopping: bool = False,
        contamination: float = 0.1,
        use_recon: bool = True,
        use_const: bool = True,
        diff_const: bool = True,
        static_only=False,
        dynamic_only=False,
        verbose: bool = False,
    ):
        super().__init__(seq_len, patch_len, stride, d_in, d_model, n_heads,
                         temporal_half, spatial_half, n_gcn, device, epoch, lr,
                         early_stopping, use_recon, use_const, diff_const,
                         static_only, dynamic_only, contamination, verbose)

    def fit(self, x, mats, evaluate=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mats = torch.FloatTensor(mats).to(self.device)
        self.early_stopping = EarlyStopping(
            100,
            trace_func=tqdm.write,
            delta=0.01,
        )
        self.model = STPatch_MGCNFormer(
            dist_mats=mats.to(self.device),
            **self.model_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        process = range(self.epoch) if not self.verbose else tqdm(
            range(self.epoch))

        for epoch in process:
            self.model.train()
            # (score_dy, score_st) = self.model(x_, return_recon=False)
            (patch_x, patch_recon), (score_dy, score_st) = self.model(x_)
            if self.use_recon:
                recon = torch.abs(patch_x - patch_recon).mean((1, 2, 3))
                loss1 = self.loss_weight[0] * recon.mean()
            if self.use_const:
                if self.diff_const:
                    discrepancy = sym_kl_loss(score_dy,
                                              score_st.detach()) - sym_kl_loss(
                                                  score_dy.detach(), score_st)
                else:
                    discrepancy = sym_kl_loss(score_dy, score_st)
                loss2 = self.loss_weight[1] * discrepancy.mean()
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            if self.use_recon and not self.use_const:
                loss = loss1
            elif self.use_const and not self.use_recon:
                loss = loss2
            else:
                loss = loss1 + loss2
                self.update_weight(loss1.item(), loss2.item())
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            if evaluate is not None:
                auc = roc_auc_score(
                    evaluate[1],
                    self.decision_function(evaluate[0]),
                )
                self.early_stopping(auc, self.model)

                if self.is_early_stopping and self.early_stopping.early_stop:
                    # print("Early stopping")
                    break

            if self.verbose:
                if evaluate is not None:
                    process.set_postfix(
                        max_auc="AUC: {:.3f}/{:.3f}, weight : {:.3f}".format(
                            auc, self.early_stopping.best_score,
                            self.loss_weight[0]),
                        refresh=True,
                    )
                else:
                    process.set_postfix(
                        max_auc="loss: {:.5f}".format(loss.item()),
                        refresh=True,
                    )
        self.model.load_state_dict(torch.load(self.early_stopping.path))
        self.model.eval()
        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self
