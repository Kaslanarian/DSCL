import pandas as pd
import numpy as np
from STAnomalyFormer.interface.estimator import STPatch_MGCNDetector, STPatchFormerDetector

from data.load_nyc import load_dataset
from data.load_pems import load_dataset
import torch
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=int, default=3)
parser.add_argument('--n_day', type=int, default=14)
parser.add_argument('--interval', type=int, default=6)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--poi_normalize', action='store_true')

parser.add_argument('--delay', action='store_true')
parser.add_argument('--lag', type=int, default=6)
parser.add_argument('--vol', action='store_true')
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--attr', action='store_true')
parser.add_argument('--k', default=None, type=int)

parser.add_argument('--mask_ratio', default=0.4, type=float)
parser.add_argument('--patch_len', default=12, type=int)
parser.add_argument('--stride', default=6, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--n_heads', default=16, type=int)
parser.add_argument('--t_half', action='store_true')
parser.add_argument('--s_half', action='store_true')
parser.add_argument('--n_gcn', default=2, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--no_recon', action='store_true')
parser.add_argument('--no_const', action='store_true')
parser.add_argument('--diff_const', action='store_true')

parser.add_argument('--cuda', action='store_true')

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

X, val_X, test_X, adj, y = load_dataset(args)
plt.rcParams['font.family'] = 'serif'

model = STPatchFormerDetector(
    seq_len=X.shape[1],
    patch_len=args.patch_len,
    stride=args.stride,
    d_in=X.shape[2],
    d_model=args.d_model,
    n_heads=args.n_heads,
    temporal_half=args.t_half,
    spatial_half=args.s_half,
    n_gcn=args.n_gcn,
    device='cuda' if args.cuda else 'cpu',
    epoch=args.epochs,
    lr=args.lr,
    verbose=args.verbose,
    use_recon=(not args.no_recon),
    use_const=(not args.no_const),
    diff_const=args.diff_const,
).fit(X, adj, (val_X, y))

score = model.decision_function(test_X)
anomaly = np.argsort(score)[::-1]
anomaly = list(anomaly)


def inverse_normalize(x):
    # max_ = np.array([637, 627])
    # return (x * max_).astype(int)
    return x


# plt.plot(np.concatenate([x[:48, 0], x_[:48, 0]]))
# plt.xticks(np.arange(0, 48 * 2 + 1, 6))

for i in range(len(anomaly)):
    plt.figure(figsize=(7, 5))
    plt.grid(True)
    plt.grid(axis="y")  #横向划线
    plt.grid(axis="x")  #纵向划线
    plt.grid(c='gray')  #横纵向划线
    plt.grid(True, ls=":", color="gray", alpha=0.3)  #alpha代表透明度
    plt.xticks(np.arange(0, 7 * 48 + 1, 48), np.arange(0, 7 * 48 + 1, 48) // 2)
    x = inverse_normalize(X[anomaly[i]])
    x_ = inverse_normalize(test_X[anomaly[i]])
    # pd.DataFrame(x).to_csv("{}_{}.csv".format(i, anomaly[i] + 1))
    plt.plot(x)
    plt.plot(x_)
    plt.xlabel("hour")
    plt.ylabel("flow")
    plt.savefig("anomaly_{}.pdf".format(i))
    plt.clf()
