import torch
import numpy as np


def precision_k(actual, predicted, k):
    sort_index = np.argsort(predicted)
    return np.sum(actual[sort_index[-k:]]) / float(k)


def recall_k(actual, predicted, k):
    sort_index = np.argsort(predicted)
    return np.sum(actual[sort_index[-k:]]) / np.sum(actual)


def predict_by_score(
    score: np.ndarray,
    contamination: float,
    return_threshold: bool = False,
):
    pred = np.zeros_like(score)
    threshold = np.percentile(score, contamination)
    pred[score > threshold] = 1
    if return_threshold:
        return pred, threshold
    return pred


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path='checkpoint.pt',
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(score, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = max(score, self.best_score)
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     self.trace_func(
        #         f'Validation Score decreased ({self.val_score_max:.6f} --> {val_score:.6f}).  Saving model ...'
        #     )
        torch.save(model.state_dict(), self.path)
