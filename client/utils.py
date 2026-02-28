import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import MinMaxScaler  # not used but kept if you need later
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import scipy.optimize as opt
import matplotlib.pyplot as plt

############################################################
# MODELS
############################################################

class AE_classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Wider & deeper for large dataset
        enc_hidden1 = 256
        enc_hidden2 = 128
        enc_hidden3 = 64
        latent = 32  # keep latent same for InfoNCE compatibility

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_hidden1),
            nn.LayerNorm(enc_hidden1),
            nn.ReLU(),

            nn.Linear(enc_hidden1, enc_hidden2),
            nn.LayerNorm(enc_hidden2),
            nn.ReLU(),

            nn.Linear(enc_hidden2, enc_hidden3),
            nn.LayerNorm(enc_hidden3),
            nn.ReLU(),

            nn.Linear(enc_hidden3, latent)  # latent (no activation here)
        )

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.Linear(latent, enc_hidden3),
            nn.BatchNorm1d(enc_hidden3),
            nn.ReLU(),

            nn.Linear(enc_hidden3, enc_hidden2),
            nn.BatchNorm1d(enc_hidden2),
            nn.ReLU(),

            nn.Linear(enc_hidden2, enc_hidden1),
            nn.BatchNorm1d(enc_hidden1),
            nn.ReLU(),

            nn.Linear(enc_hidden1, input_dim)  # reconstruction (no activation)
        )

        # ---------- Classifier ----------
        self.classifier = nn.Sequential(
    nn.Linear(latent, 64),
    nn.LayerNorm(64),
    nn.ReLU(),

    nn.Linear(64, 64),
    nn.LayerNorm(64),
    nn.ReLU(),

    nn.Linear(64, 1)
)

        # self.classifier = nn.Sequential(
        #     nn.Linear(latent, enc_hidden3),
        #     nn.BatchNorm1d(enc_hidden3),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),

        #     nn.Linear(enc_hidden3, enc_hidden3),
        #     nn.BatchNorm1d(enc_hidden3),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),

        #     nn.Linear(enc_hidden3, 1)  # final logit (no sigmoid)
        # )

        self._init_weights()

    def _init_weights(self):
        # Kaiming init for ReLU networks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return z, recon, logits


class AE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        nearest_power_of_2 = 2 ** round(np.log2(input_dim))
        second_fourth_layer_size = nearest_power_of_2 // 2
        third_layer_size = nearest_power_of_2 // 4

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


############################################################
# CONTRASTIVE LOSS
############################################################

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        features: (B, D)
        labels: (B,) or (B,1)
        """
        # normalize features
        features = F.normalize(features, p=2, dim=1)

        # pairwise similarity
        logits = torch.matmul(features, features.T) / self.temperature

        batch_size = features.size(0)
        logits_mask = torch.ones((batch_size, batch_size), device=features.device)
        logits_mask.fill_diagonal_(0)

        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float() * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        numerator = torch.sum(exp_logits * positive_mask, dim=1)
        denominator = torch.sum(exp_logits, dim=1)

        numerator = torch.clamp(numerator, min=1e-8)
        denominator = torch.clamp(denominator, min=1e-8)

        loss = -torch.log(numerator / denominator)
        return loss.mean()

class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization Loss
    Reference: VICReg (Bardes et al., 2022)
    """
    def __init__(self, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4):
        super().__init__()
        self.sim_coeff = sim_coeff  # invariance weight
        self.var_coeff = var_coeff  # variance weight
        self.cov_coeff = cov_coeff  # covariance weight
        self.eps = eps

    def forward(self, z, z_aug):
        """
        z: embeddings from original samples (B, D)
        z_aug: embeddings from augmented samples (B, D)
        """
        # --- Invariance term (MSE) ---
        sim_loss = F.mse_loss(z, z_aug)

        # --- Variance term ---
        def variance_term(x):
            std = torch.sqrt(x.var(dim=0) + self.eps)
            return torch.mean(F.relu(1 - std))
        var_loss = variance_term(z) + variance_term(z_aug)

        # --- Covariance term ---
        def covariance_term(x):
            N, D = x.size()
            x = x - x.mean(dim=0)
            cov = (x.T @ x) / (N - 1)
            off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
            return (off_diag ** 2).sum() / D
        cov_loss = covariance_term(z) + covariance_term(z_aug)

        loss = self.sim_coeff * sim_loss + self.var_coeff * var_loss + self.cov_coeff * cov_loss
        return loss

############################################################
# DATA HELPERS
############################################################

def load_data(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    return data


class SplitData(BaseEstimator, TransformerMixin):
    """
    Simple splitter for your 'new' dataset:

      - Uses 'label' as output
      - Drops 'label' from features
      - Keeps only numeric columns
      - DOES NOT normalize again (your CSV is already normalized)
    """
    def __init__(self, dataset: str):
        super().__init__()
        self.dataset = dataset

    def fit(self, X, y=None):
        return self

    def transform(self, X, labels, one_hot_label=True):
        if self.dataset != "new":
            raise ValueError("This SplitData is set up for dataset='new' only.")

        y_ = X[labels].astype("float32").values
        X_ = X.drop(labels, axis=1)

        # keep only numeric columns (drop session strings etc.)
        X_ = X_.select_dtypes(include=[np.number])

        # DO NOT NORMALIZE AGAIN – dataset is already scaled
        x_ = X_.values.astype("float32")

        return x_, y_


############################################################
# EVALUATION
############################################################

def score_detail(y_test, y_test_pred, if_print=True):
    cm = confusion_matrix(y_test, y_test_pred)
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # if if_print:
    #     print("Confusion matrix")
    #     print(cm)
    #     print("Accuracy ", acc)
    #     print("Precision ", prec)
    #     print("Recall ", rec)
    #     print("F1 score ", f1)


    return acc, prec, rec, f1


def evaluate_classifier(
    model,
    data_loader,
    device,
    threshold: float = 0.4,
    sweep_thresholds: bool = False,
    get_predict: bool = False,
):
    """
    Evaluate classifier on a DataLoader.

    - threshold: decision threshold
    - sweep_thresholds: if True, prints metrics for thresholds 0.1..0.9
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            _, _, classifications = model(inputs)
            probs = classifications.squeeze()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # main threshold
    preds = (all_probs > threshold).astype(int)
    # print(f"\n=== Evaluation at threshold = {threshold:.2f} ===")
    res = score_detail(all_labels, preds, if_print=True)

    if sweep_thresholds:
        from sklearn.metrics import precision_score, recall_score, f1_score
        # print("\n=== Threshold sweep (for analysis) ===")
        for t in np.linspace(0.1, 0.9, 9):
            p = (all_probs > t).astype(int)
            prec = precision_score(all_labels, p, zero_division=0)
            rec = recall_score(all_labels, p, zero_division=0)
            f1 = f1_score(all_labels, p, zero_division=0)
            # print(f"t={t:.2f} | P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    if get_predict:
        return preds
    else:
        return res


def evaluate_inputs(model, inputs, device):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        _, _, classifications = model(inputs)
        preds = (classifications.squeeze() > 0.3).float()
    return preds.cpu().numpy()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(f"Random seed set to: {seed}")

############################################################
# Masks optimization (M_c, M_t) and drift utilities
############################################################

def initialize_tensor(size, initialization, device):
    if initialization == '0-1':
        return torch.nn.Parameter(torch.rand(size, device=device), requires_grad=True)
    elif initialization == '0-0.5':
        return torch.nn.Parameter(torch.rand(size, device=device) * 0.5, requires_grad=True)
    elif initialization == '0.5-1':
        return torch.nn.Parameter(torch.rand(size, device=device) * 0.5 + 0.5, requires_grad=True)
    else:
        raise ValueError("Invalid initialization type. Choose from '0-1', '0-0.5', or '0.5-1'.")

def _to_1d_scores(t: torch.Tensor) -> torch.Tensor:
    """
    Make sure we have one scalar score per sample.
    If t is (N, D), average over D -> (N,).
    If t is already (N,), return as is.
    """
    if t.dim() == 1:
        return t
    return t.mean(dim=1)


def optimize_old_mask(control_res, treatment_res, device,
                      initialization='0-1',
                      num_bins=10, lr=0.1, steps=100):
    """
    Learn a 1D mask M_c of length N_control that reweights control samples
    so that their binned distribution approximates the treatment distribution.
    Scores are squashed to [0, 1] to avoid empty bins.
    """
    control_res = torch.tensor(control_res, dtype=torch.float32, device=device)
    treatment_res = torch.tensor(treatment_res, dtype=torch.float32, device=device)

    # 1D scores
    control_scores = _to_1d_scores(control_res)
    treatment_scores = _to_1d_scores(treatment_res)

    # squash to [0,1] so they always fall into bins [0,1]
    control_scores = torch.sigmoid(control_scores)
    treatment_scores = torch.sigmoid(treatment_scores)

    N_c = control_scores.size(0)
    N_t = treatment_scores.size(0)

    # One weight per control sample
    M_c = initialize_tensor(N_c, initialization, device).float()
    M_c.requires_grad_(True)

    optimizer = torch.optim.SGD([M_c], lr=lr)
    delta = 1e-4

    for step in range(steps):
        # keep weights in (delta, 1-delta)
        with torch.no_grad():
            M_c.clamp_(delta, 1 - delta)

        optimizer.zero_grad()

        bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)

        # histograms on [0,1] scores
        control_hist = torch.histc(control_scores, bins=num_bins, min=0., max=1.)
        treatment_hist = torch.histc(treatment_scores, bins=num_bins, min=0., max=1.)

        bin_obs_c = torch.zeros(num_bins, device=device)
        bin_tgt_c = torch.zeros(num_bins, device=device)

        for i in range(num_bins):
            mask_c = (control_scores >= bin_edges[i]) & (control_scores < bin_edges[i + 1])

            if mask_c.any():
                # both M_c[mask_c] and M_c are 1D -> no crazy broadcast
                bin_obs_c[i] = torch.sum(M_c[mask_c]) / torch.sum(M_c)
            else:
                bin_obs_c[i] = 0.0

            bin_tgt_c[i] = treatment_hist[i] / max(N_t, 1)

        # avoid pure zeros -> keep it differentiable
        bin_obs_c = torch.clamp(bin_obs_c, min=1e-10)
        bin_tgt_c = torch.clamp(bin_tgt_c, min=1e-10)

        bin_obs_c = bin_obs_c / bin_obs_c.sum()
        bin_tgt_c = bin_tgt_c / bin_tgt_c.sum()

        accuracy_loss_c = F.kl_div(bin_obs_c.log(), bin_tgt_c, reduction='sum')

        # If, for some pathological case, loss still doesn't depend on M_c, skip this step
        if not accuracy_loss_c.requires_grad:
            continue

        loss = accuracy_loss_c
        loss.backward()
        optimizer.step()

    return M_c.detach()


def optimize_new_mask(control_res, treatment_res, M_c, device,
                      initialization='0-1',
                      num_bins=10, lr=0.1, steps=100):
    """
    Learn a 1D mask M_t of length N_treatment for new samples, combining
    old (M_c) + new (M_t) weights in each bin to match the treatment distribution.
    Scores are squashed to [0, 1] as well.
    """
    control_res = torch.tensor(control_res, dtype=torch.float32, device=device)
    treatment_res = torch.tensor(treatment_res, dtype=torch.float32, device=device)

    control_scores = _to_1d_scores(control_res)
    treatment_scores = _to_1d_scores(treatment_res)

    control_scores = torch.sigmoid(control_scores)
    treatment_scores = torch.sigmoid(treatment_scores)

    N_c = control_scores.size(0)
    N_t = treatment_scores.size(0)

    # make sure M_c is 1D and matches N_c
    M_c = M_c.view(-1).to(device)

    # One weight per treatment sample
    M_t = initialize_tensor(N_t, initialization, device).float()
    M_t.requires_grad_(True)

    optimizer = torch.optim.SGD([M_t], lr=lr)
    delta = 1e-4

    for step in range(steps):
        with torch.no_grad():
            M_t.clamp_(delta, 1 - delta)

        optimizer.zero_grad()

        bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)

        control_hist = torch.histc(control_scores, bins=num_bins, min=0., max=1.)
        treatment_hist = torch.histc(treatment_scores, bins=num_bins, min=0., max=1.)

        bin_tgt_t = torch.zeros(num_bins, device=device)
        bin_combined = torch.zeros(num_bins, device=device)

        for i in range(num_bins):
            mask_c = (control_scores >= bin_edges[i]) & (control_scores < bin_edges[i + 1])
            mask_t = (treatment_scores >= bin_edges[i]) & (treatment_scores < bin_edges[i + 1])

            bin_tgt_t[i] = treatment_hist[i] / max(N_t, 1)

            num = 0.0
            den = 0.0

            if mask_c.any():
                num += torch.sum(M_c[mask_c])
                den += torch.sum(M_c)
            if mask_t.any():
                num += torch.sum(M_t[mask_t])
                den += torch.sum(M_t)

            bin_combined[i] = num / den if den > 0 else 0.0

        bin_combined = torch.clamp(bin_combined, min=1e-10)
        bin_tgt_t = torch.clamp(bin_tgt_t, min=1e-10)

        bin_combined = bin_combined / bin_combined.sum()
        bin_tgt_t = bin_tgt_t / bin_tgt_t.sum()

        drift_loss_t = F.kl_div(bin_combined.log(), bin_tgt_t, reduction='sum')

        if not drift_loss_t.requires_grad:
            continue

        loss = drift_loss_t
        loss.backward()
        optimizer.step()

    return M_t.detach()




def select_and_update_representative_samples(
        x_train_this_epoch, y_train_this_epoch,
        x_test_this_epoch, y_test_this_epoch,
        M_c, M_t, num_labeled_sample, device):

    M_c_bin = (M_c >= 0.3).float().to(device)
    M_t_bin = (M_t >= 0.3).float().to(device)

    representative_old = x_train_this_epoch[M_c_bin.bool()]
    representative_new = x_test_this_epoch[M_t_bin.bool()]

    # print(f"Selected representative old samples: {representative_old.shape}")
    # print(f"Selected representative new samples: {representative_new.shape}")

    old_indices = torch.arange(len(x_train_this_epoch), device=device)
    representative_old_indices = old_indices[M_c_bin.bool()]

    mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
    mask_c[representative_old_indices] = False

    non_representative_old_indices = old_indices[mask_c]
    num_to_remove = num_labeled_sample

    if len(non_representative_old_indices) < num_to_remove:
        # print(
        #     f"Not enough non-representative old samples to remove "
        #     f"({len(non_representative_old_indices)}). "
        #     f"Removing additional representative samples."
        # )
        additional_remove_needed = num_to_remove - len(non_representative_old_indices)

        remove_indices = non_representative_old_indices

        representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
        sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
        additional_remove_indices = representative_old_indices[sorted_rep_indices]

        remove_indices = torch.cat([remove_indices, additional_remove_indices])
    else:
        remove_indices = non_representative_old_indices[
            torch.randperm(len(non_representative_old_indices))[:num_to_remove]
        ]

    mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
    mask[remove_indices] = False

    x_train_this_epoch = x_train_this_epoch[mask]
    y_train_this_epoch = y_train_this_epoch[mask]

    new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

    if representative_new.shape[0] < num_labeled_sample:
        # print(
        #     f"Not enough representative new samples selected "
        #     f"({representative_new.shape[0]}). Selecting additional random samples."
        # )
        additional_samples_needed = num_labeled_sample - representative_new.shape[0]

        selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()])
        available_indices = set(torch.arange(len(x_test_this_epoch)).cpu().numpy()) - selected_indices
        available_indices = torch.tensor(list(available_indices), dtype=torch.long)

        fallback_indices = available_indices[
            torch.randperm(len(available_indices))[:additional_samples_needed]
        ]
        drift_representative_new = torch.cat(
            [representative_new, x_test_this_epoch[fallback_indices]], dim=0
        )
        new_labels = torch.cat(
            [y_test_this_epoch[M_t_bin.bool()], y_test_this_epoch[fallback_indices]], dim=0
        )
        sorted_indices_new = torch.cat(
            [torch.arange(len(representative_new)), fallback_indices], dim=0
        )
    else:
        scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
        sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
        drift_representative_new = representative_new[sorted_indices_new]
        new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

    new_sample_mask = torch.cat(
        [new_sample_mask, torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)]
    )
    x_train_this_epoch = torch.cat([x_train_this_epoch, drift_representative_new], dim=0)
    y_train_this_epoch = torch.cat([y_train_this_epoch, new_labels], dim=0)

    return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask

def select_and_update_representative_samples(x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, M_c, M_t, num_labeled_sample, device):
    M_c_bin = (M_c >= 0.3).float().to(device)
    M_t_bin = (M_t >= 0.3).float().to(device)

    representative_old = x_train_this_epoch[M_c_bin.bool()]
    representative_new = x_test_this_epoch[M_t_bin.bool()]

    # print(f"Selected representative old samples: {representative_old.shape}")
    # print(f"Selected representative new samples: {representative_new.shape}")

    old_indices = torch.arange(len(x_train_this_epoch), device=device)
    representative_old_indices = old_indices[M_c_bin.bool()]

    mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
    mask_c[representative_old_indices] = False

    non_representative_old_indices = old_indices[mask_c]
    num_to_remove = num_labeled_sample

    if len(non_representative_old_indices) < num_to_remove:
       # print(f"Not enough non-representative old samples to remove ({len(non_representative_old_indices)}). Removing additional representative samples.")
        additional_remove_needed = num_to_remove - len(non_representative_old_indices)
        
        # Remove all non-representative samples first
        remove_indices = non_representative_old_indices
        
        # Then remove the remaining number from the representative samples with the lowest scores
        representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
        sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
        additional_remove_indices = representative_old_indices[sorted_rep_indices]

        remove_indices = torch.cat([remove_indices, additional_remove_indices])
    else:
        remove_indices = non_representative_old_indices[torch.randperm(len(non_representative_old_indices))[:num_to_remove]]

    mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
    mask[remove_indices] = False

    x_train_this_epoch = x_train_this_epoch[mask]
    y_train_this_epoch = y_train_this_epoch[mask]

    new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

    if representative_new.shape[0] < num_labeled_sample:
        #print(f"Not enough representative new samples selected ({representative_new.shape[0]}). Selecting additional random samples.")
        additional_samples_needed = num_labeled_sample - representative_new.shape[0]

        selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()])
        available_indices = set(torch.arange(len(x_test_this_epoch)).cpu().numpy()) - selected_indices
        available_indices = torch.tensor(list(available_indices), dtype=torch.long)

        fallback_indices = available_indices[torch.randperm(len(available_indices))[:additional_samples_needed]]
        drift_representative_new = torch.cat([representative_new, x_test_this_epoch[fallback_indices]], dim=0)
        new_labels = torch.cat([y_test_this_epoch[M_t_bin.bool()], y_test_this_epoch[fallback_indices]], dim=0)
        sorted_indices_new = torch.cat([torch.arange(len(representative_new)), fallback_indices], dim=0)
    else:
        scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
        sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
        drift_representative_new = representative_new[sorted_indices_new]
        new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

    new_sample_mask = torch.cat([new_sample_mask, torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)])
    x_train_this_epoch = torch.cat([x_train_this_epoch, drift_representative_new], dim=0)
    y_train_this_epoch = torch.cat([y_train_this_epoch, new_labels], dim=0)

    return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask

# def select_and_update_representative_samples_when_drift(
#         x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, 
#         M_c, M_t, num_labeled_sample, device, buffer_memory_size, model, normal_recon_temp=None):

#     M_c_bin = (M_c >= 0.3).float().to(device)
#     M_t_bin = (M_t >= 0.3).float().to(device)

#     representative_old = x_train_this_epoch[M_c_bin.bool()]
#     representative_new = x_test_this_epoch[M_t_bin.bool()]

#     print(f"Selected representative old samples: {representative_old.shape}")
#     print(f"Selected representative new samples: {representative_new.shape}")

#     old_indices = torch.arange(len(x_train_this_epoch), device=device)
#     representative_old_indices = old_indices[M_c_bin.bool()]

#     mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
#     mask_c[representative_old_indices] = False

#     non_representative_old_indices = old_indices[mask_c]
#     num_to_remove = num_labeled_sample

#     # Remove all non-representative samples
#     remove_indices = non_representative_old_indices

#     if len(non_representative_old_indices) < num_to_remove:
#         print(f"Not enough non-representative old samples to remove ({len(non_representative_old_indices)}). Removing additional representative samples.")
#         additional_remove_needed = num_to_remove - len(non_representative_old_indices)
        
#         # Then remove the remaining number from the representative samples with the lowest scores
#         representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
#         sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
#         additional_remove_indices = representative_old_indices[sorted_rep_indices]

#         remove_indices = torch.cat([remove_indices, additional_remove_indices])

#     mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
#     mask[remove_indices] = False

#     x_train_this_epoch = x_train_this_epoch[mask]
#     y_train_this_epoch = y_train_this_epoch[mask]

#     new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

#     if representative_new.shape[0] < num_labeled_sample:
#         print(f"Not enough representative samples selected ({representative_new.shape[0]}). Selecting additional random samples.")
#         additional_samples_needed = num_labeled_sample - representative_new.shape[0]

#         selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()])
#         available_indices = set(torch.arange(len(x_test_this_epoch)).cpu().numpy()) - selected_indices
#         available_indices = torch.tensor(list(available_indices), dtype=torch.long)

#         fallback_indices = available_indices[torch.randperm(len(available_indices))[:additional_samples_needed]]
#         drift_representative_new = torch.cat([representative_new, x_test_this_epoch[fallback_indices]], dim=0)
#         new_labels = torch.cat([y_test_this_epoch[M_t_bin.bool()], y_test_this_epoch[fallback_indices]], dim=0)
#         sorted_indices_new = torch.cat([torch.arange(len(representative_new)), fallback_indices], dim=0)
#     else:
#         scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
#         sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
#         drift_representative_new = representative_new[sorted_indices_new]
#         new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

#     new_sample_mask = torch.cat([new_sample_mask, torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)])
#     x_train_this_epoch = torch.cat((x_train_this_epoch, drift_representative_new), dim=0)
#     y_train_this_epoch = torch.cat((y_train_this_epoch, new_labels), dim=0)

#     if len(x_train_this_epoch) < buffer_memory_size:
#         additional_samples_needed = buffer_memory_size - len(x_train_this_epoch)
#         print(f"Buffer memory has extra space for {additional_samples_needed} samples. Adding new samples with pseudo labels.")

#         if representative_new.shape[0] > num_labeled_sample:
#             remaining_new_samples = representative_new[torch.argsort(torch.tensor(scores_new), descending=True)[num_labeled_sample:]]
#             # remaining_samples_needed = num_additional_samples_needed

#             if remaining_new_samples.size(0) >= additional_samples_needed:
#                 pseudo_labeled_samples = remaining_new_samples[:additional_samples_needed]
#                 if normal_recon_temp == None:
#                     pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
#                 else:
#                     pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, pseudo_labeled_samples, 0, model)
#             else:
#                 pseudo_labeled_samples = remaining_new_samples
#                 if normal_recon_temp == None:
#                     pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
#                 else:
#                     pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, pseudo_labeled_samples, 0, model)
                
#                 random_new_additional_samples_needed = additional_samples_needed - remaining_new_samples.size(0)
                
#                 additional_indices = torch.randperm(len(x_test_this_epoch))[:random_new_additional_samples_needed]
#                 additional_pseudo_labeled_samples = x_test_this_epoch[additional_indices]
#                 if normal_recon_temp == None:
#                     additional_pseudo_labels = evaluate_inputs(model, additional_pseudo_labeled_samples, device)
#                 else:
#                     additional_pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, additional_pseudo_labeled_samples, 0, model)
                
#                 pseudo_labeled_samples = torch.cat([pseudo_labeled_samples, additional_pseudo_labeled_samples], dim=0)
#                 if normal_recon_temp == None:
#                     if random_new_additional_samples_needed > 1:
#                         pseudo_labels = torch.cat([torch.tensor(pseudo_labels), torch.tensor(additional_pseudo_labels)], dim=0)
#                     else:
#                         pseudo_labels = torch.cat([torch.tensor(pseudo_labels), torch.tensor(additional_pseudo_labels).unsqueeze(0)], dim=0)
#                 else:
#                     if random_new_additional_samples_needed > 1:
#                         pseudo_labels = torch.cat([pseudo_labels, additional_pseudo_labels], dim=0)
#                     else:
#                         pseudo_labels = torch.cat([pseudo_labels, additional_pseudo_labels.unsqueeze(0)], dim=0)
#         else:
#             additional_indices = torch.randperm(len(x_test_this_epoch))[:additional_samples_needed]
#             pseudo_labeled_samples = x_test_this_epoch[additional_indices]
#             if normal_recon_temp == None:
#                 pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
#             else:
#                 pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, pseudo_labeled_samples, 0, model)

#         x_train_this_epoch = torch.cat((x_train_this_epoch, pseudo_labeled_samples), dim=0)
#         if normal_recon_temp == None:
#             if additional_samples_needed > 1:
#                 y_train_this_epoch = torch.cat((y_train_this_epoch, torch.tensor(pseudo_labels).to(device)), dim=0)
#             else:
#                 y_train_this_epoch = torch.cat((y_train_this_epoch, torch.tensor(pseudo_labels).unsqueeze(0).to(device)), dim=0)
#         else:
#             if additional_samples_needed > 1:
#                 y_train_this_epoch = torch.cat((y_train_this_epoch, pseudo_labels.to(device)), dim=0)
#             else:
#                 y_train_this_epoch = torch.cat((y_train_this_epoch, pseudo_labels.unsqueeze(0).to(device)), dim=0)
        
#         new_sample_mask = torch.cat([new_sample_mask, torch.zeros(len(pseudo_labeled_samples), dtype=torch.float32).to(device)])

#     return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask

# 

def select_and_update_representative_samples_when_drift(
        x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch,
        M_c, M_t, num_labeled_sample, device, buffer_memory_size, model, normal_recon_temp=None):

    # -----------------------------
    # 1. Select representative old/new samples
    # -----------------------------
    M_c_bin = (M_c >= 0.3).float().to(device)
    M_t_bin = (M_t >= 0.3).float().to(device)

    representative_old = x_train_this_epoch[M_c_bin.bool()]
    representative_new = x_test_this_epoch[M_t_bin.bool()]

    # print(f"Selected representative old samples: {representative_old.shape}")
    # print(f"Selected representative new samples: {representative_new.shape}")

    old_indices = torch.arange(len(x_train_this_epoch), device=device)
    representative_old_indices = old_indices[M_c_bin.bool()]

    mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
    mask_c[representative_old_indices] = False

    non_representative_old_indices = old_indices[mask_c]
    num_to_remove = num_labeled_sample

    # remove all non-representative samples first
    remove_indices = non_representative_old_indices

    if len(non_representative_old_indices) < num_to_remove:
        # print(
        #     f"Not enough non-representative old samples to remove "
        #     f"({len(non_representative_old_indices)}). Removing additional representative samples."
        # )
        additional_remove_needed = num_to_remove - len(non_representative_old_indices)

        representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
        sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
        additional_remove_indices = representative_old_indices[sorted_rep_indices]

        remove_indices = torch.cat([remove_indices, additional_remove_indices])

    mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
    mask[remove_indices] = False

    x_train_this_epoch = x_train_this_epoch[mask]
    y_train_this_epoch = y_train_this_epoch[mask]

    # mask indicating which samples are "new" vs old
    new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

    # -----------------------------
    # 2. Add labeled/new representative samples
    # -----------------------------
    if representative_new.shape[0] < num_labeled_sample:
        # print(
        #     f"Not enough representative samples selected "
        #     f"({representative_new.shape[0]}). Selecting additional random samples."
        # )
        additional_samples_needed = num_labeled_sample - representative_new.shape[0]

        selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()])
        available_indices = set(torch.arange(len(x_test_this_epoch)).cpu().numpy()) - selected_indices
        available_indices = torch.tensor(list(available_indices), dtype=torch.long)

        fallback_indices = available_indices[torch.randperm(len(available_indices))[:additional_samples_needed]]
        drift_representative_new = torch.cat(
            [representative_new, x_test_this_epoch[fallback_indices]], dim=0
        )
        new_labels = torch.cat(
            [y_test_this_epoch[M_t_bin.bool()], y_test_this_epoch[fallback_indices]], dim=0
        )
        sorted_indices_new = torch.cat(
            [torch.arange(len(representative_new)), fallback_indices], dim=0
        )
    else:
        scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
        sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
        drift_representative_new = representative_new[sorted_indices_new]
        new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

    # mark these as new labeled samples
    new_sample_mask = torch.cat(
        [new_sample_mask, torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)]
    )
    x_train_this_epoch = torch.cat((x_train_this_epoch, drift_representative_new), dim=0)
    y_train_this_epoch = torch.cat((y_train_this_epoch, new_labels), dim=0)

    # -----------------------------
    # 3. Fill buffer with pseudo‑labeled samples if needed
    # -----------------------------
    if len(x_train_this_epoch) < buffer_memory_size:
        additional_samples_needed = buffer_memory_size - len(x_train_this_epoch)
        print(
            f"Buffer memory has extra space for {additional_samples_needed} samples. "
            f"Adding new samples with pseudo labels."
        )

        # if we still have unused representative_new, use them preferentially
        if representative_new.shape[0] > num_labeled_sample:
            scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
            # remaining reps after the top num_labeled_sample already used
            remaining_new_samples = representative_new[
                torch.argsort(torch.tensor(scores_new), descending=True)[num_labeled_sample:]
            ]

            if remaining_new_samples.size(0) >= additional_samples_needed:
                pseudo_labeled_samples = remaining_new_samples[:additional_samples_needed]
                if normal_recon_temp is None:
                    pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
                else:
                    pseudo_labels = evaluate(
                        normal_recon_temp,
                        x_train_this_epoch, y_train_this_epoch,
                        pseudo_labeled_samples, 0, model
                    )
            else:
                # use all remaining + random extra from x_test_this_epoch
                pseudo_labeled_samples = remaining_new_samples
                if normal_recon_temp is None:
                    pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
                else:
                    pseudo_labels = evaluate(
                        normal_recon_temp,
                        x_train_this_epoch, y_train_this_epoch,
                        pseudo_labeled_samples, 0, model
                    )

                random_new_additional_samples_needed = additional_samples_needed - remaining_new_samples.size(0)
                additional_indices = torch.randperm(len(x_test_this_epoch))[:random_new_additional_samples_needed]
                additional_pseudo_labeled_samples = x_test_this_epoch[additional_indices]

                if normal_recon_temp is None:
                    additional_pseudo_labels = evaluate_inputs(model, additional_pseudo_labeled_samples, device)
                else:
                    additional_pseudo_labels = evaluate(
                        normal_recon_temp,
                        x_train_this_epoch, y_train_this_epoch,
                        additional_pseudo_labeled_samples, 0, model
                    )

                # concat samples
                pseudo_labeled_samples = torch.cat(
                    [pseudo_labeled_samples, additional_pseudo_labeled_samples], dim=0
                )

                # ---- SAFE CONCAT OF LABELS (force 1‑D) ----
                base = torch.as_tensor(pseudo_labels)
                extra = torch.as_tensor(additional_pseudo_labels)
                if base.dim() == 0:
                    base = base.unsqueeze(0)
                if extra.dim() == 0:
                    extra = extra.unsqueeze(0)
                pseudo_labels = torch.cat([base, extra], dim=0)
        else:
            # no extra reps; sample directly from x_test_this_epoch
            additional_indices = torch.randperm(len(x_test_this_epoch))[:additional_samples_needed]
            pseudo_labeled_samples = x_test_this_epoch[additional_indices]
            if normal_recon_temp is None:
                pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
            else:
                pseudo_labels = evaluate(
                    normal_recon_temp,
                    x_train_this_epoch, y_train_this_epoch,
                    pseudo_labeled_samples, 0, model
                )

        # append pseudo‑labeled samples to train set
        x_train_this_epoch = torch.cat((x_train_this_epoch, pseudo_labeled_samples), dim=0)

        if normal_recon_temp is None:
            labels_tensor = torch.as_tensor(pseudo_labels, device=device)
        else:
            labels_tensor = pseudo_labels.to(device)

        if labels_tensor.dim() == 0:
            labels_tensor = labels_tensor.unsqueeze(0)

        y_train_this_epoch = torch.cat((y_train_this_epoch, labels_tensor), dim=0)

        new_sample_mask = torch.cat(
            [new_sample_mask, torch.zeros(len(pseudo_labeled_samples), dtype=torch.float32).to(device)]
        )

    return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask

# (keeping your long select_and_update_representative_samples_when_drift,
# evaluate, process_batch, drift detection as-is, since they’re not used
# in the simple offline teacher training, but they’re now syntactically correct)

EPSILON = 1e-10


def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def log_likelihood(params, data):
    mu1, sigma1, mu2, sigma2 = params
    pdf1 = gaussian_pdf(data, mu1, sigma1)
    pdf2 = gaussian_pdf(data, mu2, sigma2)

    likelihood = 0.5 * pdf1 + 0.5 * pdf2
    likelihood = np.clip(likelihood, a_min=EPSILON, a_max=None)

    if np.any(np.isnan(likelihood)):
        print("NaN values found in likelihood calculation")
        return np.inf

    return -np.sum(np.log(likelihood))


def process_batch(data, temp, layer_index, model, batch_size=128, device='cuda'):
    values = []
    model.to(device)
    temp = temp.to(device)

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(device)
        batch_features = F.normalize(model(batch)[layer_index], p=2, dim=1)
        batch_cosine_sim = F.cosine_similarity(
            batch_features, temp.reshape([-1, temp.shape[0]]), dim=1
        )
        values.append(batch_cosine_sim)
        del batch, batch_features, batch_cosine_sim
        torch.cuda.empty_cache()

    return torch.cat(values)


def evaluate(normal_recon_temp, x_train, y_train, x_test, y_test,
             model, batch_size=128, device='cuda', get_probs=False):

    model.eval()
    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)

    num_of_output = 1

    values_recon_all = []
    values_recon_normal = []
    values_recon_abnormal = []

    model.to(device)
    normal_recon_temp = normal_recon_temp.to(device)

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            features, recon_vec = model(inputs)
            values_recon_all.append(
                F.cosine_similarity(
                    F.normalize(recon_vec, p=2, dim=1),
                    normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]),
                    dim=1,
                )
            )

            normal_mask = (labels == 0)
            abnormal_mask = (labels == 1)

            if normal_mask.sum() > 0:
                values_recon_normal.append(
                    F.cosine_similarity(
                        F.normalize(recon_vec[normal_mask], p=2, dim=1),
                        normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]),
                        dim=1,
                    )
                )

            if abnormal_mask.sum() > 0:
                values_recon_abnormal.append(
                    F.cosine_similarity(
                        F.normalize(recon_vec[abnormal_mask], p=2, dim=1),
                        normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]),
                        dim=1,
                    )
                )

        values_recon_all = torch.cat(values_recon_all).cpu().numpy()
        values_recon_normal = torch.cat(values_recon_normal).cpu().numpy()
        values_recon_abnormal = torch.cat(values_recon_abnormal).cpu().numpy()

    x_test = x_test.to(device)
    values_recon_test = process_batch(
        x_test, normal_recon_temp, num_of_output, model, batch_size, device
    )

    mu3_initial = np.mean(values_recon_normal)
    sigma3_initial = np.std(values_recon_normal)
    mu4_initial = np.mean(values_recon_abnormal)
    sigma4_initial = np.std(values_recon_abnormal)

    initial_params = np.array([mu3_initial, sigma3_initial, mu4_initial, sigma4_initial])
    result = opt.minimize(
        log_likelihood, initial_params, args=(values_recon_all,), method='Nelder-Mead'
    )
    mu3_fit, sigma3_fit, mu4_fit, sigma4_fit = result.x

    if mu3_fit > mu4_fit:
        gaussian3 = Normal(mu3_fit, sigma3_fit)
        gaussian4 = Normal(mu4_fit, sigma4_fit)
    else:
        gaussian4 = Normal(mu3_fit, sigma3_fit)
        gaussian3 = Normal(mu4_fit, sigma4_fit)

    pdf3 = gaussian3.log_prob(values_recon_test.clone().detach()).exp()
    pdf4 = gaussian4.log_prob(values_recon_test.clone().detach()).exp()
    y_test_pred_4 = (pdf4 > pdf3).cpu().numpy().astype("int32")

    if get_probs:
        values_recon_test = values_recon_test.detach()
        return pdf3, pdf4, values_recon_test
    else:
        if not isinstance(y_test, int):
            if isinstance(y_test, torch.Tensor) and y_test.device != torch.device("cpu"):
                y_test = y_test.cpu().numpy()
            result_decoder = score_detail(y_test, y_test_pred_4)

        y_test_pred_no_vote = torch.from_numpy(y_test_pred_4)

        if not isinstance(y_test, int):
            result_final = score_detail(y_test, y_test_pred_no_vote, if_print=True)
            return result_decoder, result_final
        else:
            return y_test_pred_no_vote


def detect_drift(new_data, control_data, window_size, drift_threshold):
    for i in range(0, len(new_data), window_size):
        window_data = new_data[i:i + window_size]
        if len(window_data) < window_size:
            break
        ks_statistic, p_value = ks_2samp(control_data.cpu().numpy(), window_data.cpu().numpy())
        if p_value < drift_threshold:
            print(
                f"!!!!!!!!!!!!!!!!!!!!! Drift detected in window {i // window_size + 1} "
                f"(p-value: {p_value})"
            )
            return True
        else:
            print(
                f"No drift detected in window {i // window_size + 1} "
                f"(p-value: {p_value})"
            )
    return False