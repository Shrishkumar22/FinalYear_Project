# =====================================================
# INTEGRATED STUDENT (OLD SKELETON + NEW TRAINING LOGIC)
# =====================================================

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from utils import *
from utils import AE_classifier

device = torch.device("cpu")

# -----------------------------
# CONFIGURATION
# -----------------------------

sample_interval = 5000
MAX_WINDOWS = 10          # ← ENFORCED: only process 10 windows (~50k samples) per round
drift_threshold = 0.05
num_labeled_sample = 50
memory = 1000
epoch_1 = 3               # ← reduced from 5 to 3 epochs to speed up further
bs = 128

new_sample_weight = 3
lwf_lambda = 0.5

old_init = "0-0.5"
new_init = "0-1"
opt_old_lr = 0.01
opt_new_lr = 0.01

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "client_metrics.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)


# =====================================================
# IMPROVED LOSSES
# =====================================================

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
        mask_sum = mask.sum(1).clamp(min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        return -mean_log_prob_pos.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs.squeeze(), targets.float(), reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        return self.alpha * (1 - pt) ** self.gamma * BCE_loss


# =====================================================
# DATA LOADER
# =====================================================

def load_online_data_csv(memory_ratio: float = 0.20, datapath: str = "client1_test.csv"):
    data = load_data(datapath)
    splitter = SplitData(dataset="new")
    X_all, Y_all = splitter.transform(data, labels="label")

    X_all = torch.FloatTensor(X_all)
    Y_all = torch.LongTensor(Y_all)

    x_train, x_stream, y_train, y_stream = train_test_split(
        X_all, Y_all, test_size=(1 - memory_ratio),
        shuffle=True,
        random_state=42  # ← add this so same split every round
    )

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    perm = torch.randperm(x_stream.size(0))
    x_stream = x_stream[perm].to(device)
    y_stream = y_stream[perm].to(device)

    return x_train, y_train, x_stream, y_stream


def load_online_data(datapath: str = "client1_test.csv"):
    return load_online_data_csv(datapath=datapath)


# =====================================================
# ONLINE ADAPTATION
# =====================================================

def online_adaptation(model, teacher_model,
                      x_train, y_train,
                      x_test, y_test,
                      round_number=1,
                      device=None):

    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    teacher_model.to(device)

    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.2)
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    recon_loss_fn = nn.L1Loss()

    contrastive_weight = 0.4
    cls_weight = 1.2
    recon_weight = 0.001
    decision_threshold = 0.45

    y_train_detection = torch.empty(0, dtype=torch.long, device=device)
    labeled_indices = []

    start_idx = 0
    count = 0
    total_windows = min(MAX_WINDOWS, len(x_test) // sample_interval + 1)

    print(f"[TRAIN] Starting training: {total_windows} windows x {sample_interval} samples x {epoch_1} epochs", flush=True)

    while start_idx < len(x_test) and count < MAX_WINDOWS:  # ← MAX_WINDOWS enforced here

        count += 1
        end_idx = min(start_idx + sample_interval, len(x_test))
        x_test_this_epoch = x_test[start_idx:end_idx]
        y_test_this_epoch = y_test[start_idx:end_idx]
        start_idx += sample_interval

        print(f"[TRAIN] Window {count}/{total_windows} | samples {start_idx-sample_interval}-{end_idx}", flush=True)

        # ---------------- Drift detection ----------------
        with torch.no_grad():
            _, _, test_logits = model(x_test_this_epoch)
            _, _, train_logits = model(x_train)

        drift = detect_drift(test_logits, train_logits, sample_interval, drift_threshold)

        if drift:
            print(f"[TRAIN] !! Drift detected in window {count}", flush=True)
        else:
            print(f"[TRAIN] No drift in window {count}", flush=True)

        # ---------------- Mask optimization ----------------
        control_res = train_logits.cpu().numpy()
        treatment_res = test_logits.cpu().numpy()

        M_c = optimize_old_mask(control_res, treatment_res, device,
                                initialization=old_init, lr=opt_old_lr)
        M_t = optimize_new_mask(control_res, treatment_res, M_c, device,
                                initialization=new_init, lr=opt_new_lr)

        # ---------------- Sample selection ----------------
        if drift:
            x_train, y_train, labeled_idx, new_mask = \
                select_and_update_representative_samples_when_drift(
                    x_train, y_train,
                    x_test_this_epoch, y_test_this_epoch,
                    M_c, M_t,
                    num_labeled_sample,
                    device, memory, model)
        else:
            x_train, y_train, labeled_idx, new_mask = \
                select_and_update_representative_samples(
                    x_train, y_train,
                    x_test_this_epoch, y_test_this_epoch,
                    M_c, M_t,
                    num_labeled_sample,
                    device)

        labeled_indices.append(start_idx - sample_interval + labeled_idx.cpu().numpy())

        # ---------------- Training ----------------
        train_loader = DataLoader(
            TensorDataset(x_train, y_train, new_mask),
            batch_size=bs, shuffle=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        model.train()

        for epoch in range(epoch_1):
            epoch_loss = 0.0
            for inputs, labels, mask_new in train_loader:
                optimizer.zero_grad()
                features, recon_vec, logits = model(inputs)

                contrastive_loss = contrastive_loss_fn(features, labels)
                focal_loss = focal_loss_fn(logits, labels)
                weighted_focal = focal_loss * ((1 - mask_new) + mask_new * new_sample_weight)
                class_loss = weighted_focal.mean()
                recon_loss = recon_loss_fn(recon_vec, inputs)

                with torch.no_grad():
                    _, _, teacher_logits = teacher_model(inputs)
                dist_loss = F.mse_loss(logits, teacher_logits)

                total_loss = (
                    contrastive_weight * contrastive_loss +
                    cls_weight * class_loss +
                    recon_weight * recon_loss +
                    lwf_lambda * dist_loss
                )

                if torch.isnan(total_loss):
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += total_loss.item()

            print(f"[TRAIN] Window {count} Epoch {epoch+1}/{epoch_1} loss={epoch_loss:.4f}", flush=True)

        teacher_model.load_state_dict(model.state_dict())

        # ---------------- Prediction ----------------
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(x_test_this_epoch), bs):
                batch = x_test_this_epoch[i:i+bs]
                _, _, logits = model(batch)
                batch_preds = (torch.sigmoid(logits.squeeze()) > decision_threshold).long()
                preds.extend(batch_preds.cpu().numpy())

        y_train_detection = torch.cat((
            y_train_detection,
            torch.tensor(preds, device=device)
        ))

        print(f"[TRAIN] Window {count} done", flush=True)

    # ---------------- FINAL METRICS ----------------
    processed_len = min(count * sample_interval, len(x_test))
    all_labeled_indices = np.hstack(labeled_indices)
    mask = np.ones(processed_len, dtype=bool)
    valid_indices = all_labeled_indices[all_labeled_indices < processed_len]
    mask[valid_indices] = False

    y_test_pseudo = y_train_detection[:processed_len][mask]
    y_test_true = y_test[:processed_len][mask]

    perf = score_detail(y_test_true.cpu().numpy(), y_test_pseudo.cpu().numpy())

    print(f"[TRAIN] Round {round_number} metrics: {perf}", flush=True)

    log_entry = {"round": round_number, "metrics": perf}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return model


# =====================================================
# MAIN ENTRY
# =====================================================

def main(teacher_model, round, id, device=None):
    if device is None:
        device = torch.device("cpu")

    teacher_model.to(device)
    teacher_model.eval()

    model = AE_classifier(input_dim=17).to(device)
    model.load_state_dict(teacher_model.state_dict())

    data_path = f"{id}_test.csv"
    if not os.path.exists(data_path):
        data_path = f"/app/{id}_test.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")

    print(f"[TRAIN] Loading data from {data_path}", flush=True)
    x_train, y_train, x_test, y_test = load_online_data(datapath=data_path)
    print(f"[TRAIN] Data loaded: x_train={x_train.shape}, x_test={x_test.shape}", flush=True)

    final_model = online_adaptation(
        model, teacher_model,
        x_train, y_train,
        x_test, y_test,
        round_number=round,
        device=device,
    )

    torch.save(final_model.state_dict(), f"updated_student_{id}.pth")
    print(f"[TRAIN] Model saved: updated_student_{id}.pth", flush=True)
    return final_model
