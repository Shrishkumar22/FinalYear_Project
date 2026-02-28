import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from utils import *
from utils import AE_classifier  # or your model class

# -----------------------------
#      CONFIGURATION
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_interval = 5000
drift_threshold = 0.05
num_labeled_sample = 50
memory = 1000
epoch_1 = 5
bs = 128

new_sample_weight = 3
lwf_lambda = 0.5

old_init = "0-0.5"
new_init = "0-1"
opt_old_lr = 0.01
opt_new_lr = 0.01

dataset_type = "new"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "client_metrics.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)
def load_models(model_path):
    # Instantiate the model
    teacher_model = AE_classifier(input_dim=17).to(device)

    # Load weights
    teacher_model.load_state_dict(torch.load(model_path, map_location=device))

    teacher_model.eval()
    return teacher_model

# def load_online_dummy_data(datasetpath:str):
#     """
#     Returns (x_train, y_train), (x_test, y_test) as torch tensors on CPU.
#     """
    
#     UNSWTrain_dataset_path = datasetpath
#     UNSWTest_dataset_path = datasetpath
#     UNSWTrain = load_data(UNSWTrain_dataset_path)
#     UNSWTest = load_data(UNSWTest_dataset_path)
#     splitter = SplitData(dataset="new")
#     x_train, y_train = splitter.transform(UNSWTrain, labels="label")
#     x_test, y_test = splitter.transform(UNSWTest, labels="label")

#     x_train = torch.FloatTensor(x_train)
#     y_train = torch.LongTensor(y_train)
#     x_test = torch.FloatTensor(x_test)
#     y_test = torch.LongTensor(y_test)

#     print('shape of x_train ', x_train.shape)
#     print('shape of x_test is ', x_test.shape)
#     percent = 0.2
#     online_x_train, online_x_test, online_y_train, online_y_test = train_test_split(x_train, y_train, test_size=percent)
    
#     print('size of memory is ', memory)
#     memory = math.floor(memory)

#     online_x_train, online_y_train  = online_x_train.to(device), online_y_train.to(device)

#     x_train_this_epoch, x_test_left_epoch, y_train_this_epoch, y_test_left_epoch = online_x_train.clone(), online_x_test.clone().to(device), online_y_train.clone(), online_y_test.clone()
    
#     permutation = torch.randperm(x_test.size(0))

#     # Apply this permutation to both tensors to shuffle them in unison.
#     x_test = x_test[permutation]
#     y_test = y_test[permutation]
#     # Concatenating x_test and x_test_left_epoch
#     x_test_left_epoch = torch.cat((x_test_left_epoch, x_test), dim=0)
#     print('shape of x_test_left_epoch is ', x_test_left_epoch.shape)
#     # Concatenating y_test and y_test_left_epoch
#     y_test_left_epoch = torch.cat((y_test_left_epoch, y_test), dim=0)

#     return (x_train_this_epoch, y_train_this_epoch,x_test_left_epoch, y_test_left_epoch)
# def load_online_data_csv(percent_online: float = 0.10,datapath: str = "client1_test.csv"):
#     """
#     Step 1: Take only 30% of the entire dataset for online learning.
#     Step 2: Split that 30% into:
#             - x_train_this_epoch (initial memory)
#             - x_test_left_epoch  (online streaming)
#     Teacher test set is handled separately using load_offline_data().
#     """

#     # Load dataset ONCE
#     data = load_data(datapath)

#     splitter = SplitData(dataset="new")
#     X_all, Y_all = splitter.transform(data, labels="label")

#     X_all = torch.FloatTensor(X_all)
#     Y_all = torch.LongTensor(Y_all)

#     print("Full dataset size:", X_all.shape)

#     # -------------------------------------------------------
#     #         GET ONLY 30% OF THE ENTIRE DATASET
#     # -------------------------------------------------------
#     x_online_all, _, y_online_all, _ = train_test_split(
#         X_all, Y_all,
#         test_size=(1 - percent_online),   # keep 30%
#         shuffle=True
#     )

#     print(f"30% online dataset size: {x_online_all.shape}")

#     # -------------------------------------------------------
#     #     FROM THE 30%, CREATE MEMORY + STREAMING
#     # -------------------------------------------------------
#     x_train_this_epoch, x_test_left_epoch, y_train_this_epoch, y_test_left_epoch = train_test_split(
#         x_online_all, y_online_all,
#         test_size=0.8,   # 80% stream, 20% memory (best practice)
#         shuffle=True
#     )

#     print(f"Initial memory size: {x_train_this_epoch.shape}")
#     print(f"Online streaming size: {x_test_left_epoch.shape}")

#     # Move memory portion to device
#     x_train_this_epoch = x_train_this_epoch.to(device)
#     y_train_this_epoch = y_train_this_epoch.to(device)

#     # Shuffle stream data and move to device
#     perm = torch.randperm(x_test_left_epoch.size(0))
#     x_test_left_epoch = x_test_left_epoch[perm].to(device)
#     y_test_left_epoch = y_test_left_epoch[perm].to(device)

#     return (
#         x_train_this_epoch, 
#         y_train_this_epoch,
#         x_test_left_epoch, 
#         y_test_left_epoch
#     )

def load_online_data_csv(
    memory_ratio: float = 0.20,   # % kept as initial memory
    datapath: str = "client1_test.csv"
):
    """
    Uses 100% of client1_test.csv for online learning.

    Step 1: Load full dataset
    Step 2: Split into:
            - x_train_this_epoch (initial memory)
            - x_test_left_epoch  (online streaming)
    """

    # -------------------- LOAD DATA --------------------
    data = load_data(datapath)

    splitter = SplitData(dataset="new")
    X_all, Y_all = splitter.transform(data, labels="label")

    X_all = torch.FloatTensor(X_all)
    Y_all = torch.LongTensor(Y_all)

    print("Full dataset size:", X_all.shape)

    # ---------------- MEMORY + STREAM SPLIT ----------------
    x_train_this_epoch, x_test_left_epoch, \
    y_train_this_epoch, y_test_left_epoch = train_test_split(
        X_all,
        Y_all,
        test_size=(1 - memory_ratio),  # remaining is stream
        shuffle=True
    )

    print(f"Initial memory size: {x_train_this_epoch.shape}")
    print(f"Online streaming size: {x_test_left_epoch.shape}")

    # ---------------- MOVE TO DEVICE ----------------
    x_train_this_epoch = x_train_this_epoch.to(device)
    y_train_this_epoch = y_train_this_epoch.to(device)

    # Shuffle stream to simulate real-time arrival
    perm = torch.randperm(x_test_left_epoch.size(0))
    x_test_left_epoch = x_test_left_epoch[perm].to(device)
    y_test_left_epoch = y_test_left_epoch[perm].to(device)

    return (
        x_train_this_epoch,
        y_train_this_epoch,
        x_test_left_epoch,
        y_test_left_epoch
    )


# -----------------------------------------------------------
#        LOAD DATA FOR ONLINE TRAINING
# -----------------------------------------------------------
def load_online_data(datapath: str = "client1_test.csv"):
    # YOU MUST LOAD THESE:
    # x_train_this_epoch, y_train_this_epoch
    # x_test_left_epoch, y_test_left_epoch
    return load_online_data_csv(datapath=datapath)

    # data = torch.load(path, map_location=device)
    # return (data["x_train"],
    #         data["y_train"],
    #         data["x_test"],
    #         data["y_test"])


# -----------------------------------------------------------
#                 ONLINE TRAINING LOOP
# -----------------------------------------------------------

# ---------------------------
# Online adaptation function (Triplet Loss)
# ---------------------------
def online_adaptation(model, teacher_model,
                      x_train, y_train,
                      x_test, y_test,
                      sample_interval=1000,
                      bs=1000,
                      epoch_1=1,
                      new_sample_weight=2.0,
                      lwf_lambda=0.5,
                      round_number=1,
                      device=None):
    if device is None:
        device = next(model.parameters()).device

   
    model.to(device)
    teacher_model.to(device)
    
    y_train_detection = torch.empty(0, dtype=torch.long, device=device)
    labeled_indices = []
    start_idx = 0
    count = 0

    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2.0)
    classification_criterion = nn.BCEWithLogitsLoss()

    while start_idx < len(x_test):

        # print(f"\n▶ Iter = {count}, start_idx = {start_idx}")
        count += 1

        end_idx = min(start_idx + sample_interval, len(x_test))
        x_test_this_epoch = x_test[start_idx:end_idx].to(device)
        y_test_this_epoch = y_test[start_idx:end_idx].to(device)
        start_idx += sample_interval

        # ---------------------------
        # Drift detection
        # ---------------------------
        with torch.no_grad():
            _, _, test_logits = model(x_test_this_epoch)
            _, _, train_logits = model(x_train.to(device))

        drift = detect_drift(test_logits, train_logits, sample_interval, drift_threshold)
        print("Drift detected:", drift)

        # ---------------------------
        # Mask optimization
        # ---------------------------
        control_res = train_logits.cpu().numpy()
        treatment_res = test_logits.cpu().numpy()
        M_c = optimize_old_mask(control_res, treatment_res, device, initialization=old_init, lr=opt_old_lr)
        M_t = optimize_new_mask(control_res, treatment_res, M_c, device, initialization=new_init, lr=opt_new_lr)

        # ---------------------------
        # Sample selection
        # ---------------------------
        if drift:
            print("⚠ Drift detected — updating representatives...")
            x_train, y_train, labeled_idx, new_mask = select_and_update_representative_samples_when_drift(
                x_train, y_train, x_test_this_epoch, y_test_this_epoch,
                M_c, M_t, num_labeled_sample, device, memory, model
            )
        else:
            x_train, y_train, labeled_idx, new_mask = select_and_update_representative_samples(
                x_train, y_train,
                x_test_this_epoch, y_test_this_epoch,
                M_c, M_t, num_labeled_sample, device
            )

        labeled_indices.append(start_idx - sample_interval + labeled_idx.cpu().numpy())

        # ---------------------------
        # Retrain model
        # ---------------------------
        train_loader = DataLoader(
            TensorDataset(x_train.to(device), y_train.to(device), new_mask.to(device)),
            batch_size=bs, shuffle=True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()

        for epoch in range(epoch_1):
            for inputs, labels, mask_new in train_loader:
                inputs, labels, mask_new = inputs.to(device), labels.to(device), mask_new.to(device)

                optimizer.zero_grad()
                features, recon_vec, logits = model(inputs)

                # ---------------------------
                # Triplet loss on features
                # ---------------------------
                anchors, positives, negatives = [], [], []
                features_np = features.detach().cpu()
                labels_np = labels.detach().cpu()

                for i in range(len(features)):
                    anchor = features[i]
                    pos_idx = (labels_np == labels_np[i]).nonzero(as_tuple=True)[0]
                    neg_idx = (labels_np != labels_np[i]).nonzero(as_tuple=True)[0]

                    if len(pos_idx) > 1 and len(neg_idx) > 0:
                        pos = features[pos_idx[torch.randint(0, len(pos_idx), (1,))]].squeeze(0)
                        neg = features[neg_idx[torch.randint(0, len(neg_idx), (1,))]].squeeze(0)
                        anchors.append(anchor)
                        positives.append(pos)
                        negatives.append(neg)

                if len(anchors) > 0:
                    anchors = torch.stack(anchors).to(device)
                    positives = torch.stack(positives).to(device)
                    negatives = torch.stack(negatives).to(device)
                    triplet_loss = triplet_loss_fn(anchors, positives, negatives)
                else:
                    triplet_loss = torch.tensor(0.0, device=device)

                # ---------------------------
                # Classification loss
                # ---------------------------
                class_loss = classification_criterion(logits.squeeze(), labels.float())
                weighted_class_loss = class_loss * ((1 - mask_new) + mask_new * new_sample_weight)

                # ---------------------------
                # LwF distillation
                # ---------------------------
                with torch.no_grad():
                    _, _, teacher_logits = teacher_model(inputs)
                dist_loss = F.mse_loss(logits, teacher_logits)

                # ---------------------------
                # Total loss
                # ---------------------------
                total_loss = triplet_loss + weighted_class_loss.mean() + lwf_lambda * dist_loss
                total_loss.backward()
                optimizer.step()

        # Update teacher model
        teacher_model.load_state_dict(model.state_dict())

        # ---------------------------
        # Prediction for metrics
        # ---------------------------
        test_loader = DataLoader(TensorDataset(x_test_this_epoch, y_test_this_epoch),
                                 batch_size=bs, shuffle=False)
        pred = evaluate_classifier(model, test_loader, device, get_predict=True)
        y_train_detection = torch.cat((y_train_detection, torch.tensor(pred, device=device)))

    # ---------------------------
    # Final metrics
    # ---------------------------
    print("\n================= FINAL METRICS =================")
    all_labeled_indices = np.hstack(labeled_indices)
    mask = np.ones(len(x_test), dtype=bool)
    mask[all_labeled_indices] = False

    y_test_pseudo = y_train_detection[-len(x_test):][mask].to(device)
    y_test_true = y_test[mask].to(device)

    perf = score_detail(y_test_true.cpu().numpy(), y_test_pseudo.cpu().numpy())
    log_entry = {
    "client_id": id,
    "round": round_number,
    "metrics": perf
}

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    

    return model


# -----------------------------------------------------------
#                      MAIN ENTRY
# -----------------------------------------------------------
# ... [all your existing code above] ...

# -----------------------------------------------------------
#                      MAIN ENTRY
# -----------------------------------------------------------
def main(teacher_model,round,id):
    """Main function to be called from other modules"""


    print("Loading teacher model from global server...")
    teacher_model.to(device)
    teacher_model.eval()

    # student starts as copy of teacher
    model = AE_classifier(input_dim=17).to(device)
    model.load_state_dict(teacher_model.state_dict())

    print("checking if data available...")
    
    if id == 1:
        data_path = "client1_test.csv"
    else:
        data_path = "/app/client2_test.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Please ensure it's in the Docker container.")

    x_train, y_train, x_test, y_test = load_online_data(datapath=data_path)

    final_model = online_adaptation(
        model,
        teacher_model,
        x_train,
        y_train,
        x_test,
        y_test,
        round_number=round,
        device=device,
    )
    print("round "+str(round)+" completed")

    torch.save(final_model.state_dict(), f"updated_student_{id}.pth")
    print("\nSaved updated model successfully.")
    return final_model


if __name__ == "__main__":
    main()

# if __name__ == "__main__":

    

#     model_path = "teacher.pth"
#     print(f"Loading teacher model from: {model_path}")
#     model = AE_classifier(30).to(device)

#     teacher_model = load_models(model_path)
#     print("checking if data available...")
#     if not os.path.exists("merged_normalized.csv"):
#         raise FileNotFoundError("merged_normalized.csv not found. Please run collect_data.py to generate it.")
#     x_train, y_train, x_test, y_test = load_online_data(datapath="merged_normalized.csv")

#     final_model = online_adaptation(model, teacher_model,
#                                     x_train, y_train,
#                                     x_test, y_test,
#                                     device=device)



#     torch.save(final_model.state_dict(), "updated_model.pth")
#     print("\nSaved updated model successfully.")