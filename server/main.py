import time
import base64
import io
import requests
import threading
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from utils import AE_classifier

EXPECTED_CLIENTS = 3

# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------

def encode_state_dict(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def decode_state_dict(b64_string):
    raw = base64.b64decode(b64_string.encode('utf-8'))
    buffer = io.BytesIO(raw)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu")


# -----------------------------------------------------
# Server Globals
# -----------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

app = FastAPI()

def load_model(model_path):
    model = AE_classifier(input_dim=17).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model("teacher_optimized.pth")
GLOBAL_MODEL = model.state_dict()

CURRENT_ROUND = 1
REGISTERED_CLIENTS = []
CLIENT_LOCK = threading.Lock()

TRAIN_TIMEOUT = 600
ROUND_INTERVAL = 20


# -----------------------------------------------------
# API Schemas
# -----------------------------------------------------

class RegisterRequest(BaseModel):
    client_id: str
    url: str
    trust: float  # trust sent during registration


class UpdateRequest(BaseModel):
    client_id: str
    round: int
    model_bytes: str


# -----------------------------------------------------
# Endpoints
# -----------------------------------------------------

@app.get("/status")
def status():
    return {
        "status": "server_alive",
        "round": CURRENT_ROUND,
        "registered_clients": len(REGISTERED_CLIENTS)
    }


@app.post("/register_client")
def register(req: RegisterRequest):
    with CLIENT_LOCK:
        # Avoid duplicate registration
        for c in REGISTERED_CLIENTS:
            if c["id"] == req.client_id:
                return {"status": "already_registered"}

        REGISTERED_CLIENTS.append({
            "id": req.client_id,
            "url": req.url,
            "trust": float(req.trust)
        })

    print(f"[SERVER] Registered client: {req.client_id} with trust {req.trust}")
    return {"status": "registered"}


@app.post("/submit_update")
def submit_update(req: UpdateRequest):
    print(f"[SERVER] Update received from {req.client_id}")
    return {"status": "ok"}


# -----------------------------------------------------
# Trust Weighted Aggregation
# -----------------------------------------------------

def aggregate_state_dicts(state_dicts: List[dict], trusts: List[float]):
    global GLOBAL_MODEL

    agg_dict = {}

    trust_tensor = torch.tensor(trusts, dtype=torch.float32)

    # Normalize trust values
    if trust_tensor.sum() == 0:
        trust_tensor = torch.ones_like(trust_tensor) / len(trust_tensor)
    else:
        trust_tensor = trust_tensor / trust_tensor.sum()

    for key in GLOBAL_MODEL.keys():
        tensors = [sd[key].float() for sd in state_dicts]
        stacked = torch.stack(tensors, dim=0)

        # reshape trust for broadcasting
        shape = [len(trusts)] + [1] * (stacked.dim() - 1)
        weighted = stacked * trust_tensor.view(*shape)

        agg_dict[key] = weighted.sum(dim=0)

    GLOBAL_MODEL = agg_dict


# -----------------------------------------------------
# Federated Learning Loop
# -----------------------------------------------------

def fl_loop():
    global CURRENT_ROUND, GLOBAL_MODEL

    time.sleep(3)
    print("[SERVER] FL LOOP STARTED")

    while True:
        with CLIENT_LOCK:
            clients_snapshot = REGISTERED_CLIENTS.copy()

        if len(clients_snapshot) < EXPECTED_CLIENTS:
            print(f"[SERVER] Waiting for clients... ({len(clients_snapshot)}/{EXPECTED_CLIENTS})")
            time.sleep(5)
            continue

        print(f"\n===== ROUND {CURRENT_ROUND} START =====")

        updates = []
        update_trusts = []

        for client in clients_snapshot:
            url = client["url"]
            cid = client["id"]
            trust = client["trust"]

            try:
                print(f"[SERVER] Sending global model -> {cid}")

                resp = requests.post(
                    f"{url}/train_local",
                    json={
                        "round": CURRENT_ROUND,
                        "global_model_bytes": encode_state_dict(GLOBAL_MODEL),
                    },
                    timeout=TRAIN_TIMEOUT
                )

                if resp.status_code == 200:
                    update_b64 = resp.json()["model_bytes"]
                    updates.append(decode_state_dict(update_b64))
                    update_trusts.append(trust)
                    print(f"[SERVER] Update received <- {cid}")
                else:
                    print(f"[SERVER] Bad response from {cid}")

            except Exception as e:
                print(f"[SERVER] ERROR contacting {cid}: {e}")

        if updates:
            print("[SERVER] Aggregating updates (trust-weighted)...")
            aggregate_state_dicts(updates, update_trusts)
        else:
            print("[SERVER] No updates this round")

        print(f"===== ROUND {CURRENT_ROUND} END =====\n")

        CURRENT_ROUND += 1
        time.sleep(ROUND_INTERVAL)


# Start FL loop in background
@app.on_event("startup")
def start_fl():
    threading.Thread(target=fl_loop, daemon=True).start()