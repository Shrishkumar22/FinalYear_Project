import time
import base64
import io
import asyncio
import httpx
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from utils import AE_classifier


# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------

EXPECTED_CLIENTS = 3
TRAIN_TIMEOUT = 600
ROUND_INTERVAL = 20


# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------

def encode_state_dict(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def decode_state_dict(b64_string):
    raw = base64.b64decode(b64_string.encode("utf-8"))
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
CLIENT_LOCK = asyncio.Lock()


# -----------------------------------------------------
# API Schemas
# -----------------------------------------------------

class RegisterRequest(BaseModel):
    client_id: str
    url: str
    trust: float


class UpdateRequest(BaseModel):
    client_id: str
    round: int
    model_bytes: str


# -----------------------------------------------------
# Endpoints
# -----------------------------------------------------

@app.get("/status")
async def status():
    return {
        "status": "server_alive",
        "round": CURRENT_ROUND,
        "registered_clients": len(REGISTERED_CLIENTS),
    }


@app.post("/register_client")
async def register(req: RegisterRequest):
    async with CLIENT_LOCK:
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
async def submit_update(req: UpdateRequest):
    print(f"[SERVER] Update received from {req.client_id}")
    return {"status": "ok"}


# -----------------------------------------------------
# Trust Weighted Aggregation
# -----------------------------------------------------

def aggregate_state_dicts(state_dicts: List[dict], trusts: List[float]):
    global GLOBAL_MODEL

    agg_dict = {}
    trust_tensor = torch.tensor(trusts, dtype=torch.float32)

    # Normalize trust
    if trust_tensor.sum() == 0:
        trust_tensor = torch.ones_like(trust_tensor) / len(trust_tensor)
    else:
        trust_tensor = trust_tensor / trust_tensor.sum()

    for key in GLOBAL_MODEL.keys():
        tensors = [sd[key].float() for sd in state_dicts]
        stacked = torch.stack(tensors, dim=0)

        shape = [len(trusts)] + [1] * (stacked.dim() - 1)
        weighted = stacked * trust_tensor.view(*shape)

        agg_dict[key] = weighted.sum(dim=0)

    GLOBAL_MODEL = agg_dict


# -----------------------------------------------------
# Async Client Communication
# -----------------------------------------------------

async def contact_client(client, round_number, global_model_b64):
    url = client["url"]
    cid = client["id"]
    trust = client["trust"]

    try:
        async with httpx.AsyncClient(timeout=TRAIN_TIMEOUT) as client_http:
            resp = await client_http.post(
                f"{url}/train_local",
                json={
                    "round": round_number,
                    "global_model_bytes": global_model_b64,
                }
            )

        if resp.status_code == 200:
            update_b64 = resp.json()["model_bytes"]
            print(f"[SERVER] Update received <- {cid}")
            return decode_state_dict(update_b64), trust
        else:
            print(f"[SERVER] Bad response from {cid}")
            return None

    except Exception as e:
        print(f"[SERVER] ERROR contacting {cid}: {e}")
        return None


# -----------------------------------------------------
# Async Federated Learning Loop
# -----------------------------------------------------

async def fl_loop():
    global CURRENT_ROUND, GLOBAL_MODEL

    await asyncio.sleep(3)
    print("[SERVER] ASYNC FL LOOP STARTED")

    while True:

        async with CLIENT_LOCK:
            clients_snapshot = REGISTERED_CLIENTS.copy()

        if len(clients_snapshot) < EXPECTED_CLIENTS:
            print(f"[SERVER] Waiting for clients... ({len(clients_snapshot)}/{EXPECTED_CLIENTS})")
            await asyncio.sleep(5)
            continue

        print(f"\n===== ROUND {CURRENT_ROUND} START =====")

        global_model_b64 = encode_state_dict(GLOBAL_MODEL)

        # 🔥 Send to all clients concurrently
        tasks = [
            contact_client(client, CURRENT_ROUND, global_model_b64)
            for client in clients_snapshot
        ]

        results = await asyncio.gather(*tasks)

        updates = []
        update_trusts = []

        for result in results:
            if result is not None:
                state_dict, trust = result
                updates.append(state_dict)
                update_trusts.append(trust)

        if updates:
            print("[SERVER] Aggregating updates (trust-weighted)...")
            aggregate_state_dicts(updates, update_trusts)
        else:
            print("[SERVER] No updates this round")

        print(f"===== ROUND {CURRENT_ROUND} END =====\n")

        CURRENT_ROUND += 1
        await asyncio.sleep(ROUND_INTERVAL)


# -----------------------------------------------------
# Startup Event
# -----------------------------------------------------

@app.on_event("startup")
async def start_fl():
    asyncio.create_task(fl_loop())