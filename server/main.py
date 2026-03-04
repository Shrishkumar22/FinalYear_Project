import base64
import io
import asyncio
import httpx
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from utils import AE_classifier

EXPECTED_CLIENTS = 3
TRAIN_TIMEOUT = 600
ROUND_INTERVAL = 20

device = torch.device("cpu")
print("Using CPU server")

app = FastAPI()


# -----------------------------------------------------
# Encoding
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
# Load Initial Model
# -----------------------------------------------------

def load_model(path):
    model = AE_classifier(input_dim=17).to(device)
    model.load_state_dict(
        torch.load(path, map_location=device)
    )
    model.eval()
    return model


model = load_model("teacher_optimized.pth")
GLOBAL_MODEL = {k: v.clone().detach() for k, v in model.state_dict().items()}

CURRENT_ROUND = 1
REGISTERED_CLIENTS = []
CLIENT_LOCK = asyncio.Lock()


# -----------------------------------------------------
# Register Schema
# -----------------------------------------------------

class RegisterRequest(BaseModel):
    client_id: str
    url: str
    trust: float


# -----------------------------------------------------
# Register Endpoint
# -----------------------------------------------------

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

    print(f"[SERVER] Registered {req.client_id}")
    return {"status": "registered"}


# -----------------------------------------------------
# Aggregation
# -----------------------------------------------------

def aggregate(updates, trusts):

    global GLOBAL_MODEL

    trust_tensor = torch.tensor(trusts)
    trust_tensor = trust_tensor / trust_tensor.sum()

    agg = {}

    for key in GLOBAL_MODEL.keys():
        stacked = torch.stack([u[key].float() for u in updates])
        shape = [len(trusts)] + [1]*(stacked.dim()-1)
        weighted = stacked * trust_tensor.view(*shape)
        agg[key] = weighted.sum(0)

    GLOBAL_MODEL = agg


# -----------------------------------------------------
# Contact Client
# -----------------------------------------------------

async def contact_client(client, round_number, global_model_b64):

    try:
        async with httpx.AsyncClient(timeout=TRAIN_TIMEOUT) as http_client:
            resp = await http_client.post(
                f"{client['url']}/train_local",
                json={
                    "round": round_number,
                    "global_model_bytes": global_model_b64,
                }
            )

        if resp.status_code != 200:
            print(f"[SERVER] {client['id']} returned error")
            return None

        data = resp.json()

        if data.get("status") == "error":
            print(f"[SERVER] Client error:",
                  data.get("message"))
            return None

        update = decode_state_dict(data["model_bytes"])
        print(f"[SERVER] Update received from {client['id']}")
        return update, client["trust"]

    except Exception as e:
        print(f"[SERVER] ERROR contacting {client['id']}:", e)
        return None


# -----------------------------------------------------
# Federated Loop
# -----------------------------------------------------

async def fl_loop():

    global CURRENT_ROUND

    await asyncio.sleep(5)

    while True:

        async with CLIENT_LOCK:
            clients_snapshot = REGISTERED_CLIENTS.copy()

        if len(clients_snapshot) < EXPECTED_CLIENTS:
            print("[SERVER] Waiting for clients...")
            await asyncio.sleep(5)
            continue

        print(f"\n===== ROUND {CURRENT_ROUND} =====")

        global_b64 = encode_state_dict(GLOBAL_MODEL)

        tasks = [
            contact_client(c, CURRENT_ROUND, global_b64)
            for c in clients_snapshot
        ]

        results = await asyncio.gather(*tasks)

        updates = []
        trusts = []

        for r in results:
            if r is not None:
                updates.append(r[0])
                trusts.append(r[1])

        if updates:
            aggregate(updates, trusts)
            print("[SERVER] Aggregation complete")
        else:
            print("[SERVER] No updates received")

        CURRENT_ROUND += 1
        await asyncio.sleep(ROUND_INTERVAL)


# -----------------------------------------------------
# Startup
# -----------------------------------------------------

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fl_loop())