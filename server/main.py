import base64
import io
import asyncio
import threading
import time
import httpx
import torch

from fastapi import FastAPI
from pydantic import BaseModel

from utils import AE_classifier

EXPECTED_CLIENTS = 2
TRAIN_TIMEOUT = 1200
ROUND_INTERVAL = 60

device = torch.device("cpu")
print("Using CPU server")

app = FastAPI()

# -----------------------------------------------------
# Globals (no asyncio.Lock at module level!)
# -----------------------------------------------------

GLOBAL_MODEL = None
CURRENT_ROUND = 1
REGISTERED_CLIENTS = []
CLIENTS_LOCK = threading.Lock()  # use threading.Lock, NOT asyncio.Lock


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
    return torch.load(buffer, map_location="cpu", weights_only=False)


# -----------------------------------------------------
# Load Initial Model
# -----------------------------------------------------

def load_model(path):
    model = AE_classifier(input_dim=17).to(device)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=False)
    )
    model.eval()
    return model


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
    # Use a simple list append - GIL protects this, no lock needed in async context
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
        shape = [len(trusts)] + [1] * (stacked.dim() - 1)
        weighted = stacked * trust_tensor.view(*shape)
        agg[key] = weighted.sum(0)

    GLOBAL_MODEL = agg


# -----------------------------------------------------
# Contact Client (sync version using httpx)
# -----------------------------------------------------

def contact_client_sync(client, round_number, global_model_b64):
    try:
        print(f"[SERVER] Contacting {client['id']}...")
        with httpx.Client(timeout=TRAIN_TIMEOUT) as http_client:
            resp = http_client.post(
                f"{client['url']}/train_local",
                json={
                    "round": round_number,
                    "global_model_bytes": global_model_b64,
                }
            )
        print(f"[SERVER] Got response from {client['id']}, status={resp.status_code}, size={len(resp.content)} bytes")

        if resp.status_code != 200:
            print(f"[SERVER] {client['id']} returned HTTP {resp.status_code}")
            return None

        data = resp.json()
        print(f"[SERVER] Parsed JSON from {client['id']}, status={data.get('status')}")
  

        if data.get("status") == "error":
            print(f"[SERVER] Client error: {data.get('message')}")
            return None

        update = decode_state_dict(data["model_bytes"])
        print(f"[SERVER] Update received from {client['id']}")
        return update, client["trust"]

    except Exception as e:
        print(f"[SERVER] ERROR contacting {client['id']}: {e}")
        return None


# -----------------------------------------------------
# Federated Loop (background thread, fully synchronous)
# -----------------------------------------------------

def fl_loop_thread():
    global CURRENT_ROUND

    print("[SERVER] fl_thread started, waiting 20s...")
    time.sleep(20)
    print("[SERVER] fl_thread woke up!")

    while True:
        try:
            with CLIENTS_LOCK:
                clients_snapshot = REGISTERED_CLIENTS.copy()

            print(f"[SERVER] Client count: {len(clients_snapshot)}/{EXPECTED_CLIENTS}")

            if len(clients_snapshot) < EXPECTED_CLIENTS:
                print("[SERVER] Waiting for clients...")
                time.sleep(5)
                continue

            print(f"\n===== ROUND {CURRENT_ROUND} =====")
            round_start = time.time()

            global_b64 = encode_state_dict(GLOBAL_MODEL)

            results = []
            threads = []

            def call_client(c):
                r = contact_client_sync(c, CURRENT_ROUND, global_b64)
                results.append(r)

            for c in clients_snapshot:
                t = threading.Thread(target=call_client, args=(c,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=TRAIN_TIMEOUT)

            updates = []
            trusts = []
            for r in results:
                if r is not None:
                    updates.append(r[0])
                    trusts.append(r[1])

            if updates:
                aggregate(updates, trusts)
                elapsed = time.time() - round_start
                print(f"[SERVER] Aggregation complete. Round took {elapsed:.1f}s")
            else:
                print("[SERVER] No updates received this round")

            CURRENT_ROUND += 1
            # Wait a short gap AFTER clients finish, not a fixed timer
            time.sleep(10)  # just 10s cooldown between rounds

        except Exception as e:
            import traceback
            print(f"[SERVER] fl_thread ERROR: {e}")
            traceback.print_exc()
            time.sleep(5)

# -----------------------------------------------------
# Startup
# -----------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global GLOBAL_MODEL

    print("[SERVER] Loading model...")
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(None, load_model, "teacher_optimized.pth")
    GLOBAL_MODEL = {k: v.clone().detach() for k, v in model.state_dict().items()}
    print("[SERVER] Model loaded!")

    t = threading.Thread(target=fl_loop_thread, daemon=True)
    t.start()
    print("[SERVER] fl_thread launched!")