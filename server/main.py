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
ROUND_INTERVAL = 10

device = torch.device("cpu")
print("Using CPU server", flush=True)

app = FastAPI()

# -----------------------------------------------------
# Globals
# -----------------------------------------------------

GLOBAL_MODEL = None
CURRENT_ROUND = 1
REGISTERED_CLIENTS = []
CLIENTS_LOCK = threading.Lock()


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


def model_norm(state_dict):
    """Helper to compute total norm of model weights - used to verify aggregation changes the model."""
    total = 0.0
    for v in state_dict.values():
        total += v.float().norm().item()
    return round(total, 4)


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
    for c in REGISTERED_CLIENTS:
        if c["id"] == req.client_id:
            return {"status": "already_registered"}

    REGISTERED_CLIENTS.append({
        "id": req.client_id,
        "url": req.url,
        "trust": float(req.trust)
    })

    print(f"[SERVER] Registered {req.client_id}", flush=True)
    return {"status": "registered"}


# -----------------------------------------------------
# Aggregation
# -----------------------------------------------------

def aggregate(updates, trusts):
    global GLOBAL_MODEL

    norm_before = model_norm(GLOBAL_MODEL)

    trust_tensor = torch.tensor(trusts)
    trust_tensor = trust_tensor / trust_tensor.sum()

    agg = {}
    for key in GLOBAL_MODEL.keys():
        stacked = torch.stack([u[key].float() for u in updates])
        shape = [len(trusts)] + [1] * (stacked.dim() - 1)
        weighted = stacked * trust_tensor.view(*shape)
        agg[key] = weighted.sum(0)

    GLOBAL_MODEL = agg
    norm_after = model_norm(GLOBAL_MODEL)

    print(f"[SERVER] Aggregation done | model norm: {norm_before} → {norm_after} | delta: {round(norm_after - norm_before, 4)}", flush=True)


# -----------------------------------------------------
# Contact Client (sync)
# -----------------------------------------------------

def contact_client_sync(client, round_number, global_model_b64):
    try:
        print(f"[SERVER] Contacting {client['id']}...", flush=True)
        with httpx.Client(timeout=TRAIN_TIMEOUT) as http_client:
            resp = http_client.post(
                f"{client['url']}/train_local",
                json={
                    "round": round_number,
                    "global_model_bytes": global_model_b64,
                }
            )
        print(f"[SERVER] Response from {client['id']}: status={resp.status_code} size={len(resp.content)}bytes", flush=True)

        if resp.status_code != 200:
            print(f"[SERVER] {client['id']} returned HTTP {resp.status_code}", flush=True)
            return None

        data = resp.json()
        print(f"[SERVER] Parsed JSON from {client['id']}: status={data.get('status')}", flush=True)

        if data.get("status") == "error":
            print(f"[SERVER] Client error from {client['id']}: {data.get('message')}", flush=True)
            return None

        update = decode_state_dict(data["model_bytes"])
        print(f"[SERVER] Model update received from {client['id']} | norm={model_norm(update)}", flush=True)
        return update, client["trust"]

    except Exception as e:
        print(f"[SERVER] ERROR contacting {client['id']}: {e}", flush=True)
        return None


# -----------------------------------------------------
# Federated Loop
# -----------------------------------------------------

def fl_loop_thread():
    global CURRENT_ROUND

    print("[SERVER] fl_thread started, waiting 20s...", flush=True)
    time.sleep(20)
    print("[SERVER] fl_thread woke up!", flush=True)

    while True:
        try:
            with CLIENTS_LOCK:
                clients_snapshot = REGISTERED_CLIENTS.copy()

            print(f"[SERVER] Client count: {len(clients_snapshot)}/{EXPECTED_CLIENTS}", flush=True)

            if len(clients_snapshot) < EXPECTED_CLIENTS:
                print("[SERVER] Waiting for clients...", flush=True)
                time.sleep(5)
                continue

            print(f"\n[SERVER] ===== ROUND {CURRENT_ROUND} =====", flush=True)
            print(f"[SERVER] Global model norm before round: {model_norm(GLOBAL_MODEL)}", flush=True)
            round_start = time.time()

            global_b64 = encode_state_dict(GLOBAL_MODEL)

            # Contact clients in parallel
            results = []
            results_lock = threading.Lock()

            def call_client(c):
                r = contact_client_sync(c, CURRENT_ROUND, global_b64)
                with results_lock:
                    results.append(r)

            threads = []
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

            print(f"[SERVER] Received {len(updates)}/{len(clients_snapshot)} updates", flush=True)

            if updates:
                aggregate(updates, trusts)
                elapsed = time.time() - round_start
                print(f"[SERVER] Round {CURRENT_ROUND} complete in {elapsed:.1f}s", flush=True)
            else:
                print("[SERVER] No updates received this round", flush=True)

            CURRENT_ROUND += 1
            print(f"[SERVER] Sleeping {ROUND_INTERVAL}s before next round...", flush=True)
            time.sleep(ROUND_INTERVAL)

        except Exception as e:
            import traceback
            print(f"[SERVER] fl_thread ERROR: {e}", flush=True)
            traceback.print_exc()
            time.sleep(5)


# -----------------------------------------------------
# Startup
# -----------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global GLOBAL_MODEL

    print("[SERVER] Loading model...", flush=True)
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(None, load_model, "teacher_optimized.pth")
    GLOBAL_MODEL = {k: v.clone().detach() for k, v in model.state_dict().items()}
    print(f"[SERVER] Model loaded! Initial norm: {model_norm(GLOBAL_MODEL)}", flush=True)

    t = threading.Thread(target=fl_loop_thread, daemon=True)
    t.start()
    print("[SERVER] fl_thread launched!", flush=True)
