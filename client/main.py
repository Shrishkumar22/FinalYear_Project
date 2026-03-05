import os
import torch
import base64
import io
import httpx
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

from utils import AE_classifier
from ssf_student import main as student_main

app = FastAPI()

# -----------------------------------------------------
# Environment Variables
# -----------------------------------------------------

CLIENT_ID = os.getenv("CLIENT_ID", "client_default")
SERVER_URL = os.getenv("SERVER_URL", "http://fl_server:8000")
CLIENT_URL = os.getenv("CLIENT_URL", f"http://{CLIENT_ID}:9000")
CLIENT_TRUST = float(os.getenv("CLIENT_TRUST", "1.0"))

# -----------------------------------------------------
# FORCE CPU ONLY
# -----------------------------------------------------

device = torch.device("cpu")
print(f"[{CLIENT_ID}] Using CPU")

# -----------------------------------------------------
# Encoding / Decoding Utilities
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
    return torch.load(buffer, map_location=device, weights_only=True)


# -----------------------------------------------------
# Schema
# -----------------------------------------------------

class TrainCommand(BaseModel):
    round: int
    global_model_bytes: str


# -----------------------------------------------------
# Health Endpoint
# -----------------------------------------------------

@app.get("/ping")
def ping():
    return {
        "alive": True,
        "client_id": CLIENT_ID,
        "trust": CLIENT_TRUST
    }


# -----------------------------------------------------
# Training Endpoint
# -----------------------------------------------------

@app.post("/train_local")
async def train_local(req: TrainCommand):

    print(f"\n[{CLIENT_ID}] ===== ROUND {req.round} =====")

    try:
        model_path = f"updated_student_{CLIENT_ID}.pth"

        # Load global model
        model = AE_classifier(input_dim=17).to(device)
        global_state = decode_state_dict(req.global_model_bytes)
        model.load_state_dict(global_state)

        print(f"[{CLIENT_ID}] Starting local training...")

        # Run training
        student_main(model, req.round, CLIENT_ID, device=device)

        print(f"[{CLIENT_ID}] Local training finished")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not created")

        updated_model = AE_classifier(input_dim=17).to(device)
        updated_model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

        updated_state = updated_model.state_dict()

        print(f"[{CLIENT_ID}] Sending update to server")

        return {
            "status": "trained",
            "model_bytes": encode_state_dict(updated_state),
        }

    except Exception as e:
        print(f"[{CLIENT_ID}] TRAINING ERROR:", str(e))
        return {
            "status": "error",
            "message": str(e)
        }


# -----------------------------------------------------
# Auto Register
# -----------------------------------------------------

@app.on_event("startup")
async def register_with_server():
    await asyncio.sleep(5)  # give server more time too
    print(f"\n[{CLIENT_ID}] Registering with server...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{SERVER_URL}/register_client",
                json={
                    "client_id": CLIENT_ID,
                    "url": CLIENT_URL,
                    "trust": CLIENT_TRUST
                }
            )
        print(f"[{CLIENT_ID}] Registration response:", response.status_code)
    except Exception as e:
        print(f"[{CLIENT_ID}] Registration error:", e)