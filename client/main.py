import os
import torch
import base64
import io
import requests
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

# NEW: Trust value (default 1.0 if not provided)
CLIENT_TRUST = float(os.getenv("CLIENT_TRUST", "1.0"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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
    return torch.load(buffer, map_location="cpu")


# -----------------------------------------------------
# Schema
# -----------------------------------------------------

class TrainCommand(BaseModel):
    round: int
    global_model_bytes: str


# -----------------------------------------------------
# Endpoints
# -----------------------------------------------------

@app.get("/ping")
def ping():
    return {
        "alive": True,
        "client_id": CLIENT_ID,
        "trust": CLIENT_TRUST
    }


@app.post("/train_local")
def train_local(req: TrainCommand):
    model_path = "updated_student.pth"
    print(f"[{CLIENT_ID}] Received global model for round {req.round}")

    # 1️⃣ Build model and load global state
    model = AE_classifier(input_dim=17).to(device)
    global_state = decode_state_dict(req.global_model_bytes)
    model.load_state_dict(global_state)

    round_num = req.round
    print(f"[{CLIENT_ID}] Performing local training for round {round_num}")

    # 2️⃣ Run local adaptation
    student_main(model, round_num, CLIENT_ID)

    print(f"[{CLIENT_ID}] Saved updated local model")

    # 3️⃣ Load updated model from disk
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not created by student_main")

    updated_model = AE_classifier(input_dim=17).to(device)
    updated_model.load_state_dict(torch.load(model_path, map_location=device))

    updated_state = updated_model.state_dict()

    return {
        "status": "trained",
        "model_bytes": encode_state_dict(updated_state),
    }


# -----------------------------------------------------
# Auto Register on Startup
# -----------------------------------------------------

@app.on_event("startup")
async def register_with_server():
    import time
    time.sleep(3)

    print("Registering client with server...")
    print("CLIENT_ID =", CLIENT_ID)
    print("CLIENT_URL =", CLIENT_URL)
    print("SERVER_URL =", SERVER_URL)
    print("CLIENT_TRUST =", CLIENT_TRUST)

    try:
        response = requests.post(
            f"{SERVER_URL}/register_client",
            json={
                "client_id": CLIENT_ID,
                "url": CLIENT_URL,
                "trust": CLIENT_TRUST  # ✅ NEW FIELD
            },
            timeout=5
        )

        if response.status_code == 200:
            print(f"✅ Successfully registered with server as {CLIENT_ID}")
        else:
            print(f"❌ Failed to register: {response.status_code}")

    except Exception as e:
        print(f"❌ Error registering with server: {e}")