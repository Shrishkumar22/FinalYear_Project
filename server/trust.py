import torch
import numpy as np

class TrustManager:
    def __init__(self, weights, smoothing=0.8):
        self.w = weights
        self.smoothing = smoothing
        self.prev_updates = {}
        self.prev_trust = {}

    def role_score(self, role):
        mapping = {
            "PLC": 0.9,
            "RTU": 0.85,
            "Gateway": 0.7,
            "HMI": 0.65
        }
        return mapping.get(role, 0.7)

    def integrity_score(self, integrity_flag):
        return 1.0 if integrity_flag else 0.0

    def behavior_score(self, update, median_update):
        cos = torch.nn.functional.cosine_similarity(
            update.flatten(),
            median_update.flatten(),
            dim=0
        )
        return 0.5 * (1 + cos.item())

    def stability_score(self, client_id, update):
        if client_id not in self.prev_updates:
            self.prev_updates[client_id] = update
            return 1.0

        prev = self.prev_updates[client_id]
        diff = torch.norm(update - prev)
        base = torch.norm(prev) + 1e-8
        self.prev_updates[client_id] = update
        return max(0.0, 1 - (diff / base).item())

    def compute_trust(self, client_id, role, integrity_flag,
                      criticality, update, median_update):

        R = self.role_score(role)
        P = self.integrity_score(integrity_flag)
        B = self.behavior_score(update, median_update)
        C = criticality
        S = self.stability_score(client_id, update)

        num = (self.w["R"] * R +
               self.w["P"] * P +
               self.w["B"] * B +
               self.w["C"] * C +
               self.w["S"] * S)

        denom = sum(self.w.values())
        T = num / denom

        prev = self.prev_trust.get(client_id, T)
        T_smooth = self.smoothing * prev + (1 - self.smoothing) * T
        self.prev_trust[client_id] = T_smooth

        return T_smooth