from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class StudentNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class StudentConfig:
    input_dim: int = 1024
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 7


class StudentTrainer:
    def __init__(self, num_classes: int, cfg: StudentConfig, device: str = "cpu") -> None:
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.device = device
        self.model = StudentNet(cfg.input_dim, num_classes).to(device)

    def fit(self, x: np.ndarray, target_probs: np.ndarray) -> None:
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(target_probs, dtype=torch.float32)
        ds = TensorDataset(x_t, y_t)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        loss_fn = nn.KLDivLoss(reduction="batchmean")

        self.model.train()
        for _ in range(self.cfg.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = loss_fn(log_probs, yb)
                loss.backward()
                opt.step()

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        logits = self.model(x_t)
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()
