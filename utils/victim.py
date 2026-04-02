from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class VictimOutput:
    top1_id: int
    top3_ids: list[int]
    top3_probs: list[float]
    logits: torch.Tensor


class QwenVictim:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 128) -> None:
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def next_token_logits(self, prompt: str) -> torch.Tensor:
        toks = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        out = self.model(**toks)
        return out.logits[0, -1, :].detach().float().cpu()

    @torch.no_grad()
    def query(self, prompt: str) -> VictimOutput:
        logits = self.next_token_logits(prompt)
        probs = torch.softmax(logits, dim=-1)
        top3_probs, top3_ids = torch.topk(probs, k=3)
        top1_id = int(top3_ids[0].item())
        return VictimOutput(
            top1_id=top1_id,
            top3_ids=[int(i.item()) for i in top3_ids],
            top3_probs=[float(p.item()) for p in top3_probs],
            logits=logits,
        )


def logits_to_selected_probs(logits: torch.Tensor, token_ids: list[int]) -> np.ndarray:
    if not token_ids:
        return np.zeros((0,), dtype=np.float64)
    token_logits = logits[token_ids]
    log_denom = torch.logsumexp(logits, dim=-1)
    log_p = token_logits - log_denom
    return torch.exp(log_p).numpy().astype(np.float64)
