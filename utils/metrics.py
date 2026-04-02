from __future__ import annotations

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def top1_agreement(student_token_ids: list[int | None], victim_top1_ids: list[int]) -> float:
    correct = 0
    for s, v in zip(student_token_ids, victim_top1_ids, strict=True):
        if s is not None and s == v:
            correct += 1
    return correct / len(victim_top1_ids)
