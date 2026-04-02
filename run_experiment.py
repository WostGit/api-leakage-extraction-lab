from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.prompts import make_dataset
from models.student import StudentConfig, StudentTrainer
from utils.io import plot_metric, save_results_csv
from utils.metrics import kl_divergence, top1_agreement
from utils.victim import QwenVictim, logits_to_selected_probs

INTERFACES = ("argmax", "topk", "full")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hash_features(text: str, dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def build_target_distribution(
    interface: str,
    label_to_index: dict[int, int],
    other_idx: int,
    top1_id: int,
    top3_ids: list[int],
    top3_probs: list[float],
    logits: torch.Tensor,
) -> np.ndarray:
    y = np.zeros((len(label_to_index) + 1,), dtype=np.float64)

    if interface == "argmax":
        idx = label_to_index.get(top1_id, other_idx)
        y[idx] = 1.0
        return y

    if interface == "topk":
        mass = 0.0
        for tid, p in zip(top3_ids, top3_probs, strict=True):
            idx = label_to_index.get(tid)
            if idx is not None:
                y[idx] += p
                mass += p
        y[other_idx] = max(0.0, 1.0 - mass)
        if y.sum() == 0:
            y[other_idx] = 1.0
        else:
            y /= y.sum()
        return y

    # full
    selected = list(label_to_index.keys())
    probs = logits_to_selected_probs(logits, selected)
    mass = float(probs.sum())
    for tid, p in zip(selected, probs, strict=True):
        y[label_to_index[tid]] = p
    y[other_idx] = max(0.0, 1.0 - mass)
    if y.sum() == 0:
        y[other_idx] = 1.0
    else:
        y /= y.sum()
    return y


def student_top1_token(student_dist: np.ndarray, index_to_label: dict[int, int], other_idx: int) -> int | None:
    idx = int(np.argmax(student_dist))
    if idx == other_idx:
        return None
    return index_to_label[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--query-budgets", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    set_seed(args.seed)
    dataset = make_dataset(args.train_size, args.eval_size, seed=args.seed)
    victim = QwenVictim(model_name=args.model_name, device=args.device)

    print("Precomputing victim outputs...")
    all_prompts = dataset.train_prompts + dataset.eval_prompts
    victim_outputs = {}
    for prompt in tqdm(all_prompts):
        victim_outputs[prompt] = victim.query(prompt)

    rows: list[dict] = []
    max_budget = max(args.query_budgets)

    for interface in INTERFACES:
        base_prompts = dataset.train_prompts[:max_budget]
        label_tokens: set[int] = set()
        for p in base_prompts:
            out = victim_outputs[p]
            if interface == "argmax":
                label_tokens.add(out.top1_id)
            else:
                label_tokens.update(out.top3_ids)

        sorted_labels = sorted(label_tokens)
        label_to_index = {tid: i for i, tid in enumerate(sorted_labels)}
        index_to_label = {i: tid for tid, i in label_to_index.items()}
        other_idx = len(sorted_labels)

        for budget in args.query_budgets:
            train_prompts = dataset.train_prompts[:budget]
            x_train = np.stack([hash_features(p, args.feature_dim) for p in train_prompts], axis=0)
            y_train = []
            for p in train_prompts:
                out = victim_outputs[p]
                y = build_target_distribution(
                    interface=interface,
                    label_to_index=label_to_index,
                    other_idx=other_idx,
                    top1_id=out.top1_id,
                    top3_ids=out.top3_ids,
                    top3_probs=out.top3_probs,
                    logits=out.logits,
                )
                y_train.append(y)
            y_train_arr = np.stack(y_train, axis=0)

            trainer = StudentTrainer(
                num_classes=len(sorted_labels) + 1,
                cfg=StudentConfig(input_dim=args.feature_dim, epochs=args.epochs, seed=args.seed),
                device=args.device,
            )
            trainer.fit(x_train, y_train_arr)

            x_eval = np.stack([hash_features(p, args.feature_dim) for p in dataset.eval_prompts], axis=0)
            student_probs = trainer.predict_proba(x_eval)

            student_top1_ids: list[int | None] = []
            victim_top1_ids: list[int] = []
            kls: list[float] = []

            for i, p in enumerate(dataset.eval_prompts):
                out = victim_outputs[p]
                v_dist = build_target_distribution(
                    interface="full",
                    label_to_index=label_to_index,
                    other_idx=other_idx,
                    top1_id=out.top1_id,
                    top3_ids=out.top3_ids,
                    top3_probs=out.top3_probs,
                    logits=out.logits,
                )
                s_dist = student_probs[i].astype(np.float64)
                s_dist /= s_dist.sum()
                kls.append(kl_divergence(v_dist, s_dist))
                student_top1_ids.append(student_top1_token(s_dist, index_to_label, other_idx))
                victim_top1_ids.append(out.top1_id)

            agreement = top1_agreement(student_top1_ids, victim_top1_ids)
            mean_kl = float(np.mean(kls))

            row = {
                "interface": interface,
                "query_budget": budget,
                "top1_agreement": agreement,
                "kl_divergence": mean_kl,
                "num_labels": len(sorted_labels),
            }
            rows.append(row)
            print(row)

    out_csv = args.out_dir / "experiment_results.csv"
    df = save_results_csv(rows, out_csv)
    plot_metric(df, "top1_agreement", args.out_dir / "agreement_vs_queries.png")
    plot_metric(df, "kl_divergence", args.out_dir / "kl_vs_queries.png")
    print(f"Saved results to: {out_csv}")


if __name__ == "__main__":
    main()
