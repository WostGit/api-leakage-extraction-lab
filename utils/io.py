from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_results_csv(rows: list[dict], out_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for interface in sorted(df["interface"].unique()):
        sub = df[df["interface"] == interface].sort_values("query_budget")
        plt.plot(sub["query_budget"], sub[metric], marker="o", label=interface)
    plt.xlabel("Query budget")
    plt.ylabel(metric)
    plt.title(f"{metric} vs query budget")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
