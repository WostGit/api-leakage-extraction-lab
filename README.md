# API Leakage Extraction Lab

A lightweight research repo for a first experiment on **model extraction difficulty under different API interfaces**.

## Experiment goal
Given black-box access to a victim next-token model (default: `Qwen/Qwen2-0.5B-Instruct`), measure how well a student model can imitate it under three interface constraints:

1. **argmax**: API returns only the top token id.
2. **topk**: API returns top-k token ids with probabilities (`k=3`).
3. **full**: API returns full next-token logits/probabilities.

For query budgets `[100, 500, 1000]`, we train an interface-specific student and report:
- top-1 agreement with victim
- KL divergence between projected victim/student next-token distributions

## Repo layout

- `run_experiment.py` – main entrypoint
- `models/student.py` – lightweight student model + trainer
- `data/prompts.py` – deterministic synthetic prompt dataset
- `utils/victim.py` – HuggingFace Qwen victim wrapper + API interfaces
- `utils/metrics.py` – agreement + KL metrics
- `utils/io.py` – CSV and plotting helpers
- `results/` – generated CSV + PNG plots

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py
```

### CPU-safe defaults
Defaults are intentionally small, deterministic, and suitable for CPU (slower but feasible). Use fewer epochs or prompts to speed up.

## Notes on KL computation
To keep runtime manageable, KL is computed on a **projected support**:
- all student label tokens + one `OTHER` bucket.
- victim probability mass outside student labels is assigned to `OTHER`.

This preserves comparability across interfaces while avoiding huge-vocabulary student outputs.
