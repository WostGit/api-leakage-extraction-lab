from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptDataset:
    train_prompts: list[str]
    eval_prompts: list[str]


def _random_word(rng: random.Random) -> str:
    syllables = [
        "al", "be", "cor", "den", "el", "fi", "gan", "hel", "ion", "jor",
        "kel", "lin", "mor", "nel", "or", "pra", "qui", "ron", "sin", "tor",
        "ul", "ven", "wen", "xel", "yor", "zen",
    ]
    pieces = [rng.choice(syllables) for _ in range(rng.randint(1, 3))]
    return "".join(pieces)


def build_prompt_pool(total: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    templates = [
        "Write a one-line summary about {}.",
        "Complete the sentence: {} is",
        "Give a short definition of {}.",
        "The opposite of {} is",
        "Name one fact about {}.",
        "Translate to plain English: {}.",
        "What comes next in this pattern: {}",
        "Classify the topic of: {}",
    ]

    prompts: list[str] = []
    for _ in range(total):
        token_count = rng.randint(2, 6)
        phrase = " ".join(_random_word(rng) for _ in range(token_count))
        prompt = rng.choice(templates).format(phrase)
        prompts.append(prompt)
    return prompts


def make_dataset(train_size: int, eval_size: int, seed: int = 7) -> PromptDataset:
    pool = build_prompt_pool(train_size + eval_size, seed=seed)
    return PromptDataset(train_prompts=pool[:train_size], eval_prompts=pool[train_size:])
