"""Few-shot example selection from the train split."""

import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.inference.prompts import Example


class FewShotSelector:
    def __init__(self, train: list[dict], strategy: str, k: int, seed: int) -> None:
        self.train = train
        self.strategy = strategy
        self.k = k
        self.seed = seed
        if strategy == "similar":
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self.train_matrix = self.vectorizer.fit_transform([s["question"] for s in train])
        elif strategy != "random":
            raise ValueError(f"unknown few-shot strategy: {strategy}")

    def select(self, sample: dict) -> list[Example]:
        if self.k == 0:
            return []
        if self.strategy == "random":
            # seeded per sample key: identical for all models, independent of order/resume
            picked = random.Random(f"{self.seed}-{sample['key']}").sample(self.train, self.k)
        else:
            scores = linear_kernel(self.vectorizer.transform([sample["question"]]), self.train_matrix)[0]
            # stable sort: ties resolved by train index; most similar example placed last
            top = np.argsort(-scores, kind="stable")[: self.k]
            picked = [self.train[i] for i in reversed(top)]
        return [Example(s["question"], s["sql_norm"]) for s in picked]
