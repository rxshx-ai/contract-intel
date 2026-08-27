"""Shared lexical scoring. Okapi BM25, no dependencies, deterministic.

Used by both the record index (api/ask.py) and the passage index (api/rag.py)
so the two halves of retrieval rank on the same footing and can be merged.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_WORD = re.compile(r"[a-z0-9$%.]+")
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "is", "are",
    "we", "our", "us", "i", "do", "does", "did", "what", "which", "how", "when",
    "who", "whom", "any", "all", "with", "that", "this", "it", "be", "by", "at",
    "from", "as", "if", "can", "have", "has", "there", "their", "they", "shall",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower())
            if t not in STOPWORDS and len(t) > 1]


class BM25:
    K1 = 1.5
    B = 0.75

    def __init__(self, corpus: Sequence[str]):
        self.docs = [tokenize(text) for text in corpus]
        self.lengths = [len(d) for d in self.docs]
        self.n = len(self.docs)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.freqs = [Counter(d) for d in self.docs]
        self.df: Counter[str] = Counter()
        for doc in self.docs:
            self.df.update(set(doc))

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, i: int, terms: Sequence[str]) -> float:
        total = 0.0
        freq, length = self.freqs[i], self.lengths[i] or 1
        for term in terms:
            f = freq.get(term, 0)
            if not f:
                continue
            denom = f + self.K1 * (1 - self.B + self.B * length / (self.avg_len or 1))
            total += self.idf(term) * (f * (self.K1 + 1)) / denom
        return total

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        return [self.score(i, terms) for i in range(self.n)]
