"""Semantic filtering engine for MCP tool catalogs.

Provider-agnostic top-k tool selection. Ships a zero-dependency TF-IDF cosine
backend (fast, no downloads, no AWS) and a pluggable interface so a real
embedding model (MiniLM, OpenAI, etc.) can be swapped in without touching the
proxy. This is the local, self-owned answer to mcp-token-saver's Bedrock lock.
"""
from __future__ import annotations
import math
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())


def tool_signal(tool: dict) -> str:
    """The text a router 'sees' for a tool: its name + description."""
    return f"{tool.get('name', '')} {tool.get('description', '')}"


class Vectorizer(Protocol):
    """Swap this to upgrade from TF-IDF to real embeddings."""

    def fit(self, docs: list[str]) -> np.ndarray: ...
    def transform(self, query: str) -> np.ndarray: ...


class TfidfVectorizer:
    """Pure-numpy TF-IDF with L2-normalised rows for cosine via dot product."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    def fit(self, docs: list[str]) -> np.ndarray:
        toks = [_tokenize(d) for d in docs]
        df: dict[str, int] = {}
        for tl in toks:
            for w in set(tl):
                df[w] = df.get(w, 0) + 1
        self.vocab = {w: i for i, w in enumerate(df)}
        n = len(docs)
        self.idf = np.zeros(len(self.vocab))
        for w, i in self.vocab.items():
            self.idf[i] = math.log((n + 1) / (df[w] + 1)) + 1
        mat = np.zeros((n, len(self.vocab)))
        for r, tl in enumerate(toks):
            for w in tl:
                mat[r, self.vocab[w]] += 1
            mat[r] *= self.idf
            norm = np.linalg.norm(mat[r])
            if norm > 0:
                mat[r] /= norm
        return mat

    def transform(self, query: str) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for w in _tokenize(query):
            i = self.vocab.get(w)
            if i is not None:
                v[i] += 1
        v *= self.idf
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v


class ToolFilter:
    """Indexes a catalog once, then routes queries to a top-k subset."""

    def __init__(
        self,
        tools: list[dict],
        vectorizer: Vectorizer | None = None,
        max_tools: int = 15,
        min_similarity: float = 0.05,
    ) -> None:
        self.tools = [t for t in tools if t.get("name")]
        self.by_name = {t["name"]: t for t in self.tools}
        self.names = [t["name"] for t in self.tools]
        self.max_tools = max_tools
        self.min_similarity = min_similarity
        self.vectorizer = vectorizer or TfidfVectorizer()
        self._matrix = self.vectorizer.fit([tool_signal(t) for t in self.tools])

    def route(self, query: str, extra_terms: str = "") -> list[dict]:
        """Return the tools most relevant to a query.

        extra_terms lets the proxy inject session context (recent tool names,
        a task hint) since a stdio proxy never sees the user's raw prompt.
        Falls back to the full catalog only when nothing scores — a safety net,
        never silent tool loss.
        """
        if len(self.tools) <= self.max_tools:
            return self.tools
        qv = self.vectorizer.transform(f"{query} {extra_terms}".strip())
        scores = self._matrix @ qv
        order = np.argsort(-scores)
        kept = [self.by_name[self.names[i]] for i in order[:self.max_tools]
                if scores[i] > self.min_similarity]
        return kept or self.tools  # never return an empty toolset

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Ranked (name, score) pairs — powers the progressive-disclosure tool."""
        qv = self.vectorizer.transform(query)
        scores = self._matrix @ qv
        order = np.argsort(-scores)
        return [(self.names[i], float(scores[i])) for i in order[:limit]
                if scores[i] > 0]
