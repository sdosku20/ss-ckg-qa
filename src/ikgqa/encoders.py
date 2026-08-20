"""
ikgqa.encoders
==============

Text -> vector. Everything downstream (prizes, top-k, PCST) only ever sees
embeddings, so this is the single place where "what does similar mean" is
decided.

Two implementations:

  * BagOfWordsEncoder -- deterministic, dependency-free, hand-checkable.
    Cosine similarity is literally "how many words does this share with the
    question". Use it for development, tests and teaching: no model download,
    identical output on every machine and every run.

  * SentenceTransformerEncoder -- the real thing (SentenceBERT, as in
    G-Retriever Section 5.1). Loaded lazily so importing this module never
    pulls in torch.

Both return L2-normalised float32 rows, so a dot product *is* the cosine
similarity. Both satisfy the Encoder protocol, so swapping one for the other is
a one-line change in an experiment config.

A caution that matters for the clinical setting (see thesis Section 1.3): a
sentence encoder trained on open-domain text handles "kidney transplant" well
and handles "SCTID:709044004" or "2160-0" not at all. If node text is mostly
codes, similarity is close to noise and every retriever built on it inherits
that. Prefer building node text as "<type>: <label> (<code>)" so there is real
language for the encoder to work with, and check the prize tables before
trusting any retrieval result.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens. Shared by the encoder and its tests."""
    return _TOKEN.findall(str(text).lower())


def l2_normalise(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale each row to unit length, guarding the all-zero row."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, eps)


@runtime_checkable
class Encoder(Protocol):
    """Anything that turns a list of strings into a [n, d] float32 matrix."""

    @property
    def dim(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class BagOfWordsEncoder:
    """L2-normalised term-count vectors over a fixed vocabulary.

    Deterministic and tiny. Because every dimension is a known word, you can
    verify by hand why one node outranked another -- which is the whole point
    when debugging a retriever.

    The vocabulary is fixed at construction: words unseen during fit are
    ignored rather than growing the space, so embeddings stay comparable across
    calls (a growing vocabulary would silently change earlier vectors' meaning).

    Args:
        corpus: the texts whose words define the vocabulary. Pass node texts +
            edge texts + (if you have them) the questions.
    """

    def __init__(self, corpus: Iterable[str]):
        vocab: dict = {}
        for text in corpus:
            for tok in tokenize(text):
                vocab.setdefault(tok, len(vocab))
        self.vocab = vocab

    @property
    def dim(self) -> int:
        return max(len(self.vocab), 1)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                idx = self.vocab.get(tok)
                if idx is not None:
                    out[row, idx] += 1.0
        return l2_normalise(out)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"BagOfWordsEncoder(vocab={len(self.vocab)})"


class SentenceTransformerEncoder:
    """SentenceBERT via sentence-transformers, loaded on first use.

    Args:
        model_name: any sentence-transformers id. G-Retriever's released code
            uses "sentence-transformers/all-roberta-large-v1"; the paper cites
            SentenceBERT generally.
        batch_size: encoding batch size.
        device: passed to sentence-transformers ("cuda", "cpu", or None to let
            it choose).

    The model is a process-level singleton per (model_name, device): loading
    a large encoder twice wastes a GB of RAM for no benefit, and an experiment
    sweep constructs encoders freely.
    """

    _cache: dict = {}

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-roberta-large-v1",
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self._model = None
        self._dim: Optional[int] = None

    def _load(self):
        if self._model is None:
            key = (self.model_name, self.device)
            if key not in type(self)._cache:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:  # pragma: no cover - env dependent
                    raise ImportError(
                        "sentence-transformers is not installed, so real embeddings are "
                        "unavailable.\n  pip install sentence-transformers\n"
                        "For offline development use BagOfWordsEncoder instead -- every "
                        "retriever and metric in this package works with it."
                    ) from e
                type(self)._cache[key] = SentenceTransformer(self.model_name, device=self.device)
            self._model = type(self)._cache[key]
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._load().get_sentence_embedding_dimension())
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SentenceTransformerEncoder({self.model_name!r})"
