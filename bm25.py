"""
bm25.py — Okapi BM25 ranking function, vectorized over sparse term-document matrices.

This is classical information retrieval (the same family of algorithm the
JD says Redrob's *current* production system already runs). It is fit on
the actual candidate text corpus at runtime — there is no pre-baked
vocabulary or skill list. Feed it a different corpus or a different query
and it produces different scores; nothing here is JD-specific.
"""

import numpy as np
import re
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


# A generic English stopword list — not domain-specific.
ENGLISH_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "is", "am", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "it", "its", "they", "them", "their", "this",
    "that", "these", "those", "as", "also",
})


def tokenize(text: str) -> list[str]:
    """Simple, generic tokenizer: lowercase alphanumeric tokens, min length 2.
    Starts with a letter/digit, allows internal hyphen/+/# (for tokens like
    'c++', 'node-js'), never keeps a trailing period."""
    tokens = re.findall(r"[a-z0-9][a-z0-9+#\-]{1,}", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) >= 2]


class BM25Index:
    """
    BM25 index fit over an arbitrary corpus of documents.
    Usage:
        idx = BM25Index(k1=1.5, b=0.75)
        idx.fit(list_of_document_strings)
        scores = idx.score_query("some query text")   # -> np.ndarray, one score per doc
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, max_features: int = 40000):
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.vectorizer: CountVectorizer | None = None
        self.doc_term: sparse.csr_matrix | None = None
        self.doc_len: np.ndarray | None = None
        self.avgdl: float = 0.0
        self.idf: np.ndarray | None = None
        self.n_docs: int = 0

    def fit(self, documents: list[str]):
        self.vectorizer = CountVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,  # tokenizer already lowercases
            max_features=self.max_features,
        )
        self.doc_term = self.vectorizer.fit_transform(documents).tocsr()
        self.n_docs = self.doc_term.shape[0]
        self.doc_len = np.asarray(self.doc_term.sum(axis=1)).ravel().astype(np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 1.0

        df = np.asarray((self.doc_term > 0).sum(axis=0)).ravel().astype(np.float32)
        # BM25 IDF variant (Robertson-Sparck Jones, with +1 floor to avoid negatives)
        self.idf = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
        return self

    def score_query(self, query_text: str) -> np.ndarray:
        """Return a BM25 score for every document in the fitted corpus against query_text."""
        if self.vectorizer is None:
            raise RuntimeError("BM25Index.fit() must be called before score_query()")

        query_tokens = tokenize(query_text)
        vocab = self.vectorizer.vocabulary_
        term_idx = sorted({vocab[t] for t in query_tokens if t in vocab})

        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not term_idx:
            return scores

        sub = self.doc_term[:, term_idx].tocsc()  # (n_docs, n_query_terms), sparse
        norm_len = 1.0 - self.b + self.b * (self.doc_len / self.avgdl)  # (n_docs,)

        for col_pos, vocab_col in enumerate(term_idx):
            col = sub.getcol(col_pos)
            rows = col.indices
            freqs = col.data.astype(np.float32)
            idf_t = self.idf[vocab_col]

            denom = freqs + self.k1 * norm_len[rows]
            contrib = idf_t * (freqs * (self.k1 + 1.0)) / denom
            np.add.at(scores, rows, contrib)

        return scores

    def score_queries(self, query_texts: dict[str, str]) -> dict[str, np.ndarray]:
        """Score multiple named queries at once (e.g. must_have, nice_to_have)."""
        return {name: self.score_query(text) for name, text in query_texts.items()}

    def term_idf_lookup(self) -> dict[str, float]:
        """Return {term: idf} for every term in the fitted vocabulary — used
        to pick out genuinely rare/salient terms for human-readable reasoning,
        as opposed to common connector words that happen to overlap."""
        if self.vectorizer is None:
            return {}
        return {term: float(self.idf[idx]) for term, idx in self.vectorizer.vocabulary_.items()}


# Extended, domain-agnostic filler-word list. These are ordinary English
# words that are simply uncommon in a resume/JD corpus (so raw IDF ranks
# them as "rare"), but they carry no useful signal for citation purposes.
# This list is generic prose vocabulary — it says nothing about AI/ML or any
# specific role, so it doesn't reintroduce hardcoded-domain-keyword bias.
GENERIC_FILLER_WORDS = frozenset({
    "currently", "companies", "company", "tech", "care", "trust", "important",
    "matter", "real", "every", "years", "year", "team", "teams", "role",
    "roles", "work", "works", "working", "worked", "environment",
    "opportunity", "opportunities", "strong", "good", "great", "excellent",
    "ability", "abilities", "skills", "skill", "knowledge", "understanding",
    "understand", "understood", "actually", "probably", "really", "things",
    "thing", "someone", "something", "somewhere", "people", "person",
    "career", "careers", "level", "levels", "kind", "kinds", "sort", "sorts",
    "way", "ways", "lot", "lots", "bit", "bits", "part", "parts", "look",
    "looking", "looked", "find", "found", "want", "wanted", "wants", "need",
    "needed", "needs", "like", "likely", "certain", "specific", "general",
    "particular", "various", "different", "similar", "several", "many",
    "much", "most", "least", "better", "best", "worse", "worst", "long",
    "short", "high", "low", "large", "small", "big", "little", "new", "old",
    "first", "second", "third", "last", "next", "previous", "recent",
    "recently", "today", "tomorrow", "yesterday", "usually", "generally",
    "typically", "essentially", "basically", "simply", "clearly", "however",
    "therefore", "moreover", "furthermore", "meanwhile", "otherwise",
})


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize a score array to [0, 1]. Flat arrays map to 0."""
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
