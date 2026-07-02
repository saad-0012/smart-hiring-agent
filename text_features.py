"""
text_features.py — TF-IDF cosine similarity layer + candidate document builder.

This is the second half of the "hybrid retrieval" system (BM25 = sparse exact-term
matching in bm25.py; TF-IDF cosine here captures softer, corpus-relative term
importance — closer to how a real semantic-fit signal behaves without needing
a downloaded neural embedding model).

Fit at runtime on the actual candidate corpus + JD section texts. No hardcoded
vocabulary.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from bm25 import tokenize


def build_candidate_document(candidate: dict) -> str:
    """
    Concatenate everything textual and meaningful about a candidate into one
    document string: headline, summary, every career description, skill names,
    certification names. This is the same document used for both BM25 and
    TF-IDF — no separate keyword extraction step.
    """
    p = candidate.get("profile", {}) or {}
    parts = [
        p.get("headline") or "",
        p.get("summary") or "",
        p.get("current_title") or "",
    ]
    for job in candidate.get("career_history", []) or []:
        parts.append(job.get("title") or "")
        parts.append(job.get("description") or "")
    for skill in candidate.get("skills", []) or []:
        # repeat skill name proportional to proficiency so TF-IDF/BM25 naturally
        # weight higher-proficiency skills more, without a hardcoded weight table
        name = skill.get("name") or ""
        prof = (skill.get("proficiency") or "").lower()
        repeat = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 1}.get(prof, 1)
        parts.extend([name] * repeat)
    for cert in candidate.get("certifications", []) or []:
        parts.append(cert.get("name") or "")

    return " ".join(p for p in parts if p)


class TfidfSimilarityIndex:
    """
    Fits a TF-IDF vectorizer on the candidate corpus, then scores arbitrary
    query texts (JD sections) against every candidate via cosine similarity.
    """

    def __init__(self, max_features: int = 40000):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            max_features=max_features,
        )
        self.doc_matrix = None  # (n_docs, vocab), L2-normalized rows

    def fit(self, documents: list[str]):
        matrix = self.vectorizer.fit_transform(documents)
        self.doc_matrix = sk_normalize(matrix, norm="l2", axis=1)
        return self

    def score_query(self, query_text: str) -> np.ndarray:
        if self.doc_matrix is None:
            raise RuntimeError("TfidfSimilarityIndex.fit() must be called first")
        q_vec = self.vectorizer.transform([query_text])
        q_vec = sk_normalize(q_vec, norm="l2", axis=1)
        sims = self.doc_matrix.dot(q_vec.T).toarray().ravel()
        return sims

    def score_queries(self, query_texts: dict[str, str]) -> dict[str, np.ndarray]:
        return {name: self.score_query(text) for name, text in query_texts.items()}
