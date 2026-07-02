"""
<<<<<<< HEAD
Redrob Ranker — Streamlit Sandbox Demo (v2)
Team: Bug Writer

Upload any JD (.docx) and a small candidate sample (.json/.jsonl) and watch
the system parse the JD generically, fit BM25 + TF-IDF at runtime, and
produce a ranked shortlist with grounded reasoning.
=======
Redrob Ranker — Streamlit Sandbox Demo
Team: Bug Writer

Lets organizers upload a sample candidates JSON/JSONL and run the ranking system end-to-end.
>>>>>>> 16501df389063dc484603aad8e618abe66de8156
"""

import streamlit as st
import json
import csv
import io
import time
<<<<<<< HEAD
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rank as ranker
from jd_parser import parse_jd_docx
from bm25 import BM25Index, normalize_scores
from text_features import TfidfSimilarityIndex, build_candidate_document
import numpy as np

st.set_page_config(page_title="Redrob Ranker — Bug Writer", page_icon="🤖", layout="wide")

st.title("🤖 Redrob AI Candidate Ranker")
st.caption("Team: Bug Writer | Redrob Hackathon 2026 — JD-driven hybrid IR")

with st.expander("📐 Architecture — why this isn't keyword matching", expanded=False):
    st.markdown("""
    **The JD is parsed generically at runtime** — no company names, skill lists,
    or role-specific words are hardcoded anywhere in this codebase.

    1. **`jd_parser.py`** reads the uploaded `.docx` using its heading STRUCTURE
       (Heading 1/2 styles) and generic regex patterns (`"X-Y years"`,
       `"sub-N-day"`, parenthetical company lists) — not this-JD-specific wording.
    2. **`bm25.py` + `text_features.py`** fit BM25 and TF-IDF indices on the
       *actual candidate corpus* at runtime and score every candidate against
       the JD's own must-have / nice-to-have / exclude / ideal-candidate text.
    3. **Structured pillars** (behavioral signals, experience quality) use only
       numeric/structured fields — never free text — and pull their thresholds
       (YOE range, notice-period ceiling) from the *parsed JD*, not a constant.
    4. **Honeypot detection** is pure data-integrity checking.

    Swap in a completely different JD and this system produces a different
    ranking without touching the code.

    **Known limitation:** bag-of-words matching (BM25/TF-IDF) can't detect
    negation, so a term appearing in the JD's "what we don't want" section can
    occasionally get flagged even when the candidate's usage of that term is
    actually a positive (e.g. "open-source" cited as missing from a
    disqualified profile, but present as a genuine strength in another). This
    is a known tradeoff of lexical IR vs. full semantic understanding.
=======
import sys
from pathlib import Path

# Import the ranker
sys.path.insert(0, str(Path(__file__).parent))
import rank as ranker

st.set_page_config(
    page_title="Redrob Ranker — Bug Writer",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Redrob AI Candidate Ranker")
st.caption("Team: Bug Writer | Redrob Hackathon 2026")

with st.expander("📐 Architecture Overview", expanded=False):
    st.markdown("""
    **Multi-signal scoring pipeline — 4 pillars:**

    | Pillar | Weight | What it measures |
    |--------|--------|-----------------|
    | Role Fit | 35% | Career trajectory, domain depth, industry type |
    | Skill Match | 30% | Technical skills vs JD (must-have vs nice-to-have) |
    | Behavioral Signals | 20% | Recency, response rate, notice period, location |
    | Experience Quality | 15% | YOE band, product vs services ratio, tenure stability |

    **Honeypot detection**: timeline impossibilities + 0-duration expert skill anomalies.

    **No external API calls. CPU only. ~2 min for 100K candidates.**
>>>>>>> 16501df389063dc484603aad8e618abe66de8156
    """)

st.divider()

<<<<<<< HEAD
col1, col2 = st.columns(2)
with col1:
    jd_file = st.file_uploader("Upload Job Description (.docx)", type=["docx"])
with col2:
    cand_file = st.file_uploader("Upload candidates (.json or .jsonl)", type=["json", "jsonl"])

top_n = st.slider("Number of top candidates to rank", min_value=5, max_value=50, value=10)

if jd_file is not None and cand_file is not None:
    if st.button("🚀 Run Ranking", type="primary"):
        with st.spinner("Parsing JD..."):
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(jd_file.read())
                jd_path = tmp.name
            jd = parse_jd_docx(jd_path)

        st.success("JD parsed")
        with st.expander("Parsed JD structure"):
            st.text(jd.summary())

        with st.spinner("Loading candidates..."):
            content = cand_file.read().decode("utf-8")
            candidates = []
            try:
                data = json.loads(content)
                candidates = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
=======
st.subheader("📁 Upload Candidates")
st.info(
    "Upload a `.json` (array) or `.jsonl` (newline-delimited) file with candidate records. "
    "For this sandbox, use the `sample_candidates.json` from the hackathon bundle (50 candidates)."
)

uploaded_file = st.file_uploader(
    "Upload candidates file",
    type=["json", "jsonl"],
    help="sample_candidates.json from the hackathon bundle works perfectly here.",
)

top_n = st.slider("Number of top candidates to rank", min_value=5, max_value=50, value=10)

if uploaded_file is not None:
    if st.button("🚀 Run Ranking", type="primary"):
        with st.spinner("Loading candidates..."):
            content = uploaded_file.read().decode("utf-8")

            candidates = []
            # Try JSON array first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    candidates = data
                elif isinstance(data, dict):
                    candidates = [data]
            except json.JSONDecodeError:
                # Try JSONL
>>>>>>> 16501df389063dc484603aad8e618abe66de8156
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            candidates.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        if not candidates:
            st.error("Could not parse any candidates from the uploaded file.")
        else:
            st.success(f"Loaded {len(candidates)} candidates")

<<<<<<< HEAD
            progress = st.progress(0, text="Building documents + fitting indices...")
            documents = [build_candidate_document(c) for c in candidates]

            bm25 = BM25Index().fit(documents)
            tfidf = TfidfSimilarityIndex().fit(documents)

            queries = {
                "must": jd.get_section("must_have") or jd.raw_text,
                "nice": jd.get_section("nice_to_have"),
                "exclude": jd.get_section("exclude"),
                "ideal": jd.get_section("ideal_profile") or jd.get_section("general"),
            }
            bm25_raw = bm25.score_queries(queries)
            tfidf_raw = tfidf.score_queries(queries)

            combined = {}
            for key in queries:
                b = normalize_scores(bm25_raw[key])
                t = normalize_scores(tfidf_raw[key])
                combined[key] = (b + t) / 2.0

            semantic_fit = np.clip(
                0.45 * combined["must"] + 0.20 * combined["nice"]
                + 0.35 * combined["ideal"] - 0.30 * combined["exclude"],
                0, 1
            ) * 100.0

            term_idf = bm25.term_idf_lookup()
            idf_floor = float(np.percentile(list(term_idf.values()), 75)) if term_idf else 0.0

            progress.progress(0.5, text="Scoring structured pillars...")

            results = []
            for i, c in enumerate(candidates):
                honeypot = ranker.is_honeypot(c)
                behavioral = ranker.score_behavioral(c, jd)
                experience = ranker.score_experience_quality(c, jd)
                sem = float(semantic_fit[i])
                composite = 0.50 * sem + 0.25 * behavioral + 0.25 * experience
                if honeypot:
                    composite *= 0.05
                results.append({
                    "candidate_id": c["candidate_id"],
                    "score": composite,
                    "honeypot": honeypot,
                    "pillars": {
                        "semantic_fit": round(sem, 2), "behavioral": round(behavioral, 2),
                        "experience_quality": round(experience, 2), "composite": round(composite, 2),
                    },
                    "doc_text": documents[i],
                    "_candidate": c,
                })
                progress.progress(0.5 + 0.5 * (i + 1) / len(candidates), text=f"Scoring {i+1}/{len(candidates)}")

            progress.empty()
            results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
            top_results = results[:min(top_n, len(results))]

            if top_results:
                max_s, min_s = top_results[0]["score"], top_results[-1]["score"]
=======
            progress_bar = st.progress(0, text="Scoring candidates...")
            results = []
            honeypots = []

            start = time.time()
            for i, c in enumerate(candidates):
                result = ranker.score_candidate(c)
                results.append(result)
                if result["honeypot"]:
                    honeypots.append(c["candidate_id"])
                progress_bar.progress(
                    (i + 1) / len(candidates),
                    text=f"Scoring {i+1}/{len(candidates)}..."
                )

            elapsed = time.time() - start
            progress_bar.empty()

            results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
            top_results = results[:min(top_n, len(results))]

            # Normalize scores
            if top_results:
                max_s = top_results[0]["score"]
                min_s = top_results[-1]["score"]
>>>>>>> 16501df389063dc484603aad8e618abe66de8156
                rng = max_s - min_s if max_s != min_s else 1.0
                for r in top_results:
                    r["normalized_score"] = round((r["score"] - min_s) / rng, 6)

<<<<<<< HEAD
            honeypot_ct = sum(1 for r in results if r["honeypot"])
            st.success(f"Ranked {len(candidates)} candidates | {honeypot_ct} honeypot(s) detected and penalized")

            import pandas as pd
=======
            st.success(f"✓ Ranked in {elapsed:.2f}s | {len(honeypots)} honeypot(s) detected and penalized")

            if honeypots:
                with st.expander(f"🪤 Honeypots detected: {len(honeypots)}"):
                    st.write(honeypots)

            st.subheader(f"🏆 Top {len(top_results)} Candidates")

            import pandas as pd

>>>>>>> 16501df389063dc484603aad8e618abe66de8156
            rows = []
            for rank_idx, r in enumerate(top_results, 1):
                c = r["_candidate"]
                p = c["profile"]
<<<<<<< HEAD
                reasoning = ranker.generate_reasoning(c, r["pillars"], jd, r["doc_text"], term_idf, idf_floor)
                rows.append({
                    "Rank": rank_idx, "ID": r["candidate_id"], "Title": p.get("current_title", ""),
                    "YOE": p.get("years_of_experience", 0),
                    "Score": r.get("normalized_score", r["score"]),
                    "Semantic Fit": r["pillars"]["semantic_fit"],
=======
                sig = c.get("redrob_signals", {}) or {}
                reasoning = ranker.generate_reasoning(r)
                rows.append({
                    "Rank": rank_idx,
                    "ID": r["candidate_id"],
                    "Title": p.get("current_title", ""),
                    "YOE": p.get("years_of_experience", 0),
                    "Location": f"{p.get('location','')} {p.get('country','')}".strip(),
                    "Score": r.get("normalized_score", r["score"]),
                    "Role Fit": r["pillars"]["role_fit"],
                    "Skill Match": r["pillars"]["skill_match"],
>>>>>>> 16501df389063dc484603aad8e618abe66de8156
                    "Behavioral": r["pillars"]["behavioral"],
                    "Exp Quality": r["pillars"]["experience_quality"],
                    "Honeypot": "🪤" if r["honeypot"] else "",
                    "Reasoning": reasoning,
                })

            df = pd.DataFrame(rows)
<<<<<<< HEAD
            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.4f"),
                    "Semantic Fit": st.column_config.ProgressColumn("Semantic Fit", min_value=0, max_value=100, format="%.0f"),
                    "Behavioral": st.column_config.ProgressColumn("Behavioral", min_value=0, max_value=100, format="%.0f"),
                    "Exp Quality": st.column_config.ProgressColumn("Exp Quality", min_value=0, max_value=100, format="%.0f"),
                },
            )

            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer, quoting=csv.QUOTE_ALL)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
=======

            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=1, format="%.4f"
                    ),
                    "Role Fit": st.column_config.ProgressColumn(
                        "Role Fit", min_value=0, max_value=100, format="%.0f"
                    ),
                    "Skill Match": st.column_config.ProgressColumn(
                        "Skill Match", min_value=0, max_value=100, format="%.0f"
                    ),
                    "Behavioral": st.column_config.ProgressColumn(
                        "Behavioral", min_value=0, max_value=100, format="%.0f"
                    ),
                    "Exp Quality": st.column_config.ProgressColumn(
                        "Exp Quality", min_value=0, max_value=100, format="%.0f"
                    ),
                },
                hide_index=True,
            )

            # Download CSV
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer, quoting=csv.QUOTE_ALL)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])

>>>>>>> 16501df389063dc484603aad8e618abe66de8156
            prev_s = None
            for rank_idx, r in enumerate(top_results, 1):
                ns = r.get("normalized_score", r["score"])
                if prev_s is not None and ns > prev_s:
                    ns = prev_s
                prev_s = ns
<<<<<<< HEAD
                reasoning = ranker.generate_reasoning(r["_candidate"], r["pillars"], jd, r["doc_text"], term_idf, idf_floor)
                writer.writerow([r["candidate_id"], rank_idx, round(ns, 6), reasoning])

            st.download_button("📥 Download Ranked CSV", data=csv_buffer.getvalue(),
                                file_name="team_Bug_Writer.csv", mime="text/csv")

st.divider()
st.caption("Bug Writer | Redrob Hackathon 2026 | CPU-only, no external APIs, JD parsed generically at runtime")
=======
                writer.writerow([
                    r["candidate_id"],
                    rank_idx,
                    round(ns, 6),
                    ranker.generate_reasoning(r),
                ])

            st.download_button(
                label="📥 Download Ranked CSV",
                data=csv_buffer.getvalue(),
                file_name="team_Bug_Writer.csv",
                mime="text/csv",
            )

st.divider()
st.caption("Bug Writer | Redrob Hackathon 2026 | CPU-only, no external APIs")
>>>>>>> 16501df389063dc484603aad8e618abe66de8156
