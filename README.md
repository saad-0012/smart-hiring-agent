Here is the finalized, clean, and highly professional `README.md` for your repository. It reflects your robust, dynamic architecture, clearly defends your engineering decisions, and highlights the specific penalties (location, notice period, honeypots) that make your system a production-ready ranker.

---

```markdown
# 🤖 Redrob AI Candidate Ranker — Team Bug Writer

> **Hackathon:** Redrob Intelligent Candidate Discovery & Ranking Challenge
> **Team:** Bug Writer | **Participant:** Saad Shaikh
> **Contact:** saadshaikh121103@gmail.com

## 🎯 The Philosophy: What this does, and what it deliberately does NOT do

This system does **not** contain a Python dictionary of AI/ML skill names, company names, or disqualifying titles. The hackathon documentation explicitly calls out "finding candidates whose skills section contains the most AI keywords" as a trap built into the dataset — so this system is built to never do that.

Instead, **the JD itself (any `.docx`) is parsed at runtime** into structured requirements. Candidates are then scored against those requirements using a classical **Hybrid Information Retrieval pipeline (BM25 + TF-IDF)** fit fresh on the candidate corpus. This is the exact family of techniques the JD states Redrob's own production system uses. 

Swap in a different job description, and this system produces an entirely different ranking without a single code change.

---

## 📐 System Architecture

```text
job_description.docx                    candidates.jsonl
        │                                        │
        ▼                                        ▼
┌───────────────────┐              ┌──────────────────────────┐
│   jd_parser.py    │              │  text_features.py        │
│                   │              │                          │
│ Reads heading     │              │ Build candidate document │
│ STRUCTURE and     │              │ (headline + summary +    │
│ generic regex for:│              │ career desc + skills)    │
│  • YOE range      │              └──────────────┬───────────┘
│  • Notice period  │                             │
│  • City mentions  │                             ▼
│  • Exclude lists  │              ┌──────────────────────────┐
└─────────┬─────────┘              │  BM25Index (Exact Match) │
          │                        │  TF-IDF Index (Semantic) │
          │                        │  Fit fresh on 100K pool  │
          └───────────────┬────────┴──────────────┬───────────┘
                          ▼                       │
              Score against parsed queries        │
             (Must-Have, Nice-to-Have, Exclude)   │
                          │                       │
        ┌─────────────────┼───────────────────────┼─────────────────┐
        ▼                 ▼                       ▼                 ▼
  Semantic Fit    Behavioral Signals      Experience Quality    Honeypot
     (50%)              (25%)                   (25%)            Filter
        │                 │                       │                 │
        └─────────────────┴─────────┬─────────────┴─────────────────┘
                                    ▼
                          Composite Scoring & 
                    Steep Multiplicative Penalties
                                    │
                                    ▼
                           team_Bug_Writer.csv

```

### 1. Generic JD Parsing (`jd_parser.py`)

Uses **document structure**, not this JD's specific wording:

* Walks paragraphs and classifies headings into generic categories (`must_have`, `exclude`, `ideal_profile`) via common JD-vocabulary regex patterns (`"must.?have"`, `"do not want"`, `"disqualif"`).
* Extracts YOE bounds, notice-period limits, and preferred cities using a generic 40-city global gazetteer.
* **Fix Implemented:** Scans the *entire* JD for inline disqualifiers (e.g., "What we mean by X years") so no critical exclusion criteria are missed.

### 2. Hybrid Retrieval (`bm25.py` + `text_features.py`)

* Fits Okapi BM25 (`k1=1.5, b=0.75`) and TF-IDF cosine similarity on the 100K-candidate corpus at runtime via vectorized sparse-matrix operations.
* Scores candidates against the parsed JD sections. Evaluates hard exact-terms (BM25) alongside softer, corpus-relative term importance (TF-IDF).

### 3. Structured Signal Pillars & Steep Penalties

* **Behavioral Signals:** Checks recency, response rate, and notice period against the JD's *parsed* preferences.
* *Penalty Logic:* Notice periods and inactivity are treated as **steep multiplicative penalties**, not just additive nudges. A candidate inactive for >6 months sees their score slashed by 90%.


* **Experience Quality:** Evaluates YOE-band fit, tenure stability (anti-title-chasing), and education tier.
* **Location/Visa Hard Gates:** Applied as multiplicative filters. `composite *= 0.3` if outside India and not relocating; `composite *= 0.6` if in India but the wrong city and not relocating.

### 4. Honeypot Detection (`is_honeypot`)

Pure data-integrity checks, independent of text scoring:

* Career-history duration sums that mathematically exceed stated YOE.
* 3+ "expert" skills claimed with 0 months of usage.
* Single job tenure exceeding total stated experience.
* *Penalty:* Honeypots get their composite score multiplied by `0.05` — naturally dropping them out of the Top 100 without manual overrides.

---

## 🚀 How to Reproduce the Submission

The pipeline requires **zero external API calls**, downloads no external weights, and relies purely on standard optimized Python libraries (`numpy`, `scipy`, `scikit-learn`, `python-docx`).

```bash
# 1. Clone the repository
git clone [https://github.com/saad-0012/redrob-ranker](https://github.com/saad-0012/redrob-ranker)
cd redrob-ranker

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the ranking pipeline
python rank.py \
  --candidates ./candidates.jsonl \
  --jd ./job_description.docx \
  --out ./team_Bug_Writer.csv

```

### Validate the Output

```bash
python validate_submission.py team_Bug_Writer.csv

```

---

## ⏱️ Compute Constraints Met

| Constraint | Limit | This System | Status |
| --- | --- | --- | --- |
| **Runtime** | ≤ 5 min | **~35 sec** | ✅ PASS |
| **RAM** | ≤ 16 GB | **~1.5 GB** | ✅ PASS |
| **Compute** | CPU only | **CPU only** | ✅ PASS |
| **Network** | Offline | **0 external calls** | ✅ PASS |

---

## 🛡️ Design Decisions, Defended

**Why BM25 + TF-IDF instead of Neural Embeddings?**
We considered Local LLM Embeddings (SentenceTransformers) seriously. However, guaranteeing a sub-5-minute runtime on a CPU for 100,000 dense vectors is computationally volatile. Shipping a heavily optimized, sparse-matrix classical IR system ensures **100% reproducible success in the Stage 3 Sandbox**. The architecture is modular; `BM25Index` exposes a `.score_queries()` interface, meaning a neural index could be dropped in seamlessly in a production environment with GPU access.

**Why 50/25/25 weighting?**
The JD's own "how to read between the lines" section frames technical fit as the primary gate but explicitly elevates behavioral availability to a first-class signal. A ghost candidate (inactive 6 months) might score perfectly on Semantic Fit, but our steep multiplicative behavioral penalties ensure they are correctly removed from the recruiter's shortlist.

```

```