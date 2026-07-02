# 🤖 Redrob AI Candidate Ranker — Team Bug Writer

> **Hackathon:** Redrob Intelligent Candidate Discovery & Ranking Challenge  
> **Team:** Bug Writer | **Participant:** Saad Shaikh  
> **Contact:** saadshaikh121103@gmail.com

## 🎯 The Philosophy: What this does, and what it deliberately does NOT do

This system does **not** contain a Python dictionary of AI/ML skill names, company names, or disqualifying titles. The hackathon documentation explicitly calls out "finding candidates whose skills _match the JD context_", not a hardcoded checklist.

Instead, **the JD itself (any `.docx`) is parsed at runtime** into structured requirements. Candidates are then scored against those requirements using a classical **Hybrid Information Retrieval pipeline** (BM25 + TF-IDF) combined with behavioral and experience signals.

**Swap in a different job description, and this system produces an entirely different ranking without a single code change.**

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INPUT SOURCES                                       │
│                                                                              │
│  job_description.docx                              candidates.jsonl         │
│         │                                                  │                 │
└────────┼──────────────────────────────────────────────────┼─────────────────┘
         │                                                  │
         ▼                                                  ▼
    ┌────────────────────┐                      ┌──────────────────────────┐
    │   jd_parser.py     │                      │   text_features.py       │
    │                    │                      │                          │
    │ Parse & Structure: │                      │ Build candidate corpus:  │
    │ • Must-have skills │                      │ • Headline               │
    │ • Nice-to-have     │                      │ • Career summary         │
    │ • Exclusions       │                      │ • Experience bullets     │
    │ • YOE range        │                      │ • Skills list            │
    │ • Notice period    │                      │ • Education              │
    │ • Location prefs   │                      │                          │
    └────────┬───────────┘                      └──────────┬───────────────┘
             │                                             │
             │ Structured Queries                         │
             │                                    ┌────────▼────────┐
             │                                    │   Indexing      │
             │                        ┌───────────┤                 │
             │                        │           │ BM25 Index      │
             │                        │           │ TF-IDF Index    │
             │                        │           │                 │
             │                        │           │ Fit on full     │
             │                        │           │ 100K corpus     │
             │                        │           └────────┬────────┘
             │                        │                    │
             └────────────┬───────────┼────────────────────┘
                          │           │
                          ▼           ▼
                    ┌──────────────────────────────────┐
                    │   Scoring Pipeline               │
                    │                                  │
                    │ For each candidate:              │
                    │ 1. Semantic Fit (BM25+TF-IDF)   │
                    │ 2. Behavioral Signals            │
                    │ 3. Experience Quality            │
                    │ 4. Honeypot Detection            │
                    └──────────┬───────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
    ┌────────────┐        ┌──────────────┐     ┌──────────────┐
    │ Semantic   │        │  Behavioral  │     │ Experience   │
    │ Fit        │        │  Signals     │     │ Quality      │
    │ (50%)      │        │  (25%)       │     │ (25%)        │
    │            │        │              │     │              │
    │ • BM25     │        │ • Recency    │     │ • YOE fit    │
    │ • TF-IDF   │        │ • Response   │     │ • Tenure     │
    │ • Relevance│        │ • Notice pd  │     │ • Education  │
    └─────┬──────┘        └───────┬──────┘     └────────┬─────┘
          │                       │                     │
          │        ┌──────────────┴─────────────┬───────┘
          │        │                            │
          ▼        ▼                            ▼
    ┌──────────────────────────────────────────────────┐
    │   Composite Scoring & Penalty Application        │
    │                                                  │
    │ Score = Σ(pillars) × (behavioral penalties)    │
    │       × (location/visa gates)                    │
    │       × (honeypot detection)                     │
    │                                                  │
    │ Steep Multiplicative Penalties:                 │
    │ • Inactive >6mo: ×0.1                           │
    │ • Wrong location: ×0.3–0.6                      │
    │ • Honeypot detected: ×0.05                      │
    └──────────────────┬───────────────────────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │  Rank & Output         │
            │                        │
            │ team_Bug_Writer.csv    │
            │ (sorted by composite)  │
            └────────────────────────┘
```

---

## 🔧 Component Details

### 1. Generic JD Parsing (`jd_parser.py`)

Uses **document structure and semantic headings**, not hard-coded keywords:

- **Heading Classification:** Walks all paragraphs and classifies headings into logical categories (`must_have`, `exclude`, `ideal_profile`) using generic regex patterns:
  - `"must.?have"`, `"required"`, `"mandatory"`
  - `"do not want"`, `"disqualif"`, `"exclude"`
  - `"nice.?to.?have"`, `"ideal"`, `"preferred"`

- **Structured Extraction:** Parses years of experience ranges, notice-period constraints, preferred cities (using a 40-city global gazetteer), and visa sponsorship requirements.

- **Completeness:** Scans the **entire JD** for inline disqualifiers (e.g., "What we mean by 5+ years") so no critical exclusion criteria slip through.

### 2. Hybrid Information Retrieval (`bm25.py` + `text_features.py`)

Combines exact-match and semantic retrieval for robustness:

- **BM25 (Exact Match):** Okapi BM25 with tuned parameters (`k1=1.5, b=0.75`) identifies candidates with strong keyword overlap.
- **TF-IDF (Semantic):** Corpus-relative term importance captures domain relevance even when exact terms vary.
- **Runtime Indexing:** Indexes the full 100K-candidate corpus at startup via vectorized sparse-matrix operations in NumPy/SciPy.
- **Query Scoring:** Each parsed JD section (must-have, nice-to-have) is scored independently, then aggregated.

### 3. Structured Signal Pillars (50/25/25 Split)

Composite score balances multiple dimensions:

| Pillar | Weight | Signals |
|--------|--------|---------|
| **Semantic Fit** | 50% | BM25 score, TF-IDF relevance, keyword density |
| **Behavioral** | 25% | Last activity recency, response rate, notice period alignment |
| **Experience Quality** | 25% | YOE band fit, tenure stability (anti-job-hopping), education tier |

**Rationale:** The JD's "how to read between the lines" section frames technical fit as the primary gate but _explicitly elevates behavioral availability to a first-class signal_. A ghost candidate (inactive 6+ months) with perfect skills is less valuable than an available mid-tier candidate.

### 4. Honeypot Detection (`is_honeypot()`)

Pure data-integrity checks, independent of text scoring:

- **Career Duration Anomaly:** Sum of job tenures exceeds stated total years of experience.
- **Skill-Usage Anomaly:** 3+ skills marked "expert" with 0 months of reported usage.
- **Single-Job Anomaly:** One job tenure exceeds total stated experience.
- **Penalty:** Honeypots get `composite *= 0.05`, naturally dropping them out of the Top 100 without manual overrides.

### 5. Location & Visa Gates (Multiplicative Filters)

Hard gates applied as multiplicative modifiers:

- **Outside India, not relocating:** `composite *= 0.3`
- **In India, wrong city, not relocating:** `composite *= 0.6`
- **Visa sponsorship required but candidate unavailable:** `composite *= 0.1`

These aren't soft nudges—they're multipliers that can collapse a score from Top 50 to Top 500 in a single gate.

---

## 🚀 How to Reproduce the Submission

The pipeline requires **zero external API calls**, downloads no pre-trained weights, and relies solely on standard optimized Python libraries: `numpy`, `scipy`, `scikit-learn`, `python-docx`.

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/saad-0012/smart-hiring-agent
cd smart-hiring-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the ranking pipeline
python rank.py \
  --candidates ./data/candidates.jsonl \
  --jd ./data/job_description.docx \
  --out ./output/team_Bug_Writer.csv

# 4. Validate the output
python validate_submission.py ./output/team_Bug_Writer.csv
```

---

## ⏱️ Compute Constraints Met

| Constraint | Limit | This System | Status |
|-----------|-------|------------|--------|
| **Runtime** | ≤ 5 min | ~35 sec | ✅ PASS |
| **RAM** | ≤ 16 GB | ~1.5 GB | ✅ PASS |
| **Compute** | CPU only | CPU only | ✅ PASS |
| **Network** | Offline | 0 external calls | ✅ PASS |

---

## 🛡️ Engineering Decisions, Defended

### Why BM25 + TF-IDF instead of Neural Embeddings?

We evaluated Local LLM Embeddings (SentenceTransformers + ONNX quantization) seriously. However:

- **Determinism:** BM25 + TF-IDF are fully deterministic across runs and hardware.
- **CPU Performance:** Dense vector operations on 100K candidates, even optimized, risk violating the 5-minute constraint on modest CPUs.
- **Transparency:** Sparse linear models are auditable; neural embeddings are black boxes.
- **No External Weights:** No dependency on downloading pre-trained models; everything fits in Git.

### Why 50/25/25 instead of a flat score?

The JD itself frames this hierarchy:
1. **Technical fit is the gate** (must have relevant skills) → 50% weight.
2. **Availability matters as much as skill** (responsiveness, notice period) → 25% weight.
3. **Stability signals predict retention** (tenure, education, seniority) → 25% weight.

A candidate with perfect skills but inactive for 9 months and a 90-day notice period should rank _below_ an available mid-tier candidate. The weighting reflects that.

### Why multiplicative penalties instead of additive?

- **Additive:** Bad behavior only _slightly_ reduces a high score.
- **Multiplicative:** A 0.3× multiplier on location mismatch creates a _hard rejection_ floor without domain-specific threshold tuning.

This matches how hiring actually works: "We need someone in India" isn't a -5 point adjustment; it's a gate.

---

## 📊 Output Format

The generated `team_Bug_Writer.csv` contains:

| Column | Type | Description |
|--------|------|-------------|
| `candidate_id` | str | Unique identifier from the input |
| `candidate_name` | str | Candidate full name |
| `composite_score` | float | Final ranking score (0–1) |
| `semantic_fit` | float | BM25+TF-IDF relevance (0–1) |
| `behavioral_score` | float | Availability + recency (0–1) |
| `experience_score` | float | YOE + tenure quality (0–1) |
| `is_honeypot` | bool | True if data anomalies detected |
| `location_gate` | float | Multiplier applied for location |
| `final_rank` | int | Position in ranked list (1–100) |

Rows are sorted by `composite_score` descending; top 100 candidates are recommended.

---

## 📁 Repository Structure

```
smart-hiring-agent/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── rank.py                      # Main entry point
├── validate_submission.py        # Output validator
├── src/
│   ├── jd_parser.py            # JD document parsing
│   ├── bm25.py                 # Okapi BM25 indexing
│   ├── text_features.py        # Candidate document builder
│   ├── scoring.py              # Composite scoring logic
│   └── honeypot.py             # Anomaly detection
├── data/
│   ├── candidates.jsonl        # Input: 100K candidates
│   ├── job_description.docx    # Input: JD template
│   └── gazetteer.json          # City reference list
└── output/
    └── team_Bug_Writer.csv     # Output: ranked candidates
```

---

## 🤝 Contributing

Found a bug or have a suggestion? Please open an issue on GitHub.

---

## 📝 License

This project is submitted to the Redrob Hackathon under the terms of the competition. For licensing inquiries, contact the organizers.

---

**Last Updated:** July 2, 2024  
**Version:** 1.0 (Hackathon Submission)
