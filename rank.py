# rank.py
import json
import csv
import argparse
import time
from datetime import datetime
import numpy as np

from jd_parser import parse_jd_docx, ParsedJD
from bm25 import BM25Index, normalize_scores, tokenize, GENERIC_FILLER_WORDS
from text_features import TfidfSimilarityIndex, build_candidate_document

REFERENCE_DATE = datetime(2026, 6, 30)

DEFAULT_NOTICE_CEILING_DAYS = 45
DEFAULT_STABLE_TENURE_MONTHS = 24

def is_honeypot(candidate: dict) -> bool:
    p = candidate["profile"]
    yoe = p.get("years_of_experience", 0) or 0
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    total_career_months = sum(j.get("duration_months", 0) or 0 for j in career)
    if total_career_months > (yoe * 12) + 18:
        return True

    zero_dur_expert = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    if zero_dur_expert >= 3:
        return True

    if len(skills) >= 8 and all(s.get("proficiency") == "expert" for s in skills):
        return True

    for job in career:
        dur = job.get("duration_months", 0) or 0
        if dur > (yoe * 12) + 12 and yoe > 0:
            return True

    return False

def days_since(date_str: str) -> int:
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return max(0, (REFERENCE_DATE - d).days)
    except Exception:
        return 999

def score_behavioral(candidate: dict, jd: ParsedJD) -> float:
    sig = candidate.get("redrob_signals", {}) or {}
    score = 0.0

    days_inactive = days_since(sig.get("last_active_date"))
    if days_inactive <= 7:
        score += 25
    elif days_inactive <= 30:
        score += 20
    elif days_inactive <= 60:
        score += 14
    elif days_inactive <= 120:
        score += 8
    elif days_inactive <= 180:
        score += 3

    if sig.get("open_to_work_flag"):
        score += 10

    resp_rate = sig.get("recruiter_response_rate") or 0.0
    score += resp_rate * 20

    notice = sig.get("notice_period_days")
    notice_ceiling = jd.notice_preference_days or DEFAULT_NOTICE_CEILING_DAYS
    if notice is None:
        score += 7
    elif notice <= notice_ceiling * 0.5:
        score += 15
    elif notice <= notice_ceiling:
        score += 12
    elif notice <= notice_ceiling * 2:
        score += 6
    elif notice <= notice_ceiling * 3:
        score += 2

    github = sig.get("github_activity_score")
    if github is not None and github >= 0:
        score += min(8, (github / 100) * 8)

    icr = sig.get("interview_completion_rate")
    if icr is not None and icr >= 0:
        score += icr * 7

    apps = sig.get("applications_submitted_30d") or 0
    if apps >= 5:
        score += 5
    elif apps >= 2:
        score += 3
    elif apps >= 1:
        score += 1

    return min(score, 100.0)

def score_experience_quality(candidate: dict, jd: ParsedJD) -> float:
    p = candidate["profile"]
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])

    score = 0.0
    yoe = p.get("years_of_experience") or 0

    if jd.yoe_range:
        lo, hi = jd.yoe_range
        span = max(hi - lo, 1.0)
        if lo <= yoe <= hi:
            center = (lo + hi) / 2
            distance_from_center = abs(yoe - center) / (span / 2)
            score += 30 * (1.0 - 0.3 * distance_from_center)
        elif yoe < lo:
            score += max(0, 20 - (lo - yoe) * 4)
        else:
            score += max(0, 15 - (yoe - hi) * 2)
    else:
        score += 15

    product_months, excluded_months = 0, 0
    all_tenures = []
    for job in career:
        company = (job.get("company") or "").lower()
        dur = job.get("duration_months") or 0
        all_tenures.append(dur)
        is_excluded = any(ent.lower() in company for ent in jd.excluded_entities)
        if is_excluded:
            excluded_months += dur
        else:
            product_months += dur

    total_months = product_months + excluded_months
    if total_months > 0:
        score += (product_months / total_months) * 25

    if all_tenures:
        stable_threshold = jd.job_hop_threshold_months or DEFAULT_STABLE_TENURE_MONTHS
        avg_tenure = sum(all_tenures) / len(all_tenures)
        ratio = min(avg_tenure / stable_threshold, 1.5)
        score += min(20, ratio * 13.3)

    edu_score = 0
    for edu in education:
        tier = (edu.get("tier") or "").lower()
        if tier == "tier_1":
            edu_score = max(edu_score, 15)
        elif tier == "tier_2":
            edu_score = max(edu_score, 9)
        elif tier == "tier_3":
            edu_score = max(edu_score, 4)
    score += edu_score

    return min(score, 100.0)

def top_overlapping_terms(doc_text: str, query_text: str, term_idf: dict, idf_floor: float, top_k: int = 3) -> list:
    doc_tokens = set(tokenize(doc_text))
    query_tokens = dict.fromkeys(tokenize(query_text))
    candidates = [
        t for t in query_tokens
        if t in doc_tokens
        and len(t) >= 4
        and t not in GENERIC_FILLER_WORDS
        and term_idf.get(t, 0.0) >= idf_floor
    ]
    candidates.sort(key=lambda t: term_idf.get(t, 0.0), reverse=True)
    return candidates[:top_k]

def generate_reasoning(candidate: dict, pillars: dict, jd: ParsedJD, doc_text: str,
                        term_idf: dict, idf_floor: float) -> str:
    p = candidate["profile"]
    sig = candidate.get("redrob_signals", {}) or {}

    title = p.get("current_title", "Unknown")
    yoe = p.get("years_of_experience", 0)
    company = p.get("current_company", "")
    location = p.get("location", "")
    country = p.get("country", "")

    notice = sig.get("notice_period_days")
    days_ago = days_since(sig.get("last_active_date"))
    github = sig.get("github_activity_score")
    relocate = sig.get("willing_to_relocate", False)

    parts = []
    score = pillars["composite"]

    matched_must = top_overlapping_terms(doc_text, jd.get_section("must_have"), term_idf, idf_floor)
    matched_ideal = top_overlapping_terms(doc_text, jd.get_section("ideal_profile"), term_idf, idf_floor)
    matched_exclude = top_overlapping_terms(doc_text, jd.get_section("exclude"), term_idf, idf_floor)

    if score >= 60:
        evidence = matched_must or matched_ideal
        if evidence:
            parts.append(f"{yoe:.0f}yr {title} at {company}; profile overlaps JD must-haves on {', '.join(evidence)}")
        else:
            parts.append(f"{yoe:.0f}yr {title} at {company}; strong overall JD-fit signal")
    elif score >= 35:
        parts.append(f"{title} ({yoe:.0f}yrs) at {company}; partial JD overlap")
    else:
        parts.append(f"{title} ({yoe:.0f}yrs); low semantic overlap with JD requirements")

    flags = []
    if notice is not None:
        ceiling = jd.notice_preference_days or DEFAULT_NOTICE_CEILING_DAYS
        if notice <= ceiling:
            flags.append(f"available in {notice} days")
        elif notice > ceiling * 2:
            flags.append(f"{notice}-day notice (above JD's stated preference)")

    if days_ago > 180:
        flags.append(f"inactive {days_ago // 30}mo")
    elif days_ago <= 14:
        flags.append("recently active")

    if country.lower() == "india" and any(loc in location.lower() for loc in jd.preferred_locations):
        flags.append(f"based in {location}")
    elif jd.preferred_locations and country.lower() != "india" and not relocate:
        flags.append("outside India, not open to relocation")
    elif jd.preferred_locations and country.lower() == "india" and not any(loc in location.lower() for loc in jd.preferred_locations) and not relocate:
        flags.append(f"based in {location}, not open to relocation")

    if github is not None and github >= 60:
        flags.append(f"GitHub activity {github:.0f}/100")

    if len(matched_exclude) >= 2:
        flags.append(f"overlaps JD's exclusion language on {', '.join(matched_exclude[:2])}")

    if flags:
        parts.append("; ".join(flags))

    return "; ".join(parts)[:300]

def load_candidates(filepath: str):
    import gzip
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def rank_candidates(candidates_path: str, jd_path: str, output_path: str, top_n: int = 100):
    t0 = time.time()
    print(f"Parsing JD: {jd_path}", flush=True)
    jd = parse_jd_docx(jd_path)
    print(jd.summary(), flush=True)
    print(flush=True)

    print(f"Loading candidates from: {candidates_path}", flush=True)
    candidates = []
    documents = []
    for i, c in enumerate(load_candidates(candidates_path)):
        candidates.append(c)
        documents.append(build_candidate_document(c))
        if (i + 1) % 20000 == 0:
            print(f"  Loaded {i+1:,} candidates...", flush=True)

    n = len(candidates)
    print(f"Loaded {n:,} candidates. Fitting BM25 + TF-IDF indices...", flush=True)

    bm25 = BM25Index().fit(documents)
    tfidf = TfidfSimilarityIndex().fit(documents)
    print(f"  Indices fit in {time.time()-t0:.1f}s", flush=True)

    exclude_query = jd.get_section("exclude")
    if not exclude_query or len(exclude_query) < 10:
        exclude_query = jd.raw_text

    queries = {
        "must": jd.get_section("must_have") or jd.raw_text,
        "nice": jd.get_section("nice_to_have"),
        "exclude": exclude_query,
        "ideal": jd.get_section("ideal_profile") or jd.get_section("general"),
    }

    print("Scoring BM25 queries...", flush=True)
    bm25_raw = bm25.score_queries(queries)
    print("Scoring TF-IDF queries...", flush=True)
    tfidf_raw = tfidf.score_queries(queries)

    combined = {}
    for key in queries:
        b = normalize_scores(bm25_raw[key])
        t = normalize_scores(tfidf_raw[key])
        combined[key] = (b + t) / 2.0

    semantic_fit = np.clip(
        0.45 * combined["must"]
        + 0.20 * combined["nice"]
        + 0.35 * combined["ideal"]
        - 0.30 * combined["exclude"],
        0, 1
    ) * 100.0

    print(f"Semantic fit range: {semantic_fit.min():.1f} - {semantic_fit.max():.1f}", flush=True)
    print("Computing structured pillars + honeypot checks...", flush=True)

    term_idf = bm25.term_idf_lookup()
    idf_floor = float(np.percentile(list(term_idf.values()), 75)) if term_idf else 0.0

    results = []
    honeypot_count = 0
    for i, c in enumerate(candidates):
        honeypot = is_honeypot(c)
        if honeypot:
            honeypot_count += 1

        behavioral = score_behavioral(c, jd)
        experience = score_experience_quality(c, jd)
        sem = float(semantic_fit[i])

        composite = 0.50 * sem + 0.25 * behavioral + 0.25 * experience
        
        sig = c.get("redrob_signals", {}) or {}
        notice = sig.get("notice_period_days")
        days_inactive = days_since(sig.get("last_active_date"))
        notice_ceiling = jd.notice_preference_days or DEFAULT_NOTICE_CEILING_DAYS
        
        if notice is not None:
            if notice > notice_ceiling * 3:
                composite *= 0.3
            elif notice > notice_ceiling * 2:
                composite *= 0.5
            elif notice > notice_ceiling:
                composite *= 0.8
                
        if days_inactive > 180:
            composite *= 0.1
        elif days_inactive > 120:
            composite *= 0.4
        elif days_inactive > 60:
            composite *= 0.7

        country = (c.get("profile", {}).get("country") or "").lower()
        location = (c.get("profile", {}).get("location") or "").lower()
        relocate = sig.get("willing_to_relocate", False)

        if jd.preferred_locations:
            is_india = (country == "india")
            in_target_city = any(loc in location for loc in jd.preferred_locations)
            
            if not is_india and not relocate:
                composite *= 0.3
            elif is_india and not in_target_city and not relocate:
                composite *= 0.6

        if honeypot:
            composite *= 0.05

        results.append({
            "candidate_id": c["candidate_id"],
            "score": composite,
            "honeypot": honeypot,
            "pillars": {
                "semantic_fit": round(sem, 2),
                "behavioral": round(behavioral, 2),
                "experience_quality": round(experience, 2),
                "composite": round(composite, 2),
            },
            "doc_text": documents[i],
            "_candidate": c,
        })

        if (i + 1) % 20000 == 0:
            print(f"  Scored {i+1:,}/{n:,}...", flush=True)

    print(f"Scored {n:,} candidates ({honeypot_count} honeypots detected and penalized)", flush=True)

    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top_results = results[:top_n]

    max_score = top_results[0]["score"] if top_results else 1.0
    min_score = top_results[-1]["score"] if top_results else 0.0
    score_range = max_score - min_score if max_score != min_score else 1.0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        prev_score = None
        for rank_idx, r in enumerate(top_results, 1):
            normalized = (r["score"] - min_score) / score_range
            if prev_score is not None and normalized > prev_score:
                normalized = prev_score
            prev_score = normalized

            reasoning = generate_reasoning(r["_candidate"], r["pillars"], jd, r["doc_text"], term_idf, idf_floor)
            writer.writerow([r["candidate_id"], rank_idx, round(normalized, 6), reasoning])

    print(f"\nSubmission written to: {output_path}", flush=True)
    print("  Top 10 candidates:", flush=True)
    for i, r in enumerate(top_results[:10], 1):
        p = r["_candidate"]["profile"]
        pl = r["pillars"]
        print(
            f"  #{i:2d} {r['candidate_id']} | {p['current_title']:32s} | "
            f"score={r['score']:.2f} | sem={pl['semantic_fit']:.0f} "
            f"behav={pl['behavioral']:.0f} exp={pl['experience_quality']:.0f}",
            flush=True
        )

    print(f"\nTotal runtime: {time.time()-t0:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon - JD-driven hybrid-IR candidate ranker (Team: Bug Writer)"
    )
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--jd", default="./job_description.docx", help="Path to JD .docx file")
    parser.add_argument("--out", default="./team_Bug_Writer.csv")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    rank_candidates(args.candidates, args.jd, args.out, top_n=args.top)


if __name__ == "__main__":
    main()