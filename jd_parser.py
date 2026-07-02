# jd_parser.py
import re
from docx import Document

HEADING_PATTERNS = {
    "must_have": [
        r"\babsolutely need\b", r"\bmust.?have\b", r"\brequired\b",
        r"\brequirements?\b", r"\bwhat you.?ll need\b", r"\bwhat we need\b",
        r"\bnon.?negotiable\b", r"\bessential\b", r"\bmandatory\b",
    ],
    "nice_to_have": [
        r"\bnice.?to.?have\b", r"\bpreferred\b", r"\bbonus\b",
        r"\bwould be nice\b", r"\bwon.?t reject\b", r"\bgood to have\b",
        r"\bplus\b.*\bnot required\b",
    ],
    "exclude": [
        r"\bdo not want\b", r"\bdon.?t want\b", r"\bnot looking for\b",
        r"\bnot a fit\b", r"\bdisqualif", r"\bred flags?\b",
        r"\bwe will not\b", r"\bwill not move forward\b",
        r"\bwe.?ve tried.*didn.?t work\b", r"\bwhat we mean by\b"
    ],
    "ideal_profile": [
        r"\bideal candidate\b", r"\bread between the lines\b",
        r"\bwe.?re imagining\b", r"\btypical profile\b", r"\bsummary\b",
    ],
    "responsibilities": [
        r"\bwhat you.?d.*doing\b", r"\bresponsibilit", r"\bday.?to.?day\b",
        r"\byou.?ll be\b", r"\bthe role\b", r"\bmandate\b",
    ],
    "culture": [
        r"\bvibe\b", r"\bculture\b", r"\bwork style\b", r"\bhow we work\b",
    ],
}

HEADING_STYLES = {"Heading 1", "Heading 2", "Heading 3", "Title"}

INLINE_DISQUALIFIER_PATTERNS = [
    r"\bnot\s+move\s+forward\b",
    r"\bwill\s+not\s+consider\b",
    r"\bnot\s+a\s+fit\b",
    r"\bnot\s+for\s+us\b",
    r"\bnot\s+eligible\b",
    r"\bdisqualif",
    r"\bred\s+flags?\b",
    r"\bdeal.?breaker\b",
    r"\bhard\s+requirement\b",
    r"\bwill\s+be\s+filtered\s+out\b",
    r"\bwe\s+will\s+not\b",
    r"\bwon.?t\s+be\s+considered\b",
    r"\bautomatic(?:ally)?\s+reject"
]

KNOWN_CITIES = {
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "pune", "chennai",
    "kolkata", "noida", "gurgaon", "gurugram", "ahmedabad", "jaipur", "chandigarh",
    "kochi", "trivandrum", "indore", "nagpur", "surat", "lucknow", "kanpur",
    "new york", "san francisco", "seattle", "austin", "boston", "chicago",
    "los angeles", "london", "toronto", "vancouver", "sydney", "melbourne",
    "singapore", "dubai", "abu dhabi", "berlin", "munich", "amsterdam",
    "tokyo", "hong kong",
}

def classify_heading(text: str) -> str | None:
    t = text.lower()
    for category, patterns in HEADING_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, t):
                return category
    return None

def extract_yoe_range(text: str) -> tuple[float, float] | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[\-–to]{1,3}\s*(\d+(?:\.\d+)?)\s*years?", text, re.I)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*years?", text, re.I)
    if m:
        lo = float(m.group(1))
        return (lo, lo + 6)
    return None

def extract_notice_preference_days(text: str) -> int | None:
    m = re.search(r"sub[\s\-]?(\d+)[\s\-]?day", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)[\s\-]?day(?:s)?\s+notice", text, re.I)
    if m:
        return int(m.group(1))
    return None

def extract_locations_mentioned(text: str) -> list[str]:
    t = text.lower()
    found = []
    for city in KNOWN_CITIES:
        if re.search(rf"\b{re.escape(city)}\b", t):
            found.append(city)
    return found

def extract_relocation_openness(text: str) -> bool:
    t = text.lower()
    return bool(re.search(r"open to relocat|welcome to apply|relocation candidates", t))

def extract_excluded_entities(exclude_text: str) -> list[str]:
    entities = []
    for paren in re.findall(r"\(([^)]+)\)", exclude_text):
        for token in paren.split(","):
            token = token.strip().rstrip(".")
            if not token or token.lower() in ("etc", "e.g", "i.e"):
                continue
            if token and token[0].isupper():
                entities.append(token)
    return entities

def extract_title_disqualifiers(exclude_text: str) -> list[str]:
    thresholds = []
    for m in re.finditer(r"every\s+(\d+(?:\.\d+)?)\s*years?", exclude_text, re.I):
        thresholds.append(float(m.group(1)) * 12)
    return thresholds

class ParsedJD:
    def __init__(self):
        self.raw_text: str = ""
        self.title: str = ""
        self.sections: dict[str, str] = {}
        self.yoe_range: tuple[float, float] | None = None
        self.notice_preference_days: int | None = None
        self.preferred_locations: list[str] = []
        self.open_to_relocation: bool = False
        self.excluded_entities: list[str] = []
        self.job_hop_threshold_months: float | None = None

    def get_section(self, category: str) -> str:
        return self.sections.get(category, "")

    def summary(self) -> str:
        lines = [f"Parsed JD: {self.title}"]
        if self.yoe_range:
            lines.append(f"  YOE range: {self.yoe_range[0]:.0f}-{self.yoe_range[1]:.0f} years")
        if self.notice_preference_days:
            lines.append(f"  Notice preference: <= {self.notice_preference_days} days")
        if self.preferred_locations:
            lines.append(f"  Preferred locations: {', '.join(self.preferred_locations)}")
        lines.append(f"  Open to relocation: {self.open_to_relocation}")
        if self.excluded_entities:
            lines.append(f"  Excluded entities detected: {', '.join(self.excluded_entities)}")
        for cat, text in self.sections.items():
            lines.append(f"  [{cat}] {len(text)} chars, {len(text.split())} words")
        return "\n".join(lines)

def parse_jd_docx(path: str) -> ParsedJD:
    doc = Document(path)
    parsed = ParsedJD()

    paragraphs = [(p.text.strip(), p.style.name if p.style else "Normal")
                  for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        raise ValueError(f"No text found in {path}")

    parsed.title = paragraphs[0][0]
    parsed.raw_text = "\n".join(t for t, _ in paragraphs)

    current_category = "general"
    buckets: dict[str, list[str]] = {"general": []}

    for text, style in paragraphs:
        if style in HEADING_STYLES:
            category = classify_heading(text)
            if category:
                current_category = category
                buckets.setdefault(category, [])
            else:
                current_category = "general"
                buckets.setdefault("general", [])
            continue
        buckets.setdefault(current_category, []).append(text)

    parsed.sections = {cat: " ".join(lines) for cat, lines in buckets.items() if lines}

    extra_exclude_sentences = []
    for text, style in paragraphs:
        if style in HEADING_STYLES:
            continue
        lowered = text.lower()
        has_disqualifier_phrase = any(re.search(pat, lowered) for pat in INLINE_DISQUALIFIER_PATTERNS)
        addresses_candidate = bool(re.search(r"\byou(?:'ve|'re|r)?\b|\byour\b", lowered))
        if has_disqualifier_phrase and addresses_candidate:
            extra_exclude_sentences.append(text)

    if extra_exclude_sentences:
        existing = parsed.sections.get("exclude", "")
        parsed.sections["exclude"] = (existing + " " + " ".join(extra_exclude_sentences)).strip()

    parsed.yoe_range = extract_yoe_range(parsed.raw_text)
    parsed.notice_preference_days = extract_notice_preference_days(parsed.raw_text)
    parsed.preferred_locations = extract_locations_mentioned(
        parsed.sections.get("general", "") + " " + parsed.title
    )
    parsed.open_to_relocation = extract_relocation_openness(parsed.raw_text)
    
    exclude_pool = parsed.sections.get("exclude", "")
    if len(exclude_pool) < 10:
        exclude_pool = parsed.raw_text

    parsed.excluded_entities = extract_excluded_entities(exclude_pool)
    hop_thresholds = extract_title_disqualifiers(exclude_pool)
    parsed.job_hop_threshold_months = min(hop_thresholds) if hop_thresholds else None

    return parsed

if __name__ == "__main__":
    import sys
    jd = parse_jd_docx(sys.argv[1] if len(sys.argv) > 1 else "job_description.docx")
    print(jd.summary())