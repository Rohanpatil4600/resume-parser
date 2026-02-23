from __future__ import annotations

from typing import Dict, List, Tuple, Any

import io
import hashlib

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file in-memory."""
    reader = PdfReader(io.BytesIO(file_bytes))
    texts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        texts.append(text)
    return "\n".join(texts)


def fetch_jd_from_url(url: str) -> str:
    """Best-effort extraction of visible text from a job description URL."""
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # crude heuristic: join paragraph and list item text
    parts: List[str] = []
    for tag in soup.find_all(["p", "li"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _simple_keywords(text: str, top_k: int = 25) -> List[str]:
    """Fallback: tokenize, drop short/stop, return top by frequency."""
    import re
    stop = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "this", "that", "it", "as", "from"}
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]*", text)
    words = [w.lower() for w in words if len(w) > 1 and w.lower() not in stop]
    from collections import Counter
    return [w for w, _ in Counter(words).most_common(top_k)]


def extract_keywords(jd_text: str, resume_text: str, top_k: int = 30) -> Tuple[List[str], List[str]]:
    """TF-IDF keyword extraction with fallback when TF-IDF returns too few."""
    jd_text = (jd_text or "").strip()
    resume_text = (resume_text or "").strip()
    if not jd_text and not resume_text:
        return [], []

    jd_keywords: List[str] = []
    resume_keywords: List[str] = []

    if jd_text and resume_text:
        try:
            docs = [jd_text, resume_text]
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=1000,
            )
            tfidf = vectorizer.fit_transform(docs)
            feature_names = vectorizer.get_feature_names_out()

            def top_keywords_for_row(row_idx: int) -> List[str]:
                row = tfidf[row_idx].toarray()[0]
                indices = row.argsort()[::-1][:top_k]
                return [feature_names[i] for i in indices if row[i] > 0]

            jd_keywords = top_keywords_for_row(0)
            resume_keywords = top_keywords_for_row(1)
        except Exception:
            pass

    if not jd_keywords and jd_text:
        jd_keywords = _simple_keywords(jd_text, top_k)
    if not resume_keywords and resume_text:
        resume_keywords = _simple_keywords(resume_text, top_k)

    return jd_keywords, resume_keywords


def _normalized_set(items: List[str]) -> List[str]:
    return sorted({i.strip().lower() for i in items if i and isinstance(i, str)})


def _collect_jd_skills(jd_structured: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (must_have, nice_to_have) skill lists, normalizing key names and merging technologies."""
    must = list(
        jd_structured.get("required_skills")
        or jd_structured.get("must_have_skills")
        or jd_structured.get("requiredSkills")
        or []
    )
    nice = list(
        jd_structured.get("nice_to_have_skills")
        or jd_structured.get("nice_to_have")
        or jd_structured.get("niceToHaveSkills")
        or []
    )
    tech = list(jd_structured.get("technologies") or jd_structured.get("tech_stack") or [])
    kw = list(jd_structured.get("keywords") or [])
    # Always merge tech and keywords so we have something to match when parser returns empty
    must = must + tech + kw[:20]
    return _normalized_set(must), _normalized_set(nice)


def _collect_resume_skills(resume_structured: Dict[str, Any]) -> List[str]:
    """Resume skills from skills list + technologies in roles/bullets."""
    skills = list(resume_structured.get("skills") or [])
    roles = resume_structured.get("roles") or []
    for r in roles:
        for b in r.get("bullets", []) or []:
            if isinstance(b, str) and len(b) > 10:
                import re
                for word in re.findall(r"[A-Z][a-z]+(?:[A-Z][a-z]*)*|[a-z]+(?:\.(?:js|ts|py|sql|aws))?", b):
                    if len(word) > 2 and word not in ("The", "With", "Using", "This", "That"):
                        skills.append(word)
    return _normalized_set(skills)


def compute_match_score(
    *,
    resume_structured: Dict[str, Any],
    jd_structured: Dict[str, Any],
    resume_keywords: List[str],
    jd_keywords: List[str],
) -> Dict[str, Any]:
    """Deterministic scoring + explanation.

    Components:
    - skill overlap
    - keyword overlap
    - requirement coverage (must-have vs nice-to-have)
    """
    resume_skills = _collect_resume_skills(resume_structured)
    jd_must, jd_nice = _collect_jd_skills(jd_structured)

    # When parser returns empty skill lists, use keyword lists so we still get a score
    if not jd_must and jd_keywords:
        jd_must = _normalized_set(jd_keywords[:40])
    if not resume_skills and resume_keywords:
        resume_skills = _normalized_set(resume_keywords[:40])

    # Skill overlap: which JD must-haves are covered by resume (exact or substring)
    def _covered(jd_list: List[str], resume_set: set) -> List[str]:
        r_lower = {s.lower() for s in resume_set}
        out = []
        for j in jd_list:
            j_lower = j.lower()
            if j_lower in r_lower:
                out.append(j)
            elif any(j_lower in r or r in j_lower for r in r_lower):
                out.append(j)
        return sorted(set(out))

    resume_set = set(resume_skills)
    must_overlap = _covered(jd_must, resume_set) if jd_must else []
    nice_overlap = _covered(jd_nice, resume_set) if jd_nice else []

    skill_score = 0.0
    if jd_must:
        skill_score += 60.0 * (len(must_overlap) / max(len(jd_must), 1))
    if jd_nice:
        skill_score += 20.0 * (len(nice_overlap) / max(len(jd_nice), 1))

    # Keyword overlap
    jd_kw = set(k.lower() for k in jd_keywords)
    resume_kw = set(k.lower() for k in resume_keywords)
    kw_overlap = sorted(jd_kw & resume_kw)
    kw_score = 20.0 * (len(kw_overlap) / max(len(jd_kw) or 1, 1))

    # If still zero (e.g. no keywords extracted), give a small baseline from any text overlap
    if skill_score == 0 and kw_score == 0 and jd_keywords and resume_keywords:
        kw_score = 10.0 * min(1.0, len(kw_overlap) / 5)

    raw_score = min(100.0, round(skill_score + kw_score, 1))

    # Requirement-to-evidence matrix with status (Strong / Weak / Missing) and risk
    requirements = jd_structured.get("requirements", [])
    if not requirements and jd_structured.get("responsibilities"):
        requirements = [{"text": r} if isinstance(r, str) else r for r in jd_structured["responsibilities"]]
    evidence_matrix = []
    roles = resume_structured.get("roles", [])
    risk_counts = {"high": 0, "medium": 0, "strong": 0}

    for req in requirements:
        req_text = req.get("text") if isinstance(req, dict) else str(req)
        requirement_entry = {
            "requirement": req_text,
            "evidence": [],
            "satisfied": False,
            "status": "Missing",
            "risk_level": "high",
            "evidence_summary": "",
        }

        req_lower = (req_text or "").lower()
        for role in roles:
            bullets = role.get("bullets", [])
            for b in bullets:
                b_text = b if isinstance(b, str) else str(b)
                if not b_text:
                    continue
                if any(token in b_text.lower() for token in req_lower.split()):
                    requirement_entry["evidence"].append(
                        {
                            "role": role.get("title"),
                            "company": role.get("company"),
                            "bullet": b_text,
                        }
                    )
        requirement_entry["satisfied"] = bool(requirement_entry["evidence"])

        # Status: Strong (multiple solid bullets), Weak (one or short), Missing
        ev = requirement_entry["evidence"]
        if not ev:
            requirement_entry["status"] = "Missing"
            requirement_entry["risk_level"] = "high"
            requirement_entry["evidence_summary"] = "Not mentioned"
            risk_counts["high"] += 1
        else:
            long_evidence = [e for e in ev if len((e.get("bullet") or "")) > 60]
            if len(ev) >= 2 and len(long_evidence) >= 1:
                requirement_entry["status"] = "Strong"
                requirement_entry["risk_level"] = "strong"
                risk_counts["strong"] += 1
            else:
                requirement_entry["status"] = "Weak"
                requirement_entry["risk_level"] = "medium"
                risk_counts["medium"] += 1
            # One-line evidence summary for table
            requirement_entry["evidence_summary"] = ev[0].get("bullet", "")[:120] + ("…" if len(ev[0].get("bullet", "")) > 120 else "")
            if len(ev) > 1:
                requirement_entry["evidence_summary"] += f" (+{len(ev) - 1} more)"

        evidence_matrix.append(requirement_entry)

    # Keyword coverage (ATS-style)
    jd_kw_list = jd_keywords[:30]
    missing_critical = [k for k in jd_kw_list if k.lower() not in {x.lower() for x in kw_overlap}][:10]
    keyword_coverage = {
        "matched_count": len(kw_overlap),
        "total_jd_keywords": len(jd_kw_list),
        "matched_pct": round(100.0 * len(kw_overlap) / max(len(jd_kw_list), 1), 0),
        "missing_critical": missing_critical,
        "overlap_list": kw_overlap,
    }

    # Overall risk summary
    if risk_counts["high"] > 0:
        risk_summary = f"High risk: {risk_counts['high']} missing must-have(s)"
        risk_emoji = "🔴"
    elif risk_counts["medium"] > 0:
        risk_summary = f"Medium risk: weak evidence in {risk_counts['medium']} area(s)"
        risk_emoji = "🟡"
    else:
        risk_summary = "Strong fit"
        risk_emoji = "🟢"

    explanation = {
        "overall_score": raw_score,
        "breakdown": {
            "skills_component": round(skill_score, 1),
            "keywords_component": round(kw_score, 1),
        },
        "skills": {
            "must_have_list": jd_must,
            "must_have_overlap": must_overlap,
            "nice_to_have_list": jd_nice,
            "nice_to_have_overlap": nice_overlap,
        },
        "keywords": {
            "jd_keywords_top": jd_keywords,
            "resume_keywords_top": resume_keywords,
            "overlap": kw_overlap,
        },
        "requirements_matrix": evidence_matrix,
        "keyword_coverage": keyword_coverage,
        "risk_summary": risk_summary,
        "risk_emoji": risk_emoji,
        "risk_counts": risk_counts,
    }

    return explanation


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

