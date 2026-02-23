from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from utils_llm import parse_json_safe


def plan_rewrites(
    *,
    llm: BaseChatModel,
    resume_structured: Dict[str, Any],
    jd_structured: Dict[str, Any],
    match_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide which bullets to rewrite and what to emphasize."""
    prompt = ChatPromptTemplate.from_template(
        """
You are a resume tailoring strategist.

Given:
- a structured resume,
- a structured job description,
- a deterministic match report (with requirements matrix and gap analysis),

Produce a JSON plan for targeted rewrites.

Return STRICT JSON with keys:
- bullets_to_rewrite: list[{{ role_index: int, bullet_index: int, reason: string }}]
- skills_to_emphasize: list[str]
- keywords_to_include: list[str]
- guidance: string   # short narrative of the overall strategy

Guidelines:
- Focus on bullets that are close to the job needs but undersell impact, metrics, or relevant tools.
- Prefer editing existing content over inventing new experience.
- Align with must-have skills and important keywords from the JD.

Resume (JSON):
```json
{resume_structured}
```

Job description (JSON):
```json
{jd_structured}
```

Match report (JSON):
```json
{match_report}
```
"""
    )
    chain = prompt | llm
    raw = chain.invoke(
        {
            "resume_structured": resume_structured,
            "jd_structured": jd_structured,
            "match_report": match_report,
        }
    )
    content = raw.content if hasattr(raw, "content") else str(raw)
    data = parse_json_safe(content, {
        "bullets_to_rewrite": [],
        "skills_to_emphasize": [],
        "keywords_to_include": [],
        "guidance": "",
    })

    # If planner returned no bullets, default: suggest rewrites for first 3 bullets total
    roles = resume_structured.get("roles", []) or []
    if not data.get("bullets_to_rewrite") and roles:
        bullets_to_rewrite = []
        for ri, role in enumerate(roles[:2]):
            if len(bullets_to_rewrite) >= 3:
                break
            bullets = role.get("bullets", []) or []
            for bi in range(min(2, len(bullets))):
                if len(bullets_to_rewrite) >= 3:
                    break
                bullets_to_rewrite.append({
                    "role_index": ri,
                    "bullet_index": bi,
                    "reason": "Improve alignment with job description and impact wording.",
                })
        data["bullets_to_rewrite"] = bullets_to_rewrite
        if not data.get("guidance"):
            data["guidance"] = "Suggested rewrites for key experience bullets to better match the role."

    # Cap so the run finishes in reasonable time (each bullet = 1 LLM call)
    bullets_list = data.get("bullets_to_rewrite") or []
    if len(bullets_list) > 3:
        data["bullets_to_rewrite"] = bullets_list[:3]
    return data


def rewrite_bullets(
    *,
    llm: BaseChatModel,
    resume_structured: Dict[str, Any],
    jd_structured: Dict[str, Any],
    resume_facts: Dict[str, Any],
    rewrite_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Rewrite selected bullets while strictly respecting the resume facts.

    Enforces the 'No Hallucination Resume Rule':
    - Do NOT invent new companies, roles, dates, or tools beyond those already present.
    - If a JD keyword is missing, you may suggest 'Add <keyword> if accurate' separately,
      but DO NOT bake fictional experience into the bullet.
    """
    bullets_to_rewrite = (rewrite_plan.get("bullets_to_rewrite") or [])[:3]  # cap at 3 for speed
    roles = resume_structured.get("roles", []) or []

    allowed_companies = resume_facts.get("company_names", []) or []
    allowed_titles = resume_facts.get("titles", []) or []
    allowed_tools = resume_facts.get("tools", []) or []
    allowed_skills = resume_facts.get("skills", []) or []

    jd_title = jd_structured.get("title") or ""
    jd_company = jd_structured.get("company") or ""

    rewritten_entries: List[Dict[str, Any]] = []

    prompt = ChatPromptTemplate.from_template(
        """
You are rewriting a single resume bullet to better match a job description.

Rules (critical):
- You MUST NOT invent new companies, job titles, dates, or tools that are not in the allowed lists.
- You MUST stay faithful to the original achievement and responsibilities.
- You MAY sharpen impact (metrics, outcomes) ONLY if clearly implied by the original bullet.
- If a useful keyword from the job description is missing, you may add a short note in a separate
  field called "add_if_true_suggestion", e.g. "Consider adding Kubernetes if you have used it",
  but you MUST NOT claim experience that is not backed by the original bullet or facts.

Allowed facts:
- companies: {allowed_companies}
- titles: {allowed_titles}
- tools_or_technologies: {allowed_tools}
- skills: {allowed_skills}

Job title: "{jd_title}"
Job company: "{jd_company}"

Original bullet:
\"\"\"{original_bullet}\"\"\"

Job description (summary JSON):
```json
{jd_structured}
```

Rewrite this single bullet with:
- a strong action verb,
- clear outcome or metric when present,
- concise, ATS-friendly phrasing,
- keywords aligned with the JD **but only if consistent with the original content and allowed facts**.

Return STRICT JSON with keys:
- rewritten: string
- add_if_true_suggestion: string | null
"""
    )

    chain = prompt | llm

    import json

    for item in bullets_to_rewrite:
        role_index = item.get("role_index")
        bullet_index = item.get("bullet_index")
        reason = item.get("reason", "")

        try:
            role = roles[role_index]
            original_bullets = role.get("bullets", []) or []
            original_bullet = original_bullets[bullet_index]
        except Exception:
            continue

        raw = chain.invoke(
            {
                "allowed_companies": allowed_companies,
                "allowed_titles": allowed_titles,
                "allowed_tools": allowed_tools,
                "allowed_skills": allowed_skills,
                "jd_title": jd_title,
                "jd_company": jd_company,
                "original_bullet": original_bullet,
                "jd_structured": jd_structured,
            }
        )
        try:
            content = raw.content if hasattr(raw, "content") else str(raw)
            data = parse_json_safe(content, {"rewritten": "", "add_if_true_suggestion": None})
            rewritten_text = data.get("rewritten", "").strip()
            add_if_true = data.get("add_if_true_suggestion")
        except Exception:
            rewritten_text = str(original_bullet)
            add_if_true = None

        rewritten_entries.append(
            {
                "role_index": role_index,
                "bullet_index": bullet_index,
                "original": original_bullet,
                "rewritten": rewritten_text,
                "reason": reason,
                "add_if_true_suggestion": add_if_true,
            }
        )

    return {
        "bullets": rewritten_entries,
        "metadata": {
            "plan_summary": rewrite_plan.get("guidance", ""),
            "skills_to_emphasize": rewrite_plan.get("skills_to_emphasize", []),
            "keywords_to_include": rewrite_plan.get("keywords_to_include", []),
        },
    }


def evaluate_rewrites(
    *,
    llm: BaseChatModel,
    resume_facts: Dict[str, Any],
    rewritten_bullets: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate rewritten bullets for factuality, keyword inclusion, and ATS readability."""
    bullets = rewritten_bullets.get("bullets", []) or []

    prompt = ChatPromptTemplate.from_template(
        """
You are an evaluator for rewritten resume bullets.

Your tasks:
1. Check for factual consistency vs known resume facts:
   - No new companies, titles, dates, or tools beyond the allowed sets.
   - No claims of experience that are clearly unsupported by the original bullet.
2. Check whether the rewrites are:
   - clear, concise, and ATS-friendly (no heavy formatting, no long run-on sentences),
   - reasonably aligned with the intended keywords and skills.

Allowed facts (from the original resume):
```json
{resume_facts}
```

Rewritten bullets (each has original + rewritten + context):
```json
{rewritten_bullets}
```

Return STRICT JSON with keys:
- passed: bool                    # true only if bullets are factually safe and readable
- reasons: list[str]              # short explanations supporting the decision
- issues: {{
    hallucinated_companies: list[str],
    hallucinated_roles: list[str],
    hallucinated_tools: list[str],
    missing_keywords: list[str],  # important JD-aligned terms that are still missing
    ats_issues: list[str]         # e.g., "overly long bullet", "ambiguous phrasing"
  }}
"""
    )

    chain = prompt | llm
    raw = chain.invoke(
        {
            "resume_facts": resume_facts,
            "rewritten_bullets": bullets,
        }
    )
    content = raw.content if hasattr(raw, "content") else str(raw)
    default_issues = {
        "hallucinated_companies": [],
        "hallucinated_roles": [],
        "hallucinated_tools": [],
        "missing_keywords": [],
        "ats_issues": [],
    }
    data = parse_json_safe(content, {
        "passed": False,
        "reasons": ["Evaluation failed or JSON parse error."],
        "issues": default_issues,
    })
    if "issues" not in data or not isinstance(data["issues"], dict):
        data["issues"] = default_issues
    return data


def generate_cover_letter(
    *,
    llm: BaseChatModel,
    resume_structured: Dict[str, Any],
    jd_structured: Dict[str, Any],
    match_report: Dict[str, Any],
) -> str:
    """Generate a 1-page, tailored cover letter with real names (no placeholders)."""
    candidate_name = resume_structured.get("name") or "The Candidate"
    contact = resume_structured.get("contact") or {}
    candidate_email = contact.get("email") or "[email]"
    candidate_phone = contact.get("phone") or ""
    company_name = jd_structured.get("company") or "the company"
    job_title = jd_structured.get("title") or "the role"

    prompt = ChatPromptTemplate.from_template(
        """
You are writing a concise, professional cover letter for a specific role.

CRITICAL: Use these exact values—do NOT use placeholders like [Your Name] or [Company Name].
- Candidate name: {candidate_name}
- Candidate email: {candidate_email}
- Company name: {company_name}
- Job title: {job_title}

Use the resume, job description, and match report to produce a 3–5 paragraph cover letter tailored to this job.

Guidelines:
- Open with the candidate applying for the specific job title at the company name. Use the real candidate name in the sign-off.
- Highlight the strongest overlaps first (skills, technologies, domain).
- Honestly acknowledge one or two minor gaps and frame them as growth areas.
- Keep tone confident, specific, and human (no generic fluff).
- Avoid repeating the resume verbatim; instead, synthesize.
- Target ~400–600 words.
- Do NOT include [Your Name], [Company Name], [Job Title], [Your Email], [Date] or similar placeholders—use the values above.

Return ONLY the cover letter body text (you may start with "Dear Hiring Manager," and end with "Sincerely," and the candidate name).

Resume (JSON):
```json
{resume_structured}
```

Job description (JSON):
```json
{jd_structured}
```

Match report (JSON):
```json
{match_report}
```
"""
    )
    chain = prompt | llm
    raw = chain.invoke(
        {
            "candidate_name": candidate_name,
            "candidate_email": candidate_email,
            "company_name": company_name,
            "job_title": job_title,
            "resume_structured": resume_structured,
            "jd_structured": jd_structured,
            "match_report": match_report,
        }
    )
    return raw.content if hasattr(raw, "content") else str(raw)


def generate_interview_kit(
    *,
    llm: BaseChatModel,
    resume_structured: Dict[str, Any],
    jd_structured: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate an interview kit: questions + tailored answer guidance, specific to THIS job."""
    job_title = jd_structured.get("title") or "this role"
    company = jd_structured.get("company") or "the company"
    responsibilities = jd_structured.get("responsibilities") or []
    required_skills = jd_structured.get("required_skills") or jd_structured.get("technologies") or []

    prompt = ChatPromptTemplate.from_template(
        """
You are preparing an interview prep kit for a candidate applying to a SPECIFIC job.

TARGET JOB: **{job_title}** at **{company}**
Key responsibilities: {responsibilities_summary}
Required skills/tech: {skills_summary}

Your output MUST be tailored to THIS job. Do NOT give generic resume-based questions. Generate questions that an interviewer at this company for this role would actually ask, and outline answers that connect the candidate's resume to THIS role's responsibilities and stack.

Produce a JSON object with:
- tell_me_about_yourself: string   # 60–90 second pitch that ties the candidate's background to THIS role and company
- top_role_questions: list[{{ question: string, answer_outline: string }}]   # ~8–10 questions specific to {job_title} (e.g. domain, tools, responsibilities from the JD)
- project_deep_dives: list[{{ project_name: string, question: string, answer_outline: string }}]   # questions like "How does your experience on X relate to our need for Y?" using JD context
- behavioral_questions: list[{{ question: string, answer_outline: string }}]   # behavioral Qs that map to this role (e.g. collaboration if JD stresses teamwork)

Guidelines:
- Every question and answer outline must reference this role, company, or JD requirements. No generic "tell me about a project."
- Anchor answer outlines in the candidate's actual resume (roles, projects, metrics).
- Vary the kit when the job changes: different title/company/skills => different questions and talking points.

Return STRICT JSON ONLY.

Resume (JSON):
```json
{resume_structured}
```

Full job description (JSON):
```json
{jd_structured}
```
"""
    )
    responsibilities_summary = ", ".join(str(r) for r in responsibilities[:5]) if responsibilities else "See JD"
    skills_summary = ", ".join(str(s) for s in required_skills[:15]) if required_skills else "See JD"

    chain = prompt | llm
    raw = chain.invoke(
        {
            "job_title": job_title,
            "company": company,
            "responsibilities_summary": responsibilities_summary,
            "skills_summary": skills_summary,
            "resume_structured": resume_structured,
            "jd_structured": jd_structured,
        }
    )

    content = raw.content if hasattr(raw, "content") else str(raw)
    data = parse_json_safe(content, {
        "tell_me_about_yourself": "",
        "top_role_questions": [],
        "project_deep_dives": [],
        "behavioral_questions": [],
    })
    return data

