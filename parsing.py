from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from utils_llm import extract_json, parse_json_safe


def parse_job_description(llm: BaseChatModel, jd_text: str) -> Dict[str, Any]:
    """LLM-powered JD parser into a structured JSON schema."""
    prompt = ChatPromptTemplate.from_template(
        """
You are a job description parser. Given a raw job description, extract a concise, structured representation.

Return STRICT JSON with the following keys:
- title: string
- company: string | null
- responsibilities: list[str]
- required_skills: list[str]          # must-have hard or soft skills
- nice_to_have_skills: list[str]
- years_experience: int | null
- technologies: list[str]             # tech stack (languages, frameworks, tools)
- keywords: list[str]                 # additional important phrases
- requirements: list[{{ text: string }}]  # each bullet or requirement line

Only use information explicitly present in the job description.

Job description:
\"\"\"{jd_text}\"\"\"
"""
    )
    chain = prompt | llm
    raw = chain.invoke({"jd_text": jd_text})
    content = raw.content if hasattr(raw, "content") else str(raw)
    data = parse_json_safe(content, {
        "title": None,
        "company": None,
        "responsibilities": [],
        "required_skills": [],
        "nice_to_have_skills": [],
        "years_experience": None,
        "technologies": [],
        "keywords": [],
        "requirements": [],
    })
    return data


def parse_resume(llm: BaseChatModel, resume_text: str) -> Dict[str, Any]:
    """LLM-powered resume parser into structured JSON."""
    prompt = ChatPromptTemplate.from_template(
        """
You are a resume parser. Extract a structured JSON representation.

Return STRICT JSON with keys:
- name: string | null
- contact: {{
    email: string | null,
    phone: string | null,
    location: string | null,
    links: list[str]
  }}
- roles: list[{{
    title: string | null,
    company: string | null,
    location: string | null,
    start_date: string | null,
    end_date: string | null,
    current: bool | null,
    bullets: list[str]
  }}]
- skills: list[str]
- projects: list[{{ name: string, bullets: list[str] }}]
- education: list[{{ institution: string, degree: string | null, dates: string | null }}]

Important:
- Preserve bullet text as-is (do NOT rewrite or embellish).
- Do NOT invent tools, companies, or dates.
- If you are unsure, leave fields null or empty.

Resume:
\"\"\"{resume_text}\"\"\"
"""
    )
    chain = prompt | llm
    raw = chain.invoke({"resume_text": resume_text})
    content = raw.content if hasattr(raw, "content") else str(raw)
    data = parse_json_safe(content, {
        "name": None,
        "contact": {"email": None, "phone": None, "location": None, "links": []},
        "roles": [],
        "skills": [],
        "projects": [],
        "education": [],
    })
    return data


def extract_resume_facts(structured_resume: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a 'facts' map we can use for hallucination guardrails."""
    roles = structured_resume.get("roles", []) or []
    projects = structured_resume.get("projects", []) or []
    skills = structured_resume.get("skills", []) or []
    education = structured_resume.get("education", []) or []

    company_names: List[str] = []
    titles: List[str] = []
    tools: List[str] = []
    bullets: List[str] = []

    for r in roles:
        if r.get("company"):
            company_names.append(str(r["company"]))
        if r.get("title"):
            titles.append(str(r["title"]))
        for b in r.get("bullets", []) or []:
            if isinstance(b, str):
                bullets.append(b)

    for p in projects:
        for b in p.get("bullets", []) or []:
            if isinstance(b, str):
                bullets.append(b)

    # Treat skills list as tools/technologies we must not exceed
    tools = [str(s) for s in skills]

    fact_map: Dict[str, Any] = {
        "company_names": sorted(set(company_names)),
        "titles": sorted(set(titles)),
        "tools": sorted(set(tools)),
        "skills": sorted(set(skills)),
        "education_institutions": sorted(
            {e.get("institution") for e in education if e.get("institution")}
        ),
        "raw_bullets": bullets,
    }
    return fact_map

