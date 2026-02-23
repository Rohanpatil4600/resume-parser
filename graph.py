from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI

from tools import (
    extract_pdf_text,
    fetch_jd_from_url,
    extract_keywords,
    compute_match_score,
)
from parsing import parse_resume, parse_job_description, extract_resume_facts
from utils_llm import parse_json_safe
from rewriting import (
    plan_rewrites,
    rewrite_bullets,
    evaluate_rewrites,
    generate_cover_letter,
    generate_interview_kit,
)


class EngineState(TypedDict, total=False):
    # raw inputs
    resume_text: str
    jd_text: str

    # structured representations
    resume_structured: Dict[str, Any]
    jd_structured: Dict[str, Any]
    resume_facts: Dict[str, Any]

    # analysis / scoring
    match_report: Dict[str, Any]
    rewrite_plan: Dict[str, Any]
    rewritten_bullets: Dict[str, Any]
    final_outputs: Dict[str, Any]

    # control / meta
    errors: List[str]
    thread_id: str


def _get_llm() -> ChatOpenAI:
    # Single place to configure the LLM backend (timeout so one slow call doesn't hang the run)
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, request_timeout=90)


def ingest_node(state: EngineState) -> EngineState:
    """Normalize input sources and ensure we have resume_text and jd_text."""
    new_state: EngineState = dict(state)
    errors: List[str] = list(new_state.get("errors", []))

    resume_text = new_state.get("resume_text", "") or ""
    jd_text = new_state.get("jd_text", "") or ""

    # Nothing to do if already plain text
    if not resume_text:
        errors.append("Missing resume text.")

    if not jd_text:
        errors.append("Missing job description text.")

    new_state["errors"] = errors
    return new_state


def jd_parser_node(state: EngineState) -> EngineState:
    new_state: EngineState = dict(state)
    jd_text = new_state.get("jd_text", "") or ""
    llm = _get_llm()
    new_state["jd_structured"] = parse_job_description(llm, jd_text)
    return new_state


def resume_parser_node(state: EngineState) -> EngineState:
    new_state: EngineState = dict(state)
    resume_text = new_state.get("resume_text", "") or ""
    llm = _get_llm()
    structured = parse_resume(llm, resume_text)
    new_state["resume_structured"] = structured
    new_state["resume_facts"] = extract_resume_facts(structured)
    return new_state


def parse_both_node(state: EngineState) -> EngineState:
    """Run JD and resume parsing in parallel to cut wall-clock time."""
    new_state: EngineState = dict(state)
    jd_text = new_state.get("jd_text", "") or ""
    resume_text = new_state.get("resume_text", "") or ""
    llm = _get_llm()

    jd_result: Dict[str, Any] = {}
    resume_result: Dict[str, Any] = {}
    resume_facts_result: Dict[str, Any] = {}
    jd_error: Optional[Exception] = None
    resume_error: Optional[Exception] = None

    def do_jd():
        nonlocal jd_result, jd_error
        try:
            jd_result = parse_job_description(llm, jd_text)
        except Exception as e:
            jd_error = e

    def do_resume():
        nonlocal resume_result, resume_facts_result, resume_error
        try:
            resume_result = parse_resume(llm, resume_text)
            resume_facts_result = extract_resume_facts(resume_result)
        except Exception as e:
            resume_error = e

    t1 = threading.Thread(target=do_jd)
    t2 = threading.Thread(target=do_resume)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if jd_error:
        new_state.setdefault("errors", [])
        new_state["errors"].append(f"JD parse: {jd_error}")
    else:
        new_state["jd_structured"] = jd_result
    if resume_error:
        new_state.setdefault("errors", [])
        new_state["errors"].append(f"Resume parse: {resume_error}")
    else:
        new_state["resume_structured"] = resume_result
        new_state["resume_facts"] = resume_facts_result

    return new_state


def matcher_node(state: EngineState) -> EngineState:
    new_state: EngineState = dict(state)
    resume_structured = new_state.get("resume_structured", {}) or {}
    jd_structured = new_state.get("jd_structured", {}) or {}

    # keyword extraction on raw texts
    resume_text = new_state.get("resume_text", "") or ""
    jd_text = new_state.get("jd_text", "") or ""
    jd_keywords, resume_keywords = extract_keywords(jd_text, resume_text)

    report = compute_match_score(
        resume_structured=resume_structured,
        jd_structured=jd_structured,
        resume_keywords=resume_keywords,
        jd_keywords=jd_keywords,
    )
    new_state["match_report"] = report
    return new_state


def gap_analyzer_node(state: EngineState) -> EngineState:
    new_state: EngineState = dict(state)
    llm = _get_llm()
    resume_structured = new_state.get("resume_structured", {}) or {}
    jd_structured = new_state.get("jd_structured", {}) or {}
    match_report = new_state.get("match_report", {}) or {}

    # Use the LLM to produce a lightweight gap analysis and suggested quick wins.
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        """
You are a resume/job match analyst.

Given this structured resume, job description, and match report, identify:
- the most important missing or weak must-have skills
- which resume sections should be updated
- 3–7 very concrete \"quick wins\" (small edits that improve alignment)

Return STRICT JSON with keys:
- missing_must_haves: list[str]
- sections_to_update: list[str]
- quick_wins: list[str]

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
    gap_info = parse_json_safe(content, {
        "missing_must_haves": [],
        "sections_to_update": [],
        "quick_wins": [],
    })

    # Deterministic fallback when LLM returns empty (so Match report / Gaps are never blank)
    skills = match_report.get("skills") or {}
    jd_must = skills.get("must_have_list") or []
    must_overlap = skills.get("must_have_overlap") or []
    if not gap_info.get("missing_must_haves") and jd_must:
        missing = [s for s in jd_must if s not in must_overlap][:15]
        if missing:
            gap_info["missing_must_haves"] = missing
    if not gap_info.get("sections_to_update"):
        gap_info["sections_to_update"] = ["Experience", "Skills"]
    if not gap_info.get("quick_wins"):
        gap_info["quick_wins"] = [
            "Add keywords from the job description where they fit your experience.",
            "Put your most relevant role or project first.",
            "Include metrics (numbers, impact) in bullet points.",
        ]

    new_state.setdefault("match_report", {})
    new_state["match_report"]["gap_analysis"] = gap_info
    return new_state


def recruiter_perspective_node(state: EngineState) -> EngineState:
    """Generate a short 'If I were the hiring manager…' paragraph for explainability."""
    new_state: EngineState = dict(state)
    llm = _get_llm()
    resume_structured = new_state.get("resume_structured", {}) or {}
    jd_structured = new_state.get("jd_structured", {}) or {}
    match_report = new_state.get("match_report", {}) or {}

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        """
You are simulating a hiring manager reviewing this candidate for this job.

In 2–4 sentences, write a short paragraph that starts with "If I were the hiring manager…" and covers:
- How the candidate likely comes across (strengths that stand out)
- The main concern or gap, if any (be direct but fair)
- Whether you would shortlist for an interview and why or why not

Be specific to this resume and this job. No generic fluff.

Resume (summary):
```json
{resume_structured}
```

Job (summary):
```json
{jd_structured}
```

Match report (score and risk):
```json
{match_report}
```

Return ONLY the paragraph, no heading or bullet points.
"""
    )
    chain = prompt | llm
    raw = chain.invoke(
        {
            "resume_structured": resume_structured,
            "jd_structured": jd_structured,
            "match_report": {k: v for k, v in match_report.items() if k in ("overall_score", "risk_summary", "risk_counts", "keyword_coverage", "requirements_matrix")},
        }
    )
    paragraph = (raw.content if hasattr(raw, "content") else str(raw)).strip()
    new_state.setdefault("match_report", {})
    new_state["match_report"]["recruiter_perspective"] = paragraph
    return new_state


def rewrite_planner_node(state: EngineState) -> EngineState:
    new_state: EngineState = dict(state)
    llm = _get_llm()
    new_state["rewrite_plan"] = plan_rewrites(
        llm=llm,
        resume_structured=new_state.get("resume_structured", {}) or {},
        jd_structured=new_state.get("jd_structured", {}) or {},
        match_report=new_state.get("match_report", {}) or {},
    )
    return new_state


def bullet_rewriter_node(state: EngineState) -> EngineState:
    new_state: EngineState = dict(state)
    llm = _get_llm()
    new_state["rewritten_bullets"] = rewrite_bullets(
        llm=llm,
        resume_structured=new_state.get("resume_structured", {}) or {},
        jd_structured=new_state.get("jd_structured", {}) or {},
        resume_facts=new_state.get("resume_facts", {}) or {},
        rewrite_plan=new_state.get("rewrite_plan", {}) or {},
    )
    return new_state


def evaluator_node(state: EngineState) -> EngineState:
    new_state: EngineState = dict(state)
    llm = _get_llm()
    eval_result = evaluate_rewrites(
        llm=llm,
        resume_facts=new_state.get("resume_facts", {}) or {},
        rewritten_bullets=new_state.get("rewritten_bullets", {}) or {},
    )
    new_state.setdefault("final_outputs", {})
    new_state["final_outputs"]["rewrite_evaluation"] = eval_result
    return new_state


def cover_letter_node(state: EngineState) -> EngineState:
    new_state: EngineState = dict(state)
    llm = _get_llm()
    cover_letter = generate_cover_letter(
        llm=llm,
        resume_structured=new_state.get("resume_structured", {}) or {},
        jd_structured=new_state.get("jd_structured", {}) or {},
        match_report=new_state.get("match_report", {}) or {},
    )
    new_state.setdefault("final_outputs", {})
    new_state["final_outputs"]["cover_letter"] = cover_letter
    return new_state


def interview_kit_node(state: EngineState) -> EngineState:
    new_state: EngineState = dict(state)
    llm = _get_llm()
    interview_kit = generate_interview_kit(
        llm=llm,
        resume_structured=new_state.get("resume_structured", {}) or {},
        jd_structured=new_state.get("jd_structured", {}) or {},
    )
    new_state.setdefault("final_outputs", {})
    new_state["final_outputs"]["interview_kit"] = interview_kit
    return new_state


def _should_rewrite_again(state: EngineState) -> str:
    """Routing: always proceed to cover letter (skip retry loop to stay within timeout)."""
    return "done"


def build_graph() -> CompiledStateGraph:
    """Build and compile the LangGraph state graph for the engine."""
    workflow: StateGraph = StateGraph(EngineState)

    workflow.add_node("ingest", ingest_node)
    workflow.add_node("parse_both", parse_both_node)
    workflow.add_node("matcher", matcher_node)
    workflow.add_node("gap_analyzer", gap_analyzer_node)
    workflow.add_node("recruiter_perspective", recruiter_perspective_node)
    workflow.add_node("rewrite_planner", rewrite_planner_node)
    workflow.add_node("bullet_rewriter", bullet_rewriter_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("cover_letter", cover_letter_node)
    workflow.add_node("interview_kit", interview_kit_node)

    workflow.set_entry_point("ingest")

    # Ingest → parallel parse (JD + resume) → matcher
    workflow.add_edge("ingest", "parse_both")
    workflow.add_edge("parse_both", "matcher")

    workflow.add_edge("matcher", "gap_analyzer")
    workflow.add_edge("gap_analyzer", "recruiter_perspective")
    workflow.add_edge("recruiter_perspective", "rewrite_planner")
    workflow.add_edge("rewrite_planner", "bullet_rewriter")
    workflow.add_edge("bullet_rewriter", "evaluator")

    workflow.add_conditional_edges(
        "evaluator",
        _should_rewrite_again,
        {
            "rewrite_again": "bullet_rewriter",
            "done": "cover_letter",
        },
    )

    workflow.add_edge("cover_letter", "interview_kit")
    workflow.add_edge("interview_kit", END)

    return workflow.compile()


graph: CompiledStateGraph = build_graph()

# Friendly labels for progress (internal node name → what to show)
STEP_LABELS = {
    "ingest": "Validating input",
    "parse_both": "Parsing JD & resume",
    "matcher": "Computing match score",
    "gap_analyzer": "Finding gaps",
    "recruiter_perspective": "Recruiter view",
    "rewrite_planner": "Planning bullet rewrites",
    "bullet_rewriter": "Rewriting bullets",
    "evaluator": "Checking quality",
    "cover_letter": "Writing cover letter",
    "interview_kit": "Building interview kit",
}


def run_engine(
    thread_id: str,
    resume_text: str,
    jd_text: str,
) -> EngineState:
    """Run the graph end-to-end and return final state."""
    initial: EngineState = {
        "thread_id": thread_id,
        "resume_text": resume_text,
        "jd_text": jd_text,
    }
    return graph.invoke(initial, config={"configurable": {"thread_id": thread_id}})


def run_engine_stream(thread_id: str, resume_text: str, jd_text: str):
    """Run the graph and yield (step_label, state) after each node so the UI can show progress.
    The last yielded state is the final result.
    """
    initial: EngineState = {
        "thread_id": thread_id,
        "resume_text": resume_text,
        "jd_text": jd_text,
    }
    config = {"configurable": {"thread_id": thread_id}}
    state: EngineState = dict(initial)
    for event in graph.stream(initial, config=config, stream_mode="updates"):
        for node_name, update in event.items():
            if isinstance(update, dict):
                state = {**state, **update}
            label = STEP_LABELS.get(node_name, node_name)
            yield label, state

