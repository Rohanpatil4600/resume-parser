"""
AI Resume + Job Match Intelligence Engine — Streamlit UI.
Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import date
from pathlib import Path

import streamlit as st

# Load env before any LangChain/OpenAI calls
from dotenv import load_dotenv
load_dotenv()

from tools import extract_pdf_text, fetch_jd_from_url, hash_text
from graph import run_engine_stream
import database


def _cover_letter_header(resume_structured: dict) -> str:
    """Build name, location, email, date block from resume (before Dear Hiring Manager)."""
    name = resume_structured.get("name") or "Your Name"
    contact = resume_structured.get("contact") or {}
    email = contact.get("email") or "your.email@example.com"
    location = contact.get("location") or "City, State"
    today = date.today().strftime("%B %d, %Y")
    return f"{name}\n{location}\n{email}\n{today}"


def _cover_letter_to_pdf(full_text: str) -> bytes:
    """Return PDF bytes for the cover letter text (wrapped)."""
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=11, leading=14)

    parts = []
    for para in full_text.replace("\r\n", "\n").split("\n\n"):
        para = para.strip()
        if not para:
            parts.append(Spacer(1, 12))
            continue
        # Escape & < > for XML; keep line breaks (e.g. header: name, location, email, date)
        para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        parts.append(Paragraph(para, body_style))
        parts.append(Spacer(1, 8))
    doc.build(parts)
    buf.seek(0)
    return buf.read()


def _sanitize_filename(company: str) -> str:
    """Safe filename from company name."""
    if not company or not str(company).strip():
        return "Cover_Letter"
    s = re.sub(r'[<>:"/\\|?*]', "", str(company).strip())
    return s[:80] or "Cover_Letter"

st.set_page_config(page_title="Resume × Job Match", layout="wide")

# Session state
if "result" not in st.session_state:
    st.session_state.result = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "approved_rewrites" not in st.session_state:
    st.session_state.approved_rewrites = {}


def ensure_resume_text(resume_text: str, uploaded_file) -> str:
    if uploaded_file is not None:
        return extract_pdf_text(uploaded_file.read())
    return (resume_text or "").strip()


# Max chars to send to the engine (keeps runs fast; job listing URLs can return huge pages)
MAX_JD_CHARS = 6000
MAX_RESUME_CHARS = 8000


def ensure_jd_text(jd_text: str, jd_url: str) -> str:
    text = (jd_text or "").strip()
    if jd_url and jd_url.strip().startswith("http"):
        try:
            fetched = fetch_jd_from_url(jd_url.strip())
            if fetched:
                text = fetched.strip()
        except Exception as e:
            st.warning(f"Could not fetch URL: {e}. Using pasted text if any.")
    if len(text) > MAX_JD_CHARS:
        text = text[:MAX_JD_CHARS] + "\n\n[Text truncated for faster analysis.]"
    return text


# ----- Sidebar -----
with st.sidebar:
    st.title("Resume × Job Match")
    st.caption("Upload resume, paste or link JD, then Analyze.")

    resume_option = st.radio("Resume input", ["Upload PDF", "Paste text"], horizontal=True)
    resume_text_paste = ""
    resume_file = None
    if resume_option == "Upload PDF":
        resume_file = st.file_uploader("Resume PDF", type=["pdf"])
    else:
        resume_text_paste = st.text_area("Paste resume text", height=120, placeholder="Paste your resume content…")

    st.divider()
    jd_option = st.radio("Job description", ["Paste text", "URL"], horizontal=True)
    jd_text_paste = ""
    jd_url = ""
    if jd_option == "URL":
        jd_url = st.text_input("Job description URL", placeholder="https://…")
        st.caption("Use the **single job posting** URL, not the jobs listing page, for faster runs.")
        jd_text_paste = st.text_area("Or paste JD (overrides URL if not empty)", height=100, placeholder="Optional…")
    else:
        jd_text_paste = st.text_area("Paste job description", height=150, placeholder="Paste the job description…")

    company = st.text_input("Company (for saving)", placeholder="Acme Inc.")
    role = st.text_input("Role (for saving)", placeholder="Senior Engineer")

    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Saved applications")
    database.init_db()
    apps = database.list_applications()
    if not apps:
        st.caption("No saved applications yet.")
    else:
        for app in apps:
            tid = app["thread_id"]
            label = f"{app.get('company') or 'Unknown'} — {app.get('role') or 'N/A'}"
            if st.button(label, key=f"load_{tid}", use_container_width=True):
                st.session_state.select_thread = tid
                st.rerun()

# ----- Main area -----
if analyze_clicked:
    resume_text = ensure_resume_text(resume_text_paste, resume_file)
    jd_text = ensure_jd_text(jd_text_paste, jd_url)

    if not resume_text:
        st.error("Provide resume: upload a PDF or paste text.")
        st.stop()
    if not jd_text:
        st.error("Provide job description: paste text or a valid URL.")
        st.stop()

    if len(resume_text) > MAX_RESUME_CHARS:
        resume_text = resume_text[:MAX_RESUME_CHARS] + "\n\n[Resume truncated.]"

    thread_id = str(uuid.uuid4())
    st.session_state.thread_id = thread_id
    st.session_state.last_resume_text = resume_text
    st.session_state.last_jd_text = jd_text

    st.info("Usually takes **2–5 minutes**. For best speed: paste the **job description text** or use a single job posting URL (not a listing).")

    progress_placeholder = st.empty()
    steps_done: list[str] = []
    result = None

    try:
        for step_label, state in run_engine_stream(thread_id=thread_id, resume_text=resume_text, jd_text=jd_text):
            steps_done.append(step_label)
            # Show current step and completed steps so you can see what's going on
            progress_placeholder.markdown(
                "**Current:** " + step_label + "\n\n"
                + "**Done:** " + (", ".join(steps_done[:-1]) if len(steps_done) > 1 else "—")
            )
            result = state
        progress_placeholder.markdown("**Done.** Completed: " + ", ".join(steps_done))
    except Exception as e:
        progress_placeholder.empty()
        st.error("Engine error: " + str(e))
        st.stop()

    if not result:
        st.warning("No result. Please try again.")
        st.stop()

    st.session_state.result = result

# Load saved thread from sidebar (optional: persist full result in DB or in-memory cache; here we only list, no load of past result)
if st.session_state.get("select_thread") and not st.session_state.result:
    st.info("Saved applications list only. Run Analyze for a new application to see results here.")

# Display result
result = st.session_state.result
if result is None:
    st.info("Upload a resume and job description, then click **Analyze** to see your match report, gaps, rewrites, cover letter, and interview kit.")
    st.stop()

# Tabs
match_report = result.get("match_report") or {}
final_outputs = result.get("final_outputs") or {}
rewritten_bullets = result.get("rewritten_bullets") or {}
rewrite_plan = result.get("rewrite_plan") or {}

t1, t2, t3, t4, t5, t6 = st.tabs([
    "Match report",
    "Gaps",
    "Bullet rewrites",
    "Tailored section",
    "Cover letter",
    "Interview kit",
])

with t1:
    score = match_report.get("overall_score", 0) or 0
    if isinstance(score, (int, float)):
        st.metric("Match score", f"{score:.0f}/100")
    breakdown = match_report.get("breakdown") or {}
    if breakdown:
        st.caption("Skills component: " + str(breakdown.get("skills_component", 0)) + " | Keywords: " + str(breakdown.get("keywords_component", 0)))

    # Must-have risk level
    risk_emoji = match_report.get("risk_emoji") or "🟡"
    risk_summary = match_report.get("risk_summary") or ""
    if risk_summary:
        st.markdown(f"**Risk:** {risk_emoji} {risk_summary}")

    # Keyword coverage (ATS-style)
    kw_cov = match_report.get("keyword_coverage") or {}
    matched = kw_cov.get("matched_count", 0)
    total = kw_cov.get("total_jd_keywords", 0) or 1
    missing_critical = kw_cov.get("missing_critical") or []
    st.subheader("Keyword coverage (ATS-style)")
    st.markdown(f"**Matched:** {matched} / {total} key terms ({kw_cov.get('matched_pct', 0)}%)")
    if missing_critical:
        st.caption("Critical missing: " + ", ".join(missing_critical[:10]))

    # Requirement → Evidence matrix (table with status)
    req_matrix = match_report.get("requirements_matrix") or []
    st.subheader("Requirement → evidence matrix")
    if req_matrix:
        table_rows = []
        for row in req_matrix:
            req = row.get("requirement", "")[:80] + ("…" if len(row.get("requirement", "")) > 80 else "")
            status = row.get("status", "Missing")
            if status == "Strong":
                status_badge = "✅ Strong"
            elif status == "Weak":
                status_badge = "⚠️ Weak"
            else:
                status_badge = "❌ Missing"
            evidence_summary = (row.get("evidence_summary") or "—").replace("|", " ")[:180] + ("…" if len(row.get("evidence_summary") or "") > 180 else "")
            table_rows.append((req, evidence_summary, status_badge))
        st.table([("JD requirement", "Evidence from resume", "Status")] + table_rows)
        st.caption("Expand evidence: below are the resume bullets that support each requirement.")
        for i, row in enumerate(req_matrix):
            ev_list = row.get("evidence", [])
            if len(ev_list) > 0:
                with st.expander(f"Evidence for: {(row.get('requirement') or '')[:50]}…"):
                    for ev in ev_list[:3]:
                        st.caption(f"**{ev.get('role')}** @ {ev.get('company', '')}")
                        st.write((ev.get("bullet") or "")[:400] + ("…" if len(ev.get("bullet", "")) > 400 else ""))
    else:
        st.caption("No requirements parsed from the job description.")

    # Recruiter perspective
    recruiter = match_report.get("recruiter_perspective", "")
    if recruiter:
        st.subheader("Recruiter perspective")
        st.markdown("*If I were the hiring manager…*")
        st.info(recruiter)

with t2:
    gap = (match_report.get("gap_analysis") or {})
    st.subheader("Missing must-haves")
    for x in gap.get("missing_must_haves", []):
        st.markdown(f"- {x}")
    st.subheader("Sections to update")
    for x in gap.get("sections_to_update", []):
        st.markdown(f"- {x}")
    st.subheader("Quick wins")
    for x in gap.get("quick_wins", []):
        st.markdown(f"- {x}")

with t3:
    bullets = rewritten_bullets.get("bullets", [])
    for i, entry in enumerate(bullets):
        with st.expander(f"Bullet {i+1}: { (entry.get('original') or '')[:60]}…"):
            st.markdown("**Original:**")
            st.write(entry.get("original", ""))
            st.markdown("**Suggested:**")
            st.write(entry.get("rewritten", ""))
            if entry.get("add_if_true_suggestion"):
                st.caption(f"Add if true: {entry['add_if_true_suggestion']}")
            key_approve = f"approve_{i}"
            key_reject = f"reject_{i}"
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve", key=key_approve):
                    st.session_state.approved_rewrites[i] = entry.get("rewritten", "")
                    st.rerun()
            with col2:
                if st.button("Reject", key=key_reject):
                    st.session_state.approved_rewrites[i] = None
                    st.rerun()
            if i in st.session_state.approved_rewrites and st.session_state.approved_rewrites[i]:
                st.success("Approved")

with t4:
    st.caption("Copy-ready tailored bullets (approved only).")
    approved = [st.session_state.approved_rewrites.get(i) for i in range(len(bullets)) if st.session_state.approved_rewrites.get(i)]
    if approved:
        for a in approved:
            st.markdown(f"- {a}")
    else:
        st.info("Approve rewrites in the previous tab to see them here.")

with t5:
    cover_body = final_outputs.get("cover_letter", "")
    resume_structured = result.get("resume_structured") or {}
    jd_structured = result.get("jd_structured") or {}

    header = _cover_letter_header(resume_structured)
    st.text(header)
    st.markdown("---")
    st.markdown(cover_body if cover_body else "_No cover letter generated._")

    if cover_body:
        full_cover = header + "\n\n" + cover_body
        company_name = jd_structured.get("company") or "Cover_Letter"
        pdf_filename = _sanitize_filename(company_name) + ".pdf"
        try:
            pdf_bytes = _cover_letter_to_pdf(full_cover)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                key="download_cover_pdf",
            )
        except Exception as e:
            st.caption(f"PDF download unavailable: {e}")

with t6:
    kit = final_outputs.get("interview_kit", {})
    if isinstance(kit, str):
        st.markdown(kit)
    else:
        st.markdown("**Tell me about yourself**")
        st.write(kit.get("tell_me_about_yourself", ""))
        st.subheader("Top role questions")
        for q in kit.get("top_role_questions", [])[:10]:
            st.markdown(f"- **{q.get('question', '')}**")
            st.caption(q.get("answer_outline", ""))
        st.subheader("Project deep dives")
        for p in kit.get("project_deep_dives", []):
            st.markdown(f"- **{p.get('project_name', '')}**: {p.get('question', '')}")
            st.caption(p.get("answer_outline", ""))
        st.subheader("Behavioral")
        for b in kit.get("behavioral_questions", []):
            st.markdown(f"- **{b.get('question', '')}**")
            st.caption(b.get("answer_outline", ""))

# Save application
st.divider()
col_save, _ = st.columns(2)
with col_save:
    if st.button("Save application"):
        tid = st.session_state.thread_id
        if tid:
            database.save_application(
                thread_id=tid,
                company=company or None,
                role=role or None,
                jd_url=jd_url if jd_option == "URL" and jd_url else None,
            )
            resume_text_final = getattr(st.session_state, "last_resume_text", "") or ensure_resume_text(resume_text_paste, resume_file)
            jd_text_final = getattr(st.session_state, "last_jd_text", "") or ensure_jd_text(jd_text_paste, jd_url)
            database.save_run(
                thread_id=tid,
                resume_hash=hash_text(resume_text_final),
                jd_hash=hash_text(jd_text_final),
                match_score=float(match_report.get("overall_score", 0) or 0),
                report_json=match_report,
                rewrites_json=rewritten_bullets,
            )
            st.success("Application saved.")
        else:
            st.warning("Run Analyze first.")
