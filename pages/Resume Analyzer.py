
from typing import Tuple, Dict, Any
import joblib
import difflib
import json
import os
from pathlib import Path

import streamlit as st
import pdfplumber
import docx
import re
from io import BytesIO

# List of target skills (expanded)
COMMON_SKILLS = [
    "python", "java", "sql", "excel", "power bi", "tableau", "aws", "azure",
    "machine learning", "deep learning", "flask", "django", "react", "node.js", "node", "javascript", "js", "html", "css",
    "data analysis", "pandas", "numpy", "git", "docker", "linux", "spring", "powerbi", "spark", "kubernetes"
]

# Role mapping rules (simple)
ROLE_RULES = [
    ("Data Scientist", ["python", "sql", "pandas", "machine learning"]),
    ("Data Analyst", ["python", "sql", "power bi", "excel", "tableau"]),
    ("Backend Developer", ["java", "spring", "sql"]),
    ("Full Stack Developer", ["react", "node.js", "javascript"]),
    ("DevOps Engineer", ["aws", "docker", "linux", "kubernetes"]),
    ("Software Developer", ["python", "java", "git", "docker"]),
    ("Frontend Developer", ["react", "javascript", "css", "html"]),
    ("SQL Developer", ["sql", "excel"]),
]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return "\n".join(text)


def extract_text_from_docx(file_bytes: bytes) -> str:
    text = []
    try:
        doc = docx.Document(BytesIO(file_bytes))
        for para in doc.paragraphs:
            if para.text:
                text.append(para.text)
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return "\n".join(text)


def extract_text(uploaded_file) -> str:
    """Dispatch to the correct extractor based on file type."""
    if uploaded_file is None:
        return ""
    content = uploaded_file.read()
    type_lower = uploaded_file.name.lower()
    if type_lower.endswith('.pdf'):
        return extract_text_from_pdf(content)
    elif type_lower.endswith('.docx') or type_lower.endswith('.doc'):
        return extract_text_from_docx(content)
    else:
        st.error("Unsupported file type. Please upload PDF or DOCX.")
        return ""


def extract_skills(resume_text: str) -> list:
    """Simple keyword matching for skill extraction. Returns normalized skill names found."""
    found = set()
    txt = resume_text.lower()

    # Normalize some variants
    txt = txt.replace('\r', ' ')

    for skill in COMMON_SKILLS:
        # Use word boundaries for most, but allow '.' in node.js variations
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, txt):
            # normalize some names
            if skill == 'powerbi':
                found.add('power bi')
            elif skill == 'node':
                found.add('node.js')
            else:
                found.add(skill)

    return sorted(found)


def predict_role(skills: list) -> Tuple[str, float, Dict[str, Any]]:
    """Predict a role from skills using simple matching rules.
    Returns (role_name, confidence_percentage, details)
    details contains matched counts and matched skills per role to help debug decisions.
    """
    details: Dict[str, Any] = {}
    if not skills:
        return ("Unknown", 0.0, {"reason": "no skills detected"})

    skills_set = set(skills)
    best_role = "Unknown"
    best_score = -1.0
    best_matched_count = 0

    # For each role, compute fraction of rule skills matched and absolute matched count
    for role, rule_skills in ROLE_RULES:
        rule_set = set(rule_skills)
        matched = sorted(list(skills_set.intersection(rule_set)))
        matched_count = len(matched)
        score = matched_count / len(rule_set) if rule_set else 0
        details[role] = {"rule_skills": rule_skills, "matched": matched, "matched_count": matched_count, "score": round(score, 3)}

        # Choose by higher score, tie-breaker: higher matched_count, then smaller rule_set (more specific)
        if score > best_score or (score == best_score and matched_count > best_matched_count) or (score == best_score and matched_count == best_matched_count and len(rule_set) < len(next(r for r in ROLE_RULES if r[0] == best_role)[1]) if best_role != "Unknown" else False):
            best_score = score
            best_role = role
            best_matched_count = matched_count

    confidence = round(best_score * 100, 1) if best_score >= 0 else 0.0
    return (best_role, confidence, details)


def resume_page():
    st.title("📄 Resume Analyzer")
    st.markdown("Upload your resume (PDF or DOCX). The app will extract skills and suggest a job role.")

    uploaded = st.file_uploader("Upload Resume", type=["pdf", "docx", "doc"])

    if uploaded is not None:
        st.info(f"Uploaded: {uploaded.name}")
        raw_text = extract_text(uploaded)
        if not raw_text:
            st.warning("No text could be extracted from the uploaded file.")
            return

        st.success("Resume uploaded successfully.")

        with st.expander("Extracted Text (first 1000 chars)"):
            st.text(raw_text[:1000])

        skills = extract_skills(raw_text)
        role, confidence, details = predict_role(skills)

        st.subheader("Extracted Skills")
        if skills:
            st.write(", ".join(skills))
        else:
            st.write("No known skills detected.")

        st.subheader("Predicted Job Role")
        st.write(f"**{role}** — Confidence: {confidence}%")

        # Show per-role details to help distinguish why a role was chosen
        with st.expander("Why this role? (details)"):
            for r, info in details.items():
                st.write(f"- {r}: matched {info['matched_count']} / {len(info['rule_skills'])} -> {', '.join(info['matched']) if info['matched'] else 'none'}")

        # Allow user to pick an alternate role if desired
        all_suggested = [r for r in details.keys()]
        chosen = st.selectbox("Choose role to use (predicted is selected)", options=all_suggested, index=all_suggested.index(role) if role in all_suggested else 0)

        # Try to map chosen role to existing dropdown options for exact or fuzzy match
        mapped_label = None
        job_options = []
        try:
            dropdowns = joblib.load("dropdown_options.pkl")
            job_options = dropdowns.get("job_title", [])

            # Load persisted role mappings if present
            mappings_file = Path("role_mappings.json")
            if mappings_file.exists():
                try:
                    with mappings_file.open("r", encoding="utf-8") as f:
                        role_mappings = json.load(f)
                except Exception:
                    role_mappings = {}
            else:
                role_mappings = {}

            # If user previously mapped this analyzer role to a dropdown value, use it
            if chosen in role_mappings:
                mapped_label = role_mappings[chosen]
            else:
                # exact match first
                if chosen in job_options:
                    mapped_label = chosen
                else:
                    # use difflib get_close_matches on job_options
                    matches = difflib.get_close_matches(chosen, job_options, n=1, cutoff=0.6)
                    if matches:
                        mapped_label = matches[0]
                    else:
                        # fallback: substring match (case-insensitive)
                        for opt in job_options:
                            if chosen.lower() in opt.lower() or opt.lower() in chosen.lower():
                                mapped_label = opt
                                break
        except Exception:
            job_options = []

        if mapped_label:
            st.write(f"Mapped to dropdown option: **{mapped_label}**")
        else:
            st.write("No exact dropdown mapping found for this role.")
            # If job_options are available, allow the user to pick the correct one and save mapping
            if job_options:
                pick = st.selectbox("Pick the correct job title from the app dropdown options", options=job_options)
                if st.button("Save mapping for this analyzer role"):
                    # load existing mappings, update and save
                    mappings_file = Path("role_mappings.json")
                    if mappings_file.exists():
                        try:
                            with mappings_file.open("r", encoding="utf-8") as f:
                                role_mappings = json.load(f)
                        except Exception:
                            role_mappings = {}
                    else:
                        role_mappings = {}

                    role_mappings[chosen] = pick
                    try:
                        with mappings_file.open("w", encoding="utf-8") as f:
                            json.dump(role_mappings, f, indent=2, ensure_ascii=False)
                        st.success(f"Saved mapping: '{chosen}' -> '{pick}'")
                        mapped_label = pick
                    except Exception as e:
                        st.error(f"Failed to save mapping: {e}")
            else:
                st.info("The app's job dropdown has no options configured, so mapping to an existing option isn't possible. You can still use the predicted role as-is.")

        # Show top 3 candidate roles with scores to help choose
        try:
            # build list of (role, score) sorted desc
            scored = [(r, info.get('score', 0)) for r, info in details.items()]
            scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
            top3 = scored_sorted[:3]
            if top3:
                st.subheader("Top candidates")
                for r, s in top3:
                    st.write(f"{r}: {round(s*100,1) if s<=1 else s}%")
        except Exception:
            pass

        # Allow user to persist the mapped label into dropdown_options.pkl (with backup)
        if mapped_label:
            st.markdown("---")
            st.info("If you'd like this mapped label to appear in the app dropdowns for future sessions, you can append it to the saved dropdown options. This will create a backup of the previous file.")
            if st.button("Append mapped label to app dropdown options (create backup)"):
                try:
                    dp_path = Path("dropdown_options.pkl")
                    if dp_path.exists():
                        # load existing
                        data = joblib.load(str(dp_path))
                        # backup
                        backup_path = Path(str(dp_path) + ".bak")
                        joblib.dump(data, str(backup_path))
                        # append if not present
                        job_list = data.get('job_title', [])
                        if mapped_label not in job_list:
                            job_list.append(mapped_label)
                            data['job_title'] = job_list
                            joblib.dump(data, str(dp_path))
                            st.success(f"Appended '{mapped_label}' to dropdown options and created backup '{backup_path.name}'.")
                        else:
                            st.warning(f"'{mapped_label}' is already present in dropdown options.")
                    else:
                        st.error('dropdown_options.pkl not found in repo root; cannot append.')
                except Exception as e:
                    st.error(f"Failed to append to dropdown options: {e}")

        if st.button("Use this job role for salary prediction"):
            # Set session state so main app can prefill; prefer mapped_label when available
            final_label = mapped_label if mapped_label else chosen
            st.session_state["job_title"] = final_label
            # also store extracted skills (if available) to help downstream explainability
            try:
                st.session_state["resume_skills"] = skills
            except Exception:
                st.session_state["resume_skills"] = []
            st.success(f"Job role '{final_label}' set. Switch to main page to see it prefilled.")


# When Streamlit imports this file as a page, call resume_page()
if __name__ == "__main__":
    resume_page()