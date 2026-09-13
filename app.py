import streamlit as st
import re
from io import BytesIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


st.set_page_config(
    page_title="Smart Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Smart Resume Screening System")
st.write(
    "Upload a resume and enter a job description to calculate "
    "how closely the candidate matches the role."
)

# -----------------------------
# Text extraction
# -----------------------------

def extract_pdf_text(file):
    if PdfReader is None:
        return ""

    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def extract_docx_text(file):
    if Document is None:
        return ""

    document = Document(BytesIO(file.read()))
    return "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )


def extract_text(file):
    file_type = file.name.lower()

    if file_type.endswith(".pdf"):
        return extract_pdf_text(file)

    elif file_type.endswith(".docx"):
        return extract_docx_text(file)

    elif file_type.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    return ""


# -----------------------------
# Text preprocessing
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------
# Skill matching
# -----------------------------

SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "sql",
    "html",
    "css",
    "react",
    "next.js",
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "generative ai",
    "artificial intelligence",
    "rag",
    "streamlit",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "pandas",
    "numpy",
    "git",
    "github",
    "docker",
    "flask",
    "fastapi",
]


def find_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        if skill.lower() in text:
            found.append(skill)

    return sorted(set(found))


# -----------------------------
# Resume matching
# -----------------------------

def calculate_match(resume_text, job_description):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_description)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform([
        resume_clean,
        job_clean
    ])

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return round(similarity * 100, 2)


# -----------------------------
# User interface
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📎 Upload Resume")

    resume_file = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"]
    )

with col2:
    st.subheader("💼 Job Description")

    job_description = st.text_area(
        "Paste the job description here",
        height=220,
        placeholder="Example: Looking for a Python developer with Machine Learning, NLP and SQL skills..."
    )


if st.button("🔍 Screen Resume", type="primary"):

    if resume_file is None:
        st.warning("Please upload a resume first.")

    elif not job_description.strip():
        st.warning("Please enter a job description.")

    else:
        with st.spinner("Analyzing resume..."):

            resume_text = extract_text(resume_file)

            if not resume_text.strip():
                st.error(
                    "Could not extract text from the uploaded resume."
                )

            else:
                score = calculate_match(
                    resume_text,
                    job_description
                )

                resume_skills = find_skills(resume_text)
                job_skills = find_skills(job_description)

                matching_skills = sorted(
                    set(resume_skills) & set(job_skills)
                )

                missing_skills = sorted(
                    set(job_skills) - set(resume_skills)
                )

                st.divider()

                st.subheader("📊 Screening Result")

                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:
                    st.metric(
                        "Match Score",
                        f"{score}%"
                    )

                with result_col2:
                    st.metric(
                        "Matching Skills",
                        len(matching_skills)
                    )

                with result_col3:
                    st.metric(
                        "Missing Skills",
                        len(missing_skills)
                    )

                if score >= 70:
                    st.success(
                        "✅ Strong Match — Candidate appears highly relevant."
                    )

                elif score >= 50:
                    st.info(
                        "🟡 Moderate Match — Candidate may be suitable."
                    )

                else:
                    st.warning(
                        "🔴 Low Match — Resume has limited similarity to the job description."
                    )

                st.subheader("✅ Matching Skills")

                if matching_skills:
                    st.write(", ".join(matching_skills))
                else:
                    st.write("No matching skills detected.")

                st.subheader("⚠️ Skills to Improve")

                if missing_skills:
                    st.write(", ".join(missing_skills))
                else:
                    st.write("No major missing skills detected.")

                with st.expander("📄 Extracted Resume Text"):
                    st.write(resume_text)