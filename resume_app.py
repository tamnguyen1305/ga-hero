import PyPDF2
import pandas as pd
import spacy
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
from pdfminer.high_level import extract_text
from docx2txt import process as docx_process

# Ensure spaCy model is available
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading missing spaCy model... (This may take a minute)")
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Load NLP Model
nlp = load_spacy_model()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    return extract_text(uploaded_file)

# Function to extract text from Word Document
def extract_text_from_docx(uploaded_file):
    return docx_process(uploaded_file)

# Function to match criteria using NLP similarity
def match_criteria(cv_text, criteria_list):
    scores = []
    for criteria in criteria_list:
        cv_embedding = model.encode(cv_text, convert_to_tensor=True)
        criteria_embedding = model.encode(criteria, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(cv_embedding, criteria_embedding).item()
        scores.append(similarity)
    return scores

# Streamlit UI
st.title("AI-Powered Resume Evaluator")
st.write("Upload a CV and an evaluation grid to assess candidate suitability.")

cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
grid_file = st.file_uploader("Upload Evaluation Grid (DOCX)", type=["docx"])

if cv_file and grid_file:
    st.write("Processing...")
    
    cv_text = extract_text_from_pdf(cv_file)
    grid_text = extract_text_from_docx(grid_file)
    
    mandatory_criteria = [line.strip() for line in grid_text.split("\n") if line.startswith("M")]
    rated_criteria = [line.strip() for line in grid_text.split("\n") if line.startswith("R")]
    
    mandatory_scores = match_criteria(cv_text, mandatory_criteria)
    rated_scores = match_criteria(cv_text, rated_criteria)
    
    mandatory_results = ["Met" if score > 0.6 else "Not Met" for score in mandatory_scores]
    rated_results = [round(score * 10) for score in rated_scores]
    
    df = pd.DataFrame({
        "Mandatory Criteria": mandatory_criteria,
        "Met or Not Met": mandatory_results,
        "Rated Criteria": rated_criteria,
        "Score": rated_results
    })
    
    st.write("### Evaluation Results")
    st.dataframe(df)
    
    output = BytesIO()
    df.to_excel(output, index=False, engine='xlsxwriter')
    output.seek(0)
    st.download_button(label="Download Results as Excel", data=output, file_name="cv_evaluation.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    st.success("Evaluation complete! Download your results above.")
