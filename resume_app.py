import PyPDF2
import pandas as pd
import spacy
from flask import Flask, request, jsonify, send_file
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
from pdfminer.high_level import extract_text
from docx2txt import process as docx_process

app = Flask(__name__)

# Load NLP Model
nlp = spacy.load("en_core_web_sm")
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

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'cv' not in request.files or 'grid' not in request.files:
        return jsonify({"error": "Please upload both CV and evaluation grid."}), 400
    
    cv_file = request.files['cv']
    grid_file = request.files['grid']
    
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
    
    output = BytesIO()
    df.to_excel(output, index=False, engine='xlsxwriter')
    output.seek(0)
    
    return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='cv_evaluation.xlsx')

if __name__ == '__main__':
    app.run(debug=True)
