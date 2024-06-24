import streamlit as st
import requests
import json
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from pdfminer.high_level import extract_text as extract_text_from_pdf
from PIL import Image
import pytesseract
import base64
import mimetypes
import google.generativeai as genai
import io

# Load secrets
gemini_key = st.secrets["gemini"]["api_key"]

# Configure the Gemini API key
genai.configure(api_key=gemini_key)

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to extract text from different file types
def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        text = extract_text_from_pdf(file)
        if not text.strip():
            images = pdf_to_images(file)
            text = " ".join(pytesseract.image_to_string(image) for image in images)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    elif file.type.startswith("image/"):
        text = pytesseract.image_to_string(Image.open(file))
    else:
        text = file.read().decode("utf-8")
    return text

# Function to interact with the Gemini API for text analysis
def gemini_json(system_prompt, user_prompt, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key=" + api_key
    
    # Create a JSON payload for the API request
    payload = json.dumps({
        "contents": [
            {
                "parts": [
                    {"text": system_prompt},
                    {"text": user_prompt},
                    {"text": ""}
                ]
            }
        ],
        "generationConfig": {"response_mime_type": "application/json"}
    })

    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, headers=headers, data=payload)
    response_data = response.json()
    
    if "candidates" in response_data and response_data["candidates"]:
        return json.loads(response_data["candidates"][0]["content"]["parts"][0]["text"])
    return {}

# Helper function to read image bytes and encode them in base64
def read_image_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Helper function to get MIME type based on file extension
def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

# Function to process images and send to Gemini Vision API
def process_images_gemini(images, prompt, api_key):
    encoded_images = []
    for image in images:
        encoded_images.append({
            'mime_type': get_mime_type(image),
            'data': read_image_base64(image)
        })

    model = genai.GenerativeModel("gemini-1.5-flash")
    content = [prompt] + encoded_images

    try:
        response = model.generate_content(content, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

# Function to convert PDF to images using PyMuPDF
def pdf_to_images(pdf_file):
    images = []
    file_bytes = pdf_file.read()
    if not file_bytes:
        raise ValueError("The PDF file is empty.")
    document = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(image)
    return images

# Streamlit app
st.set_page_config(layout="wide")

st.title("Assignment Evaluation Environment")

# Top left analysis type input
col1, col2 = st.columns([1, 3])
with col1:
    analysis_type = st.selectbox("Select analysis type", ["Text only", "Vision"])

# Context and total marks input
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    context = st.text_input("Enter context for the project:")
with col2:
    total_marks = st.number_input("Enter total marks for the assignment:", min_value=0, value=100)
with col3:
    max_word_count = st.slider("Set maximum word count:", min_value=100, max_value=3000, value=300, step=100)

# File uploader
uploaded_files = st.file_uploader("Upload assignment files", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"], accept_multiple_files=True)

# Expected JSON format for SWOT analysis
expected_json_format = {
    "Strengths": "",
    "Weaknesses": "",
    "Opportunities": "",
    "Threats": "",
    "Total Marks": total_marks,
    "Word Count": 0
}
expected_json_keys = expected_json_format.keys()

# Grading criteria to guide the LLM
grading_criteria = """
Grading Criteria:
1. Content quality (40%): How well the text addresses the topic.
2. Clarity and coherence (30%): How clear and logical the text is.
3. Grammar and language (20%): Correct use of grammar and language.
4. Originality (10%): Uniqueness and originality of the content.

Use these criteria to assign marks out of {total_marks}.
"""

# Process files
if uploaded_files:
    progress_bar = st.progress(0)
    for idx, file in enumerate(uploaded_files):
        text = extract_text(file)
        if not text:
            st.error(f"No text found in {file.name}.")
            continue
        
        word_count = len(text.split())
        
        # Construct prompts and call appropriate model
        system_prompt = f"Perform a SWOT analysis with each category limited to {max_word_count} words. Return a JSON object with keys: Strengths, Weaknesses, Opportunities, Threats, Total Marks, Word Count."
        user_prompt = f"Text: {text}\nTotal Marks: {total_marks}\nWord Count: {word_count}\n{grading_criteria.format(total_marks=total_marks)}"
        
        if analysis_type == "Text only":
            swot_analysis = gemini_json(system_prompt, user_prompt, gemini_key)
        else:
            images = pdf_to_images(file)
            if not images:
                st.error(f"No images extracted from {file.name}.")
                continue
            user_prompt = f"Extract data from these images: {images}\nTotal Marks: {total_marks}\nWord Count: {word_count}\n{grading_criteria.format(total_marks=total_marks)}"
            swot_analysis = process_images_gemini(images, user_prompt, gemini_key)
        
        # Validate returned JSON keys
        if not all(key in swot_analysis for key in expected_json_keys):
            st.error(f"Invalid SWOT analysis response for {file.name}. Missing keys.")
            continue
        
        # Display SWOT analysis with bounding boxes and colors
        with st.expander(f"SWOT Analysis for {file.name}"):
            st.markdown(f"<div style='border:2px solid #FFFFFF; padding: 10px; margin-bottom: 10px;'><strong>Word Count:</strong> {swot_analysis.get('Word Count', 'N/A')}<br><strong>Total Marks:</strong> {swot_analysis.get('Total Marks', 'N/A')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='border:2px solid #75FF33; padding: 10px; margin-bottom: 10px;'><strong>Strengths:</strong> {swot_analysis.get('Strengths', 'N/A')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='border:2px solid #FF33D4; padding: 10px; margin-bottom: 10px;'><strong>Weaknesses:</strong> {swot_analysis.get('Weaknesses', 'N/A')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='border:2px solid #FF5733; padding: 10px; margin-bottom: 10px;'><strong>Opportunities:</strong> {swot_analysis.get('Opportunities', 'N/A')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='border:2px solid #33FFBD; padding: 10px; margin-bottom: 10px;'><strong>Threats:</strong> {swot_analysis.get('Threats', 'N/A')}</div>", unsafe_allow_html=True)
        
        # Update progress bar
        progress_bar.progress((idx + 1) / len(uploaded_files))
    progress_bar.empty()
