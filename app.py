import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from groq import Groq
from pdfminer.high_level import extract_text as extract_text_from_pdf
from docx import Document
from PIL import Image
import pytesseract
import base64

# Load secrets
groq_key = st.secrets["groq"]["api_key"]

# Initialize clients
groq_client = Groq(api_key=groq_key)

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to extract text from different file types
def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        text = extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    elif file.type.startswith("image/"):
        text = pytesseract.image_to_string(Image.open(file))
    else:
        text = file.read().decode("utf-8")
    return text

# Function to call OpenAI for SWOT analysis
def call_openai_for_swot(text, system_prompt, user_prompt, expected_format, openai_api_key):
    openai_client = OpenAI(api_key=openai_api_key)
    completion = openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
    ).choices
    return json.loads(completion.message.content)

# Function to call Groq for SWOT analysis
def call_groq_for_swot(text, system_prompt, user_prompt, expected_format):
    completion = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": f"output only JSON object. {system_prompt}"},
            {"role": "user", "content": f"{expected_format}"},
            {"role": "user", "content": f"{user_prompt}"}
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content)

# Function to create bar graph
def create_bar_graph(data, title):
    categories = list(data.keys())
    values = list(data.values())
    fig, ax = plt.subplots()
    ax.barh(categories, values, color='skyblue')
    ax.set_xlabel('Count')
    ax.set_title(title)
    st.pyplot(fig)

# Streamlit app
st.set_page_config(layout="wide")

st.title("Assignment Evaluation Environment")

# Top left analysis type and OpenAI API key input
col1, col2 = st.columns([1, 3])
with col1:
    analysis_type = st.selectbox("Select analysis type", ["Text only", "Vision"])
with col2:
    if analysis_type == "Vision":
        openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key

# Context and total marks input
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    context = st.text_input("Enter context for the project:")
with col2:
    total_marks = st.number_input("Enter total marks for the assignment:", min_value=0, value=100)
with col3:
    pass

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

# Process files
if uploaded_files:
    if analysis_type == "Vision" and 'openai_api_key' not in st.session_state:
        st.warning("Please enter your OpenAI API key.")
    else:
        progress_bar = st.progress(0)
        for idx, file in enumerate(uploaded_files):
            text = extract_text(file)
            if not text:
                st.error(f"No text found in {file.name}.")
                continue
            
            word_count = len(text.split())
            
            # Call appropriate model
            if analysis_type == "Text only":
                system_prompt = "Perform a SWOT analysis and return a JSON object with keys: Strengths, Weaknesses, Opportunities, Threats, Total Marks, Word Count."
                user_prompt = f"Text: {text} Total Marks: {total_marks} Word Count: {word_count}"
                swot_analysis = call_groq_for_swot(text, system_prompt, user_prompt, expected_json_format)
            else:
                system_prompt = "Perform a SWOT analysis on this image and return a JSON object with keys: Strengths, Weaknesses, Opportunities, Threats, Total Marks, Word Count."
                base64_image = encode_image(file)
                user_prompt = f"Image: {base64_image} Total Marks: {total_marks} Word Count: {word_count}"
                swot_analysis = call_openai_for_swot(base64_image, system_prompt, user_prompt, expected_json_format, openai_api_key=st.session_state.openai_api_key)
            
            # Validate returned JSON keys
            if not all(key in swot_analysis for key in expected_json_keys):
                st.error(f"Invalid SWOT analysis response for {file.name}. Missing keys.")
                continue
            
            # Display SWOT analysis
            with st.expander(f"SWOT Analysis for {file.name}"):
                st.json(swot_analysis)
            
            # Generate bar graph data
            scores = {key: len(swot_analysis.get(key, "")) for key in expected_json_keys if key not in ["Total Marks", "Word Count"]}
            create_bar_graph(scores, title=f"SWOT Analysis for {file.name}")
            
            # Update progress bar
            progress_bar.progress((idx + 1) / len(uploaded_files))
        progress_bar.empty()
