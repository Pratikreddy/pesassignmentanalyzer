import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI
from groq import Groq
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from docx import Document
import base64
import os

# Load secrets
openai_key = st.secrets["openai"]["api_key"]
groq_key = st.secrets["groq"]["api_key"]

# Initialize clients
openai_client = OpenAI(api_key=openai_key)
groq_client = Groq(api_key=groq_key)

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to extract text from different file types
def extract_text(file):
    if file.type == "application/pdf":
        images = convert_from_path(file)
        text = ''
        for image in images:
            text += pytesseract.image_to_string(image)
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    elif file.type.startswith("image/"):
        return pytesseract.image_to_string(Image.open(file))
    else:
        return file.read().decode("utf-8")

# Function to call OpenAI for SWOT analysis
def call_openai_for_swot(text, system_prompt, user_prompt, expected_format):
    completion = openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"system_prompt : {system_prompt}"},
            {"role": "user", "content": f"user_prompt : {user_prompt}"},
            {"role": "user", "content": f"expected_JSON_format : {expected_format}"}
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

# Function to create spider graph
def create_spider_graph(data, title):
    categories = list(data.keys())
    values = list(data.values())
    values += values[:1]
    N = len(categories)
    angles = [n / float(N) * 2 * 3.14 for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.plot(angles, values)
    ax.fill(angles, values, 'b', alpha=0.1)
    plt.title(title)
    st.pyplot(plt)

# Streamlit app
st.title("Assignment Evaluation Environment")

# Context input
context = st.text_input("Enter context for the project:")

# File uploader
uploaded_files = st.file_uploader("Upload assignment files", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"], accept_multiple_files=True)

# Dropdown to select analysis type
analysis_type = st.selectbox("Select analysis type", ["Text only", "Vision"])

# Process files
if uploaded_files:
    for file in uploaded_files:
        text = extract_text(file)
        if not text:
            st.error(f"No text found in {file.name}.")
            continue
        
        # Call appropriate model
        if analysis_type == "Text only":
            swot_analysis = call_groq_for_swot(text, system_prompt="Perform a SWOT analysis.", user_prompt=f"Text: {text}", expected_format="JSON")
        else:
            image_path = file.name
            base64_image = encode_image(image_path)
            swot_analysis = call_openai_for_swot(base64_image, system_prompt="Perform a SWOT analysis on this image.", user_prompt=f"Image: {base64_image}", expected_format="JSON")
        
        # Display SWOT analysis
        st.subheader(f"SWOT Analysis for {file.name}")
        st.json(swot_analysis)
        
        # Generate spider graph data
        scores = {key: swot_analysis.get(key, 0) for key in ["Strengths", "Weaknesses", "Opportunities", "Threats"]}
        create_spider_graph(scores, title=f"SWOT Analysis for {file.name}")
