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
import time

# Load secrets
groq_key = st.secrets["groq"]["api_key"]

# Initialize clients
groq_client = Groq(api_key=groq_key)

# User credentials
users = {
    "pratik": "pratik",
    "user1": "user1",
    "user2": "user2"
}

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

# Login page
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.header("Login")
    username = st.selectbox("Select Username", list(users.keys()))
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

# Post-login
if st.session_state.logged_in:
    st.header("Welcome, " + st.session_state.username)

    # OpenAI API Key input
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key

    # Context input
    context = st.text_input("Enter context for the project:")

    # File uploader
    uploaded_files = st.file_uploader("Upload assignment files", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"], accept_multiple_files=True)

    # Dropdown to select analysis type
    analysis_type = st.selectbox("Select analysis type", ["Text only", "Vision"])

    # Expected JSON format for SWOT analysis
    expected_json_format = {
        "Strengths": "",
        "Weaknesses": "",
        "Opportunities": "",
        "Threats": ""
    }
    expected_json_keys = expected_json_format.keys()

    # Process files
    if uploaded_files and 'openai_api_key' in st.session_state:
        progress_bar = st.progress(0)
        for idx, file in enumerate(uploaded_files):
            text = extract_text(file)
            if not text:
                st.error(f"No text found in {file.name}.")
                continue
            
            # Call appropriate model
            if analysis_type == "Text only":
                system_prompt = "Perform a SWOT analysis and return a JSON object with keys: Strengths, Weaknesses, Opportunities, Threats."
                user_prompt = f"Text: {text}"
                swot_analysis = call_groq_for_swot(text, system_prompt, user_prompt, expected_json_format)
            else:
                image_path = file.name
                base64_image = encode_image(image_path)
                system_prompt = "Perform a SWOT analysis on this image and return a JSON object with keys: Strengths, Weaknesses, Opportunities, Threats."
                user_prompt = f"Image: {base64_image}"
                swot_analysis = call_openai_for_swot(base64_image, system_prompt, user_prompt, expected_json_format, openai_api_key=st.session_state.openai_api_key)
            
            # Validate returned JSON keys
            if not all(key in swot_analysis for key in expected_json_keys):
                st.error(f"Invalid SWOT analysis response for {file.name}. Missing keys.")
                continue
            
            # Display SWOT analysis
            st.subheader(f"SWOT Analysis for {file.name}")
            st.json(swot_analysis)
            
            # Generate spider graph data
            scores = {key: len(swot_analysis.get(key, "")) for key in expected_json_keys}
            create_spider_graph(scores, title=f"SWOT Analysis for {file.name}")
            
            # Update progress bar
            progress_bar.progress((idx + 1) / len(uploaded_files))
        progress_bar.empty()

    elif not uploaded_files:
        st.warning("Please upload assignment files.")
    elif 'openai_api_key' not in st.session_state:
        st.warning("Please enter your OpenAI API key.")
