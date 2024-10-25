import os
import asyncio
import streamlit as st
import streamlit.components.v1 as components
import boto3
from botocore.client import Config
import pandas as pd
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ast
import textwrap
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM
import aiohttp

# Authentication details
PASSWORD = st.secrets["PASSWORD"]
ACCESS_KEY = st.secrets["ACCESS_KEY"]
SECRET_KEY = st.secrets["SECRET_KEY"]
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]

# Initialize session state for summaries
if "summaries" not in st.session_state:
    st.session_state["summaries"] = {}


# Load Hugging Face token
def load_hf_token():
    return HUGGINGFACE_TOKEN


# Load Whisper model for speech-to-text
@st.cache_resource
def load_whisper_model():
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")
    return processor, model


# Asynchronous function for speech recognition API call
async def speech2text(audio_data) -> dict:
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
    hf_token = load_hf_token()
    headers = {"Authorization": f"Bearer {hf_token}"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=headers, data=audio_data) as response:
                result = await response.json()
                return result
    except Exception as e:
        st.error(f"Ошибка при обращении к API: {e}")
        return {}


# Function to transcribe speech from audio file
def transcribe_speech(audio_file):
    try:
        audio_bytes = audio_file.getvalue()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(speech2text(audio_bytes))
        transcription = result.get("text", "")
        return transcription
    except Exception as e:
        st.error(f"Ошибка транскрибации: {e}")
        return ""


# Load embeddings data from S3
@st.cache_data
def load_data_from_s3():
    try:
        vk_cloud_endpoint = 'https://hb.bizmrg.com'
        s3 = boto3.client(
            's3',
            endpoint_url=vk_cloud_endpoint,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(signature_version='s3v4')
        )
        bucket_name = 'templates97'
        object_name = "embeddings.xlsx"
        response = s3.get_object(Bucket=bucket_name, Key=object_name)
        file_content = response["Body"].read()
        df = pd.read_excel(BytesIO(file_content), engine='openpyxl')
        df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        embeddings = df['embedding'].apply(lambda x: np.array(x)).tolist()
        return df, embeddings
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке данных: {e}")
        st.stop()


# Load text embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-large")


# Load summary model
@st.cache_resource
def load_summary_model():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rut5-base-absum")
    model = AutoModelForSeq2SeqLM.from_pretrained("cointegrated/rut5-base-absum")
    return tokenizer, model


# Generate summary for text
def generate_summary(text):
    tokenizer, model = load_summary_model()
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=100, min_length=20, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Find relevant templates based on input text
def find_relevant_templates(input_text, embeddings, df, top_n):
    model = load_model()
    input_embedding = model.encode(input_text)
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_templates = df.iloc[top_indices]['Текст шаблона в выбранной версии Типологизатора'].values
    top_scores = similarities[top_indices]
    return top_templates, top_scores


# Main function for the Streamlit app
def main():
    st.title("Поиск релевантных шаблонов")
    df, embeddings = load_data_from_s3()

    if "input_phrase" not in st.session_state:
        st.session_state["input_phrase"] = ""

    # Audio input
    audio_input = st.experimental_audio_input("Голосовой ввод 🎙️")
    if audio_input is not None:
        st.write("Аудио получено. Выполняется транскрибация звука...")
        transcription = transcribe_speech(audio_input)
        if transcription:
            st.write("Транскрибация завершена:")
            st.write(transcription)
            st.session_state["input_phrase"] = transcription

    # Text input for search phrase
    st.session_state["input_phrase"] = st.text_input(
        "Введите текст для поиска релевантных шаблонов:",
        value=st.session_state["input_phrase"],
        key="search_phrase"
    )

    # Slider for number of templates to return
    top_n = st.slider("Выберите количество шаблонов:", min_value=1, max_value=11, step=1)

    # Button to find relevant templates
    if st.button("Найти шаблоны"):
        relevant_templates, scores = find_relevant_templates(st.session_state["input_phrase"], embeddings, df, top_n)
        st.write("Релевантные шаблоны:")

        for i, (template, score) in enumerate(zip(relevant_templates, scores)):
            wrapped_template = textwrap.fill(template, width=100)
            st.write(f"**Шаблон {i + 1}:**\n{wrapped_template}")
            st.write(f"**Схожесть:** {score:.4f}")

            # Generate summary and display it using JavaScript
            summary = generate_summary(template)
            summary_display_html = f"""
                <div id="summary_{i}" style="margin-top: 10px; color: #333; font-weight: bold;"></div>
                <button onclick="document.getElementById('summary_{i}').innerText='{summary}';" style="background-color: #007bff; color: white; padding: 8px 12px; border-radius: 4px; border: none; cursor: pointer;">Сделать краткое содержание</button>
            """
            components.html(summary_display_html)

            # Copy button for the template
            copy_button_html = f"""
                <style>
                    .copy-button {{
                        background-color: #4CAF50;
                        border: none;
                        color: white;
                        padding: 10px 20px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 16px;
                        margin: 4px 2px;
                        cursor: pointer;
                        border-radius: 12px;
                    }}
                </style>
                <button class="copy-button" onclick="copyToClipboard('template_{i}')">Скопировать шаблон {i + 1}</button>
                <textarea id="template_{i}" style="display:none;">{wrapped_template}</textarea>
                <script>
                function copyToClipboard(id) {{
                    var copyText = document.getElementById(id);
                    copyText.style.display = 'block';
                    copyText.select();
                    document.execCommand('copy');
                    copyText.style.display = 'none';
                    alert('Шаблон скопирован в буфер обмена!');
                }}
                </script>
            """
            components.html(copy_button_html)
            st.write("************")


# Authentication check
if "password_entered" not in st.session_state:
    st.session_state["password_entered"] = False

if not st.session_state["password_entered"]:
    st.title("Вход в приложение")
    password_input = st.text_input("Введите пароль:", type="password")
    if st.button("Войти"):
        if password_input == PASSWORD:
            st.session_state["password_entered"] = True
            st.success("Вы успешно вошли!")
        else:
            st.error("Неверный пароль. Попробуйте снова.")
else:
    main()
