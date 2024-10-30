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
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import aiohttp
import requests
import time

PASSWORD = st.secrets["PASSWORD"]
ACCESS_KEY = st.secrets["ACCESS_KEY"]
SECRET_KEY = st.secrets["SECRET_KEY"]
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]

API_URL = "https://api-inference.huggingface.co/models/RussianNLP/FRED-T5-Summarizer"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

def load_hf_token():
    return HUGGINGFACE_TOKEN

@st.cache_resource
def load_whisper_model():
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")
    return processor, model

async def speech2text(audio_data) -> dict:
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
    hf_token = load_hf_token()
    if not hf_token:
        st.error("Токен Hugging Face не найден.")
        return {}
    headers = {"Authorization": f"Bearer {hf_token}"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=headers, data=audio_data) as response:
                result = await response.json()
                return result
    except Exception as e:
        st.error(f"Ошибка при обращении к API: {e}")
        return {}

def transcribe_speech(audio_file):
    try:
        audio_bytes = audio_file.getvalue()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(speech2text(audio_bytes))
        transcription = result.get("text", "")
        return transcription
    except Exception as e:
        st.error(f"Ошибка транскрипции: {e}")
        return ""

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

@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-large")

def find_relevant_templates(input_text, embeddings, df, top_n):
    model = load_model()
    input_embedding = model.encode(input_text)
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_templates = df.iloc[top_indices]['Текст шаблона в выбранной версии Типологизатора'].values
    top_scores = similarities[top_indices]
    return top_templates, top_scores

def summarize_text(text, retries=3, delay=10):
    payload = {"inputs": text, "parameters": {"max_length": 300, "min_length": 100, "do_sample": False}}
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result[0].get('summary_text', "Ошибка: ключ 'summary_text' не найден в ответе.")
            else:
                st.error(f"Ошибка {response.status_code} при обращении к API. Ответ: {response.text}")
                return f"Ошибка: {response.status_code}, попробуйте позже."
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка сети: {e}")
            return "Ошибка при попытке суммаризации."

    return "Сервер недоступен, попробуйте позже."


def main():
    st.title("Поиск релевантных шаблонов")

    if "input_phrase" not in st.session_state:
        st.session_state["input_phrase"] = ""
    if "show_modal" not in st.session_state:
        st.session_state["show_modal"] = False
    if "current_summary" not in st.session_state:
        st.session_state["current_summary"] = ""

    # Загружаем данные из S3 перед использованием
    df, embeddings = load_data_from_s3()

    audio_input = st.experimental_audio_input("Голосовой ввод 🎙️")
    if audio_input is not None:
        st.write("Аудио получено. Выполняется транскрибация звука...")
        transcription = transcribe_speech(audio_input)
        if transcription:
            st.write("Транскрибация завершена:")
            st.write(transcription)
            st.session_state["input_phrase"] = transcription

    # Поле для ввода текста и выбор количества шаблонов
    st.session_state["input_phrase"] = st.text_input(
        "Введите текст для поиска релевантных шаблонов:",
        value=st.session_state["input_phrase"],
        key="search_phrase"
    )
    top_n = st.slider("Выберите количество шаблонов:", min_value=1, max_value=11, step=1)

    # Кнопка для поиска релевантных шаблонов
    if st.button("Найти шаблоны"):
        relevant_templates, scores = find_relevant_templates(st.session_state["input_phrase"], embeddings, df, top_n)
        st.write("Релевантные шаблоны:")

        for i, (template, score) in enumerate(zip(relevant_templates, scores)):
            st.write(f"**Шаблон {i + 1}:**")
            st.write(template)  # Выводим текст шаблона без `textwrap.fill`
            st.write(f"**Схожесть:** {score:.4f}")

            # HTML-кнопка для суммаризации и копирования с использованием JavaScript
            copy_button_html = f"""
                <button onclick="copyToClipboard('template_{i}')">Скопировать шаблон {i + 1}</button>
                <button onclick="openModal('modal_{i}')">Суммаризировать шаблон {i + 1}</button>
                <textarea id="template_{i}" style="display:none;">{template}</textarea>
                <div id="modal_{i}" class="modal" style="display:none;">
                    <div class="modal-content">
                        <span class="close" onclick="closeModal('modal_{i}')">&times;</span>
                        <p>Суммаризация:</p>
                        <p>{summarize_text(template)}</p>
                    </div>
                </div>
                <script>
                    function copyToClipboard(id) {{
                        var copyText = document.getElementById(id);
                        copyText.style.display = 'block';
                        copyText.select();
                        document.execCommand('copy');
                        copyText.style.display = 'none';
                        alert('Шаблон скопирован в буфер обмена!');
                    }}
                    function openModal(id) {{
                        document.getElementById(id).style.display = 'block';
                    }}
                    function closeModal(id) {{
                        document.getElementById(id).style.display = 'none';
                    }}
                </script>
                <style>
                    .modal {{
                        display: none;
                        position: fixed;
                        z-index: 1;
                        left: 0;
                        top: 0;
                        width: 100%;
                        height: 100%;
                        overflow: auto;
                        background-color: rgba(0,0,0,0.4);
                    }}
                    .modal-content {{
                        background-color: #fefefe;
                        margin: 15% auto;
                        padding: 20px;
                        border: 1px solid #888;
                        width: 80%;
                    }}
                    .close {{
                        color: #aaa;
                        float: right;
                        font-size: 28px;
                        font-weight: bold;
                    }}
                    .close:hover, .close:focus {{
                        color: black;
                        text-decoration: none;
                        cursor: pointer;
                    }}
                </style>
            """
            # Вставка HTML и JavaScript в Streamlit
            st.components.v1.html(copy_button_html, height=300)


# Проверка аутентификации
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