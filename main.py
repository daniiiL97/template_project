import os
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
import requests

PASSWORD = st.secrets["PASSWORD"]
ACCESS_KEY = st.secrets["ACCESS_KEY"]
SECRET_KEY = st.secrets["SECRET_KEY"]
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
API_TOKEN = HUGGINGFACE_TOKEN


def load_hf_token():
    return HUGGINGFACE_TOKEN


@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-large")


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


def find_relevant_templates(input_text, embeddings, df, top_n):
    model = load_model()
    input_embedding = model.encode(input_text)
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_templates = df.iloc[top_indices]['Текст шаблона в выбранной версии Типологизатора'].values
    top_scores = similarities[top_indices]
    return top_templates, top_scores


# Функция для суммаризации текста
def summarize_text_sync(text, model="RussianNLP/FRED-T5-Summarizer"):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {"max_length": 50, "min_length": 25, "do_sample": False}
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['summary_text']
    else:
        st.error(f"Ошибка при суммаризации: {response.status_code}")
        return None


# Функция для транскрипции аудио
def transcribe_audio(audio_data):
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, data=audio_data)
    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        st.error(f"Ошибка транскрипции: {response.status_code}")
        return ""


def main():
    st.title("Поиск релевантных шаблонов")

    # Загрузка данных и настроек
    df, embeddings = load_data_from_s3()
    if "input_phrase" not in st.session_state:
        st.session_state["input_phrase"] = ""

    # Аудио ввод и его транскрипция
    audio_input = st.file_uploader("Загрузите аудиофайл для транскрипции 🎙️", type=["wav", "mp3"])
    if audio_input:
        st.write("Транскрибируем аудио...")
        transcription = transcribe_audio(audio_input.read())
        if transcription:
            st.write("Транскрипция завершена:")
            st.write(transcription)
            st.session_state["input_phrase"] = transcription

    # Текстовый ввод
    st.session_state["input_phrase"] = st.text_input(
        "Введите текст для поиска релевантных шаблонов:",
        value=st.session_state["input_phrase"],
        key="search_phrase"
    )

    # Слайдер для выбора количества шаблонов
    top_n = st.slider("Выберите количество шаблонов:", min_value=1, max_value=11, step=1)

    # Кнопка для поиска шаблонов
    if st.button("Найти шаблоны"):
        relevant_templates, scores = find_relevant_templates(st.session_state["input_phrase"], embeddings, df, top_n)
        st.write("Релевантные шаблоны:")

        # Цикл для отображения шаблонов и кнопок суммаризации
        for i, (template, score) in enumerate(zip(relevant_templates, scores)):
            wrapped_template = textwrap.fill(template, width=100)
            st.write(f"**Шаблон {i + 1}:**\n{wrapped_template}")
            st.write(f"**Схожесть:** {score:.4f}")

            # Проверка состояния для отображения суммаризации
            if f"summary_{i}" not in st.session_state:
                st.session_state[f"summary_{i}"] = ""

            # Кнопка суммаризации с сохранением результата в session_state
            if st.button(f"Суммаризовать Шаблон {i + 1}", key=f"sum_button_{i}"):
                summary = summarize_text_sync(template)
                if summary:
                    st.session_state[f"summary_{i}"] = summary

            # Отображение суммаризированного текста, если он есть
            if st.session_state[f"summary_{i}"]:
                st.write(f"**Суммаризация шаблона {i + 1}:** {st.session_state[f'summary_{i}']}")

            # HTML для кнопки копирования
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


# Вход в приложение
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
