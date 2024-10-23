import os
import logging
import asyncio
import streamlit as st
import boto3
from botocore.client import Config
import pandas as pd
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ast
import textwrap
import pyperclip
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import aiohttp


# Load password from file
with open("keys/password.txt", "r") as f:
    PASSWORD = f.read().strip()


@st.cache_resource
def load_whisper_model():
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")
    return processor, model


async def speech2text(audio_file) -> dict:
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
    headers = {"Authorization": f'Bearer {os.environ.get("HF_TOKEN")}'}

    async with aiohttp.ClientSession() as session:
        async with session.post(
                API_URL,
                headers=headers,
                data=audio_file.read()
        ) as response:
            return await response.json()


def transcribe_speech(audio_file):
    try:
        # Read audio file as bytes
        audio_bytes = audio_file.getvalue()

        # Create BytesIO object from audio data
        audio_file = BytesIO(audio_bytes)

        # Perform transcription
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(speech2text(audio_file))

        # Extract text from result
        transcription = result.get("text", "")
        return transcription
    except Exception as e:
        st.error(f"Ошибка транскрипции: {e}")
        return ""


@st.cache_data
def load_data_from_s3():
    try:
        with open("keys/secret_keys.txt", "r") as f:
            keys = f.readlines()

        access_key = keys[0].split('=')[1].strip()
        secret_key = keys[1].split('=')[1].strip()

        vk_cloud_endpoint = 'https://hb.bizmrg.com'
        s3 = boto3.client(
            's3',
            endpoint_url=vk_cloud_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
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


def copy_to_clipboard(text):
    pyperclip.copy(text)
    st.success("Шаблон скопирован в буфер обмена!")


# Session state check for app login
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
    st.stop()

if st.session_state["password_entered"]:
    st.title("Поиск релевантных шаблонов")

    # Load data from S3
    df, embeddings = load_data_from_s3()

    # Initialize input_phrase in session state to avoid multiple input boxes
    if "input_phrase" not in st.session_state:
        st.session_state["input_phrase"] = ""

    # Capture audio input
    audio_input = st.file_uploader("Запишите ваше сообщение:", type=["wav", "mp3"])

    if audio_input is not None:
        # Process audio input and transcription
        transcription = transcribe_speech(audio_input)

        if transcription:
            st.write("Транскрипция:")
            st.write(transcription)
            # Update the session state with the transcription
            st.session_state["input_phrase"] = transcription

    # Use one text input field that is populated with either manual or transcribed input
    st.session_state["input_phrase"] = st.text_input("Введите текст для поиска релевантных шаблонов:", value=st.session_state["input_phrase"], key="search_phrase")

    top_n = st.slider("Выберите количество шаблонов:", min_value=1, max_value=11, step=1)

    if st.button("Найти шаблоны"):
        relevant_templates, scores = find_relevant_templates(st.session_state["input_phrase"], embeddings, df, top_n)

        st.write("Релевантные шаблоны:")

        for i, (template, score) in enumerate(zip(relevant_templates, scores)):
            wrapped_template = textwrap.fill(template, width=100)
            st.write(f"**Шаблон {i + 1}:**\n{wrapped_template}")
            st.write(f"**Схожесть:** {score:.4f}")

            if st.button(f"Скопировать шаблон {i + 1}", key=f"copy_{i}"):
                copy_to_clipboard(wrapped_template)

            st.write("************")
