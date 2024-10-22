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


with open("keys/password.txt", "r") as f:
    PASSWORD = f.read().strip()


@st.cache_data
def load_data_from_s3():
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

    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_name)
        file_content = response["Body"].read()
        df = pd.read_excel(BytesIO(file_content), engine='openpyxl')
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке данных: {e}")
        st.stop()

    df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    embeddings = df['embedding'].apply(lambda x: np.array(x)).tolist()
    return df, embeddings


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


if "password_entered" not in st.session_state:
    st.session_state["password_entered"] = False

if not st.session_state["password_entered"]:
    st.title("Вход в приложение")
    password_input = st.text_input("Введите пароль:", type="password")

    if st.button("Войти"):
        if password_input == PASSWORD:
            st.session_state["password_entered"] = True
        else:
            st.error("Неверный пароль. Попробуйте снова.")
    st.stop()

if st.session_state["password_entered"]:
    st.title("Поиск релевантных шаблонов")

    df, embeddings = load_data_from_s3()

    input_phrase = st.text_input("Введите текст для поиска релевантных шаблонов:", "участники СВО")

    top_n = st.slider("Выберите количество шаблонов:", min_value=1, max_value=11, step=1)

    if st.button("Найти шаблоны"):
        relevant_templates, scores = find_relevant_templates(input_phrase, embeddings, df, top_n)

        st.write("Релевантные шаблоны:")
        for i, (template, score) in enumerate(zip(relevant_templates, scores)):
            wrapped_template = textwrap.fill(template, width=100)
            st.write(f"**Шаблон {i+1}:**\n{wrapped_template}")
            st.write(f"**Схожесть:** {score:.4f}")
            if st.button(f"Скопировать шаблон {i+1}", key=i):
                copy_to_clipboard(wrapped_template)
            st.write("************")
