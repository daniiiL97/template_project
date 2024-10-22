import boto3
from botocore.client import Config
import pandas as pd
from io import BytesIO
vk_cloud_endpoint = 'https://hb.bizmrg.com'
access_key = key1
secret_key = key2

s3 = boto3.client(
    's3',
    endpoint_url=vk_cloud_endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version='s3v4')
)

bucket_name = 'templates97'
object_name = "typology_templates.xlsx"

try:
    response = s3.get_object(Bucket = bucket_name, Key = object_name)
    file_content = response["Body"].read()
    excel_data = pd.read_excel(BytesIO(file_content),engine='openpyxl')
    print(excel_data.head(5))
except Exception as e:
    print(f"Произошла неизвестная ошибка: {e}")