import os
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables del .env
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str):
    result = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return result.data[0].embedding