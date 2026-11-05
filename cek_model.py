import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Daftar model yang bisa dipakai:")
# Meminta Google menyebutkan semua menu model yang tersedia
for model in client.models.list():
    print(model.name)