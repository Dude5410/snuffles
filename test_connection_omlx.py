import chromadb
import requests
import os
from dotenv import load_dotenv

# Laddar in de hemliga variablerna från .env-filen
load_dotenv()

print("--- Testing Connections for Snuffles ---")

# 1. Verifiera kontakt med oMLX säkert
try:
    # Hämtar nyckeln bakom kulisserna
    api_key = os.getenv("OMLX_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    res = requests.get("http://127.0.0.1:8000/v1/models", headers=headers, timeout=5)
    if res.status_code == 200:
        models = [m["id"] for m in res.json().get("data", [])]
        print(f"[OK] Contact with oMLX! Active models: {models}")
    else:
        print(f"[Warning] oMLX responded with status code: {res.status_code}")
except Exception as e:
    print(f"[Error] Could not reach oMLX on port 8000: {e}")

# 2. Verifiera persistent ChromaDB
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="snuffles_test")
    collection.add(
        documents=["This is a testvector."],
        ids=["id1"]
    )
    count = collection.count()
    print(f"[OK] ChromaDB saves locally! Number of objects in test database: {count}")
except Exception as e:
    print(f"[Error] Error during ChromaDB initialization: {e}")