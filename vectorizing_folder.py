import os
import requests
import chromadb
import pypdf
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

OMLX_API_KEY = os.getenv("OMLX_API_KEY", "")
OMLX_EMBEDDING_URL = "http://127.0.0.1:8000/v1/embeddings"

# UPPDATERA DENNA TILL NAMNET PÅ DIN EMBEDDING-MODELL (t.ex. "gemma-2-embedding")
EMBEDDING_MODEL_NAME = "embeddinggemma-2-bf16" 

# --- STEG 1: Skapa bryggan mellan ChromaDB och oMLX ---
class OMLXEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        headers = {
            "Authorization": f"Bearer {OMLX_API_KEY}",
            "Content-Type": "application/json"
        }
        
        embeddings = []
        for text in input:
            payload = {
                "model": EMBEDDING_MODEL_NAME,
                "input": text
            }
            try:
                res = requests.post(OMLX_EMBEDDING_URL, json=payload, headers=headers)
                res.raise_for_status()
                vector = res.json()["data"][0]["embedding"]
                embeddings.append(vector)
            except Exception as e:
                print(f"Error creating vector for text: {e}")
                
        return embeddings

# --- STEG 2: Starta databasen ---
print("Starting ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
omlx_ef = OMLXEmbeddingFunction()

# Vi kallar databasen för snuffles_docs och kopplar vår embedding-funktion
collection = client.get_or_create_collection(
    name="snuffles_docs",
    embedding_function=omlx_ef
)

# --- STEG 3: Läs igenom mappen (Nu med PDF-stöd!) ---
def index_folder(folder_path: str):
    print(f"Searching folder: '{folder_path}'...")
    
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist. Create it and place some files there.")
        return

    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 1. Hantera Text & Markdown
        if filename.lower().endswith(".txt") or filename.lower().endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                if content.strip():
                    documents.append(content)
                    metadatas.append({"filename": filename})
                    ids.append(filename)
                    print(f"Found text file: {filename}")
                    
        # 2. Hantera PDF-filer
        elif filename.lower().endswith(".pdf"):
            try:
                reader = pypdf.PdfReader(file_path)
                content = ""
                # Plocka ut texten från varje sida
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
                
                if content.strip():
                    documents.append(content)
                    metadatas.append({"filename": filename})
                    ids.append(filename)
                    print(f"Extracted text from PDF: {filename}")
            except Exception as e:
                print(f"Could not read PDF {filename}: {e}")

        # 3. Känner igen bilder, men hoppar över dem
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Found image: {filename} - (Skipping: Requires Vision model for text indexing)")

    # --- STEG 4: Spara i databasen ---
    if documents:
        print("\nSending text to your oMLX model for vectorization...")
        print("This may take a moment depending on the amount of text you have in the files.")
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"\n[Added] {len(documents)} files to the database.")
    else:
        print("\nDid not find any text or PDF files to index. Add some files to the folder and rerun the script.")

if __name__ == "__main__":
    mapp_att_lasa_in = "/Users/johnbodeklint/dev/snuffles/snuffles_files" 
    
    if not os.path.exists(mapp_att_lasa_in):
        os.makedirs(mapp_att_lasa_in)
        print(f"Created a new folder called '{mapp_att_lasa_in}'. Place your PDFs and text files there and run the script again!")
    else:
        index_folder(mapp_att_lasa_in)