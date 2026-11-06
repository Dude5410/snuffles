import os
import requests
import chromadb
import pypdf
import base64  # New import required for handling images
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

OMLX_API_KEY = os.getenv("OMLX_API_KEY", "")
OMLX_EMBEDDING_URL = "http://127.0.0.1:8000/v1/embeddings"
EMBEDDING_MODEL_NAME = "embeddinggemma-2-bf16"


class OMLXEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        headers = {
            "Authorization": f"Bearer {OMLX_API_KEY}",
            "Content-Type": "application/json"
        }
        embeddings = []
        for item in input:
            if item.startswith("data:image/"):
                payload = {
                    "model": EMBEDDING_MODEL_NAME,
                    "input": item  
                    # OBS: Om din oMLX-server kräver ett annat format för bilder (t.ex. "image": item), 
                    # är det här vi justerar det senare. De flesta kompatibla API:er löser detta via 'input'.
                }
            else:
                payload = {
                "model": EMBEDDING_MODEL_NAME,
                "input": item
            }
            try:
                res = requests.post(OMLX_EMBEDDING_URL, json=payload, headers=headers)
                res.raise_for_status()
                vector = res.json()["data"][0]["embedding"]
                embeddings.append(vector)
            except Exception as e:
                print(f"Error during vectorization: {e}")
        return embeddings


print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
omlx_ef = OMLXEmbeddingFunction()

try:
    client.delete_collection(name="snuffles_docs")
except Exception:
    pass

collection = client.get_or_create_collection(name="snuffles_docs", embedding_function=omlx_ef)


def split_text_into_chunks(text: str, chunk_size: int = 750, overlap: int = 200):
    # """Splits text into overlapping segments to preserve context across boundaries."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
        
    return [c for c in chunks if c]


def index_folder(folder_path: str):
    print(f"Scanning folder: '{folder_path}'...")
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 1. Plain text & Markdown
        if filename.lower().endswith((".txt", ".md")):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                if content.strip():
                    chunks = split_text_into_chunks(content)
                    for idx, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({
                            "filename": filename,
                            "page": 1,
                            "chunk_index": idx
                        })
                        ids.append(f"{filename}_p1_c{idx}")

        # 2. PDF processing with page-level tracking
        
        elif filename.lower().endswith(".pdf"):
            try:
                reader = pypdf.PdfReader(file_path)
                total_pages = len(reader.pages)
                print(f"Processing PDF '{filename}' ({total_pages} pages)...")

                # Check if the PDF has embedded page labels (e.g. from LaTeX)
                has_page_labels = False
                try:
                    labels = reader.page_labels
                    if labels and len(labels) == total_pages:
                        has_page_labels = True
                except Exception:
                    has_page_labels = False

                for page_idx, page in enumerate(reader.pages):
                    # Use document page label if available, otherwise physical page number
                    if has_page_labels:
                        page_label = str(labels[page_idx])
                    else:
                        page_label = str(page_idx + 1)

                    page_text = page.extract_text() or ""
                    
                    if page_text.strip():
                        chunks = split_text_into_chunks(page_text)
                        for chunk_idx, chunk in enumerate(chunks):
                            documents.append(chunk)
                            metadatas.append({
                                "filename": filename,
                                "page": page_label,
                                "chunk_index": chunk_idx
                            })
                            ids.append(f"{filename}_p{page_label}_c{chunk_idx}")
        
            except Exception as e:
                print(f"Failed to read PDF '{filename}': {e}")
        # 3. Image processing (Convert to Base64)
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Processing Image '{filename}'...")
            try:
                with open(file_path, "rb") as image_file:
                    # Read the binary image data and encode it to a base64 string
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # Create the standard Data URI format required by most vision/embedding APIs
                    file_ext = "jpeg" if filename.lower().endswith(".jpg") else filename.split('.')[-1].lower()
                    base64_image = f"data:image/{file_ext};base64,{encoded_string}"
                    
                    documents.append(base64_image)
                    metadatas.append({
                        "filename": filename,
                        "page": "Image",
                        "chunk_index": 0,
                        "is_image": True  # Denna flagga blir livsviktig i chatten sen!
                    })
                    ids.append(f"{filename}_img")
                    
            except Exception as e:
                print(f"Failed to read image '{filename}': {e}")
    if documents:
        print(f"\nSending {len(documents)} text chunks to oMLX for embedding...")
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[OK] Successfully indexed {len(documents)} entries into 'snuffles_docs'.")
    else:
        print("No documents found to index.")


if __name__ == "__main__":
    folder_to_index = "snuffles_files"
    index_folder(folder_to_index)