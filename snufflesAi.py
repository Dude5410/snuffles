import os
import requests
import json
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

# --- Settings ---
OMLX_API_KEY = os.getenv("OMLX_API_KEY", "")
OMLX_CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions"
OMLX_EMBEDDING_URL = "http://127.0.0.1:8000/v1/embeddings"
OMLX_RERANK_URL = "http://127.0.0.1:8000/v1/rerank"  # Assuming standard local rerank endpoint

MODEL_NAME = "gemma-4-E2B-it-MLX-8bit"
EMBEDDING_MODEL_NAME = "embeddinggemma-2-bf16"

RERANKER_MODEL_NAME = "jina-reranker-v3-mlx" 

# --- How ChromaDB should embed and search texts ---
class OMLXEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

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
                print(f"Error when vectorizing: {e}")
        return embeddings
    
# --- Connect to the database ---
client = chromadb.PersistentClient(path="./chroma_db")
omlx_ef = OMLXEmbeddingFunction()

try:
    collection = client.get_collection(name="snuffles_docs", embedding_function=omlx_ef)
except ValueError:
    collection = None
    print("Warning: No ChromaDB collection found. Snuffles will have no memory.")

# --- Chat history ---
chat_history = [
    {"role": "system", "content": "You are Snuffles, you are a helpful AI assistant."}
]

def rerank_documents(query: str, documents: list, metadatas: list, distances: list) -> tuple:
    if not documents:
        return [], []

    text_docs = []
    text_metas = []
    image_docs = []
    image_metas = []

    # Återinför tröskelvärde enbart för bilder (ChromaDB-avstånd, lägre = bättre)
    IMAGE_DISTANCE_THRESHOLD = 0.60 

    # Separera text och bilder, och filtrera bort dåliga bilder direkt
    for doc, meta, dist in zip(documents, metadatas, distances):
        if meta.get('is_image', False):
            if dist < IMAGE_DISTANCE_THRESHOLD:
                image_docs.append(doc)
                image_metas.append(meta)
            else:
                print(f"Bild: {meta.get('filename')} | Avstånd: {dist:.4f} -> [Discarded by ChromaDB]")
        else:
            text_docs.append(doc)
            text_metas.append(meta)

    if not text_docs:
        return image_docs, image_metas

    headers = {
        "Authorization": f"Bearer {OMLX_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": RERANKER_MODEL_NAME,
        "query": query,
        "documents": text_docs
    }

    try:
        print(f"\n--- [Reranking {len(text_docs)} documents...] ---")
        res = requests.post(OMLX_RERANK_URL, json=payload, headers=headers, timeout=30)
        res.raise_for_status()
        
        response_data = res.json()
        results = response_data.get("results", [])
        
        if not results:
            print("Warning: Reranker returned empty results. Using default ChromaDB order.")
            return documents, metadatas

        results = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        ranked_docs = []
        ranked_metas = []
        
        # Höj denna marginellt om texter fortfarande slinker igenom vid korta hälsningar
        MIN_RERANK_SCORE = 0.20 
        
        for rank_item in results:
            original_idx = rank_item.get("index")
            score = rank_item.get("relevance_score", 0)
            meta = text_metas[original_idx]
            
            if score >= MIN_RERANK_SCORE:
                ranked_docs.append(text_docs[original_idx])
                ranked_metas.append(meta)
                print(f"File: {meta.get('filename')} | Score: {score:.4f} -> [Relevant]")
            else:
                print(f"File: {meta.get('filename')} | Score: {score:.4f} -> [Discarded]")
        
        print("---------------------------------\n")
        
        final_docs = ranked_docs + image_docs
        final_metas = ranked_metas + image_metas
        
        return final_docs, final_metas

    except requests.exceptions.RequestException as e:
        print(f"\n[FELSÖKNING] oMLX avvisade anropet: {e}")
        print("Falling back to standard ChromaDB results.")
        return documents, metadatas

def chat_with_gemma(prompt: str):
    global chat_history 
    
    headers = {
        "Authorization": f"Bearer {OMLX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Kollar om "@files" finns med i meddelandet
    use_rag = "@files" in prompt.lower()
    
    # Tvätta bort "@files" från texten så det inte förvirrar sökningen eller modellen
    clean_prompt = prompt.lower().replace("@files", "").strip()
    
    # Fallback om du enbart skrev "@files" utan någon fråga
    if not clean_prompt:
        clean_prompt = prompt
        
    context = ""
    used_sources = []
    retrieved_images = []
    matching_docs = []
    
    # Databassökningen körs ENBART om use_rag är True
    if collection and use_rag:
        # Step 1: Broad Retrieval
        results = collection.query(
            query_texts=[clean_prompt], # Använd den tvättade frågan
            n_results=15, 
            include=["documents", "distances", "metadatas"]
        )
        
        if results['documents'] and results['documents'][0]:
            candidate_docs = results['documents'][0]
            candidate_metas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(candidate_docs)
            candidate_distances = results['distances'][0]
            
            # Skicka den tvättade söksträngen till rerankern
            ranked_docs, ranked_metas = rerank_documents(clean_prompt, candidate_docs, candidate_metas, candidate_distances)

            # Step 3: Select top 4 highest-scoring documents
            top_docs = ranked_docs[:4]
            top_metas = ranked_metas[:4]
            
            for doc, meta in zip(top_docs, top_metas):
                source_file = meta.get('filename', 'Unknown file')
                page_num = meta.get('page', None)
                is_image = meta.get('is_image', False)
                
                source_label = f"{source_file} (Page {page_num})" if page_num else source_file
                
                if is_image:
                    matching_docs.append(f"[System notification: A relevant image file ({source_label}) was found. Base64 data hidden.]")
                    retrieved_images.append(doc)
                else:
                    matching_docs.append(doc)
            
                if source_label not in used_sources:
                    used_sources.append(source_label)
            
            if matching_docs:
                context = "\n\n---\n\n".join(matching_docs)
        else:
            print("No relevant documents found in ChromaDB.\n")

    # Build the augmented prompt
    # Build the augmented prompt
    if context:
        augmented_prompt = (
        f"Document Context:\n{context}\n\n"
        f"Question: {clean_prompt}\n\n"
        f"Instruction: Primarily use the provided documents if they contain the answer to the question."
        )
    else:
        augmented_prompt = clean_prompt

    # Attach images to the user prompt if any were retrieved
    if retrieved_images:
        user_content = [{"type": "text", "text": augmented_prompt}]
        for img_b64 in retrieved_images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": img_b64}
            })
        chat_history.append({"role": "user", "content": user_content})
    else:
        chat_history.append({"role": "user", "content": augmented_prompt})

    # SLIDING WINDOW MEMORY: Keep only the last 4 messages + system prompt
    if len(chat_history) > 5:
        chat_history = [chat_history[0]] + chat_history[-4:]
        
    payload = {
        "model": MODEL_NAME,
        "messages": chat_history,
        "temperature": 0.7,
        "stream": True
    }

    try:
        # Communication with oMLX server for Generation
        res = requests.post(OMLX_CHAT_URL, json=payload, headers=headers, timeout=60, stream=True)
        res.raise_for_status()
        
        print("\n Snuffles: ", end="", flush=True)
        full_answer = ""
        
        for line in res.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        if "choices" in data_json:
                            chunk = data_json["choices"][0]["delta"].get("content", "")
                            if chunk:
                                print(chunk, end="", flush=True)
                                full_answer += chunk
                        elif "error" in data_json:
                            print(f"\n[Server Error]: {data_json['error']}")
                            
                    except json.JSONDecodeError:
                        pass
        
        print("\n")
        if used_sources:
          print(" Sources used:")
          for src in used_sources:
            print(f"  • {src}")
          print()
          
        if full_answer:
            chat_history[-1]["content"] = clean_prompt 
            chat_history.append({"role": "assistant", "content": full_answer})
        
    except requests.exceptions.RequestException as e:
        print(f"\nCould not reach the model. Error: {e}\n")


if __name__ == "__main__":
    print("=== Snuffles AI===")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting chat...")
            break
        if user_input.strip():
            chat_with_gemma(user_input)
