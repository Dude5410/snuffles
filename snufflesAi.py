from multiprocessing import context
import os
import requests
import json
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from typer import prompt

load_dotenv()

# --- Inställningar ---
OMLX_API_KEY = os.getenv("OMLX_API_KEY", "")
OMLX_CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions"
OMLX_EMBEDDING_URL = "http://127.0.0.1:8000/v1/embeddings"

MODEL_NAME = "gemma-4-E2B-it-MLX-8bit"
EMBEDDING_MODEL_NAME = "embeddinggemma-2-bf16"

# --- How ChromaDB should embed and search texts ---
class OMLXEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        # Fixing the warning about the missing __init__ method in the base class
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
    # Get the collection snuffles_docs if it exists, otherwise set collection to None
    collection = client.get_collection(name="snuffles_docs", embedding_function=omlx_ef)
except ValueError:
    collection = None
    print("Warning: No ChromaDB collection found. Snuffles will have no memory.")

# --- Chat history ---
chat_history = [
    {"role": "system", "content": "You are Snuffles, the smart dog from rick and morty and a helpful AI assistant. You do not answer questions about yourself, you are a dog, You do not answer questions with emojis. When i am wrong you correct me and explain why i am wrong not just say 'no im sorry' or 'i am right'."}
]

def chat_with_gemma(prompt: str):
    global chat_history # Allows us to overwrite the global chat_history variable for sliding window memory
    
    headers = {
        "Authorization": f"Bearer {OMLX_API_KEY}",
        "Content-Type": "application/json"
    }
    bypass_phrases = ["own knowledge", "egen kunskap", "utan dokument", "general knowledge"]
    skip_rag = any(phrase in prompt.lower() for phrase in bypass_phrases)
    
    context = ""
    used_sources = []
    if collection and not skip_rag:
        # Make chromadb to find distance between the prompt and the documents in the collection
        results = collection.query(
            query_texts=[prompt],
            n_results=4,
            include=["documents", "distances", "metadatas"]
        )
        
        # Print out the distances in terminal for debugging purposes
        print("\n--- [Relevance Analysis from documents (lower is better)] ---")
        if results['documents'] and results['documents'][0]:
            matching_docs = []
            
            # Threshold to determine if a document is relevant enough to include in the context
            DISTANCE_THRESHOLD = 0.52 
            
            for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
                source = results['metadatas'][0][i].get('filename', 'Unknown file') if results.get('metadatas') else 'Unknown file'
                accepted = dist < DISTANCE_THRESHOLD
                
                # ChromaDB returns distances where lower is better, so we check if the distance is below our threshold
                status_str = "Relevant" if accepted else "Not enough relevance (neglected)"
                print(f"Hit {i+1} | File: {source} | Distance: {dist:.4f} -> [{status_str}]")
                
                if accepted:
                    matching_docs.append(doc)
                
                    # Extract filename and page number from metadata
                    meta = results['metadatas'][0][i] if results.get('metadatas') else {}
                    source_file = meta.get('filename', 'Unknown file')
                    page_num = meta.get('page', None)
                
                    source_label = f"{source_file} (Page {page_num})" if page_num else source_file
                if source_label not in used_sources:
                    used_sources.append(source_label)

            print("---------------------------------\n")
            
            if matching_docs:
                context = "\n\n---\n\n".join(matching_docs)
        else:
            print("No relevant documents found.")
            print("---------------------------------\n")
    # Build the augmented prompt with context if available
    if context:
        
        augmented_prompt = (
        f"Document Context:\n{context}\n\n"
        f"Question: {prompt}\n\n"
        f"Instruction: If the user explicitly asks you to answer from your own knowledge "
        f"or to ignore the context, disregard the documents above completely. Otherwise, "
        f"primarily use the provided documents if they contain the answer to the question."
        )
    else:
        # If no context is found, we just use the original prompt and models own knowledge
        augmented_prompt = prompt
    chat_history.append({"role": "user", "content": augmented_prompt})
    
    # SLIDING WINDOW MEMORY: Keep only the last 4 messages + system prompt to avoid exceeding token limits
    # Keeps the first system message (index 0) and the last 4 messages from the user and assistant
    if len(chat_history) > 5:
        chat_history = [chat_history[0]] + chat_history[-4:]
    
    payload = {
        "model": MODEL_NAME,
        "messages": chat_history,
        "temperature": 0.7,
        "stream": True
    }

    try:
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
                        
                        # --- FIX 2: SÄKER JSON-PARSNING ---
                        # Kontrollerar om servern skickade riktig chatt-data eller ett felmeddelande
                        if "choices" in data_json:
                            chunk = data_json["choices"][0]["delta"].get("content", "")
                            if chunk:
                                print(chunk, end="", flush=True)
                                full_answer += chunk
                        elif "error" in data_json:
                            print(f"\n[Serverfel]: {data_json['error']}")
                            
                    except json.JSONDecodeError:
                        pass
        
        print("\n")
        if used_sources:
          print(" Sources used:")
          for src in used_sources:
            print(f"  • {src}")
          print()
        # If we got a full answer, we update the last user message to be the original prompt and append the assistant's response
        if full_answer:
            chat_history[-1]["content"] = prompt 
            chat_history.append({"role": "assistant", "content": full_answer})
        
    except requests.exceptions.RequestException as e:
        print(f"\nCould not reach the model. Error: {e}\n")


if __name__ == "__main__":
    print("=== Snuffles AI ===")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting chat...")
            break
        if user_input.strip():
            chat_with_gemma(user_input)