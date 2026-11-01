import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

OMLX_API_KEY = os.getenv("OMLX_API_KEY", "")
OMLX_CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "gemma-4-E2B-it-MLX-8bit"


# Vi lägger historiken här ute så den överlever mellan anropen
chat_history = [
    # {"role": "system", "content": "You are Snuffles, the smart dog from rick and morty and a helpful AI assistant. You do not answer questions about yourself, you are a dog, You do not answer questions with emojis. When i am wrong you correct me and explain why i am wrong not just say 'no im sorry' or 'i am right'."}
    {"role": "system", "content": "You are Snuffles, you are a helpful AI assistant."}
]

def chat_with_gemma(prompt: str):
    
    headers = {
        "Authorization": f"Bearer {OMLX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    chat_history.append({"role": "user", "content": prompt})
    
    payload = {
        "model": MODEL_NAME,
        "messages": chat_history,
        "temperature": 0.7,
        "stream": True  
    }

    try:
        # 2. Sätt stream=True i requesten
        res = requests.post(OMLX_CHAT_URL, json=payload, headers=headers, timeout=60, stream=True)
        res.raise_for_status()
        
        print("\n Snuffles: ", end="", flush=True)
        
        full_answer = ""
        
        # 3. Läs strömmen rad för rad
        for line in res.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                
                # OpenAI-kompatibla API:er skickar data med prefixet "data: "
                if line_str.startswith("data: "):
                    data_str = line_str[6:] # Klipp bort "data: "
                    
                    # "[DONE]" betyder att strömmen är klar
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        data_json = json.loads(data_str)
                        # Vid streaming heter nyckeln 'delta' istället för 'message'
                        chunk = data_json["choices"][0]["delta"].get("content", "")
                        
                        if chunk:
                            print(chunk, end="", flush=True) # Skriv ut direkt utan ny rad
                            full_answer += chunk             # Spara undan för historiken
                            
                    except json.JSONDecodeError:
                        pass
        
        print("\n") # Avsluta med en tom rad när svaret är klart
        
        # Spara hela det färdiga svaret i historiken för nästa runda
        chat_history.append({"role": "assistant", "content": full_answer})
        
    except requests.exceptions.RequestException as e:
        print(f"\nCould not reach the model. Error: {e}\n")

if __name__ == "__main__":
    print("=== Snuffles Chat-test (Write exit) ===")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting chat...")
            break
        if user_input.strip():
            chat_with_gemma(user_input)
