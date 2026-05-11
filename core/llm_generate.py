import ollama
from ollama import Client
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()

class LLM_RaceEngineer:
    # 'host' must come before 'model' because it doesn't have a default value
    def __init__(self, host: str, model: str = "gemma3:4b", instructions_md: str = ""):
        # The Client object handles the /api/chat pathing automatically
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma3:4b")
        self.client = Client(host=self.host)
        self.instructions = instructions_md

    def generate(self, user_prompt: str) -> str:
        print(f"Requesting analysis from model: {self.model}")
        
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": user_prompt}
        ]

        options = {
            "temperature": 0.0,
            "num_predict": 2048,
            "num_ctx": 16384,
            "stop": ["In Session", "<pad>"],
            "presence_penalty": 0.0,
            "frequency_penalty": 1.0,
            "top_k": 60,
            "top_p": 0.1,
        }

        try:
            # We use .chat() instead of requests.post to avoid manual URL errors
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=False
            )
            return response['message']['content']
            
        except Exception as e:
            return f"Ollama Connection Error: {str(e)}"