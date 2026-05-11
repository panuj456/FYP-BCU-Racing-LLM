import time
import gc
import torch
from pathlib import Path
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import os, glob, re, random
import networkx as nx
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Core imports
from core.llm_engine import STGAdapter
from core.graph_processor import STGTokeniser
from core.llm_generate import LLM_RaceEngineer
from core.nlp_engine import NLP_Analysis
from core.telem_utils import TelemUtils

# This dictionary will hold all your NetworkX objects in RAM
sessions_db = {}
INSTRUCTIONS_PATH = Path("./data/race_engineer_context.md")
INSTRUCTIONS = INSTRUCTIONS_PATH.read_text() if INSTRUCTIONS_PATH.exists() else "You are a Race Engineer."
llm = LLM_RaceEngineer(host="http://ollama-server:11434", instructions_md=INSTRUCTIONS)
adapter = STGAdapter(instructions=INSTRUCTIONS)
nlp = NLP_Analysis()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs when the Docker container starts
    print("Loading pre-baked STG graphs into memory...")
    sessions_db.update(TelemUtils.load_all_session_graphs("./data")) #loads all pickle files
    print(f"Loaded sessions: {list(sessions_db.keys())}")
    yield
    # Clean up on shutdown
    sessions_db.clear()
    gc.collect()
    # 3. Flush the ROCm/HIP memory pool
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

# Mount the folder where your index.html lives
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Serve the HTML file when someone visits http://localhost:8000
@app.get("/")
async def serve_ui():
    return FileResponse('frontend/index.html')

@app.get("/analyze/{session_id}")
async def analyze_telemetry(session_id: str, query: str):
    G = sessions_db.get(session_id)
    if not G:
        return {"error": "Session graph not found"}
    # Pass G to your LLM logic...

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    all_rich_tokens = {}
    query = request.query
    print(f"User Query: {request.query}")
    if not query:
        return {"error": "No query provided"}
    # NLP defines intent
    intent = nlp.extract_intent(query) 
    print(f"Sessions found: {intent.get('sessions')}")
    print(f"Comparison flag: {intent.get('comparison')}")
    domain_key = intent.get('domain', 'default')
    print(f"Domain found: {domain_key}")
    
    tokeniser = STGTokeniser(intent=intent)
    print("Tokenisation...")
    session_ids = intent.get('sessions')
    for s_id in session_ids:
        lookup_id = f"session{s_id}" if not s_id.startswith("session") else s_id
        G = sessions_db.get(lookup_id)
        if G:
            # G saved to dictionary to handle multiple IDs
            # Max_Tokens at 10000 as cot_encode takes full sample and reduces sample size.
            all_rich_tokens[lookup_id] = tokeniser.tokenize_from_graph(G, max_tokens=10000) 
    flat_tokens = []
    for tokens in all_rich_tokens.values():
        flat_tokens.extend(tokens)
    
    # Adapter encode builds the base prompt
    print("Encoding Rich Tokens...")
    final_prompt = adapter.encode(flat_tokens, query)
    
    # Adapter cot_encode builds the CoT prompt
    # Removed 'self.' and extracted domain safely from intent
    print("Chain of Thought Prompt Encoding...")
    final_prompt = adapter.cot_encode(final_prompt, query, domain_key)
    
    # LLM responds
    print("Starting Race Engineer Generation...")
    start_time = time.perf_counter()
    llm_output = llm.generate(final_prompt)
    end_time = time.perf_counter()
    inference_duration = end_time - start_time
    print("--- RACE ENGINEER OUTPUT ---")
    
    # Return proper JSON dictionary
    return {
        "query": str(query),
        "time_to_infer_seconds": round(inference_duration, 2),
        # CHANGE THIS LINE: Use all_rich_tokens instead of flat_tokens
        "sessions_analyzed": list(all_rich_tokens.keys()), 
        "intent": intent,
        "insight": llm_output
    }