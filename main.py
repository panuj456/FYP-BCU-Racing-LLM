import csv   # <-- was missing
import time
import gc
import torch
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import os, glob, re, random
import networkx as nx
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
import asyncio
from fastapi import HTTPException
import serial

# Core imports
from core.llm_engine import STGAdapter
from core.graph_processor import STGTokeniser
from core.llm_generate import LLM_RaceEngineer
from core.nlp_engine import NLP_Analysis
from core.telem_utils import TelemUtils
from core.telemetry_service import TelemetryService

# ── PATH RESOLUTION ──────────────────────────────────────────
# Use the directory this script lives in, NOT os.getcwd().
# This prevents relative-path failures when Docker's working
# directory doesn't match the project root.
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTBOX_PATH  = os.path.join(BASE_DIR, "data", "outbox")
DATA_DIR     = os.path.join(BASE_DIR, "data")
ARCHIVE_PATH  = os.path.join(BASE_DIR, "data", "archive")

print("--- PATH DIAGNOSTICS ---")
print(f"  __file__    : {__file__}")
print(f"  BASE_DIR    : {BASE_DIR}")
print(f"  OUTBOX_PATH : {OUTBOX_PATH}")
print(f"  outbox exists: {os.path.exists(OUTBOX_PATH)}")
if os.path.exists(OUTBOX_PATH):
    print(f"  outbox files : {os.listdir(OUTBOX_PATH)}")
print(f"  OUTBOX_PATH : {ARCHIVE_PATH}")
print(f"  outbox exists: {os.path.exists(ARCHIVE_PATH)}")
if os.path.exists(ARCHIVE_PATH):
    print(f"  outbox files : {os.listdir(ARCHIVE_PATH)}")
print("------------------------")

sessions_db = {}
INSTRUCTIONS_PATH = Path(os.path.join(DATA_DIR, "race_engineer_context.md"))
INSTRUCTIONS = INSTRUCTIONS_PATH.read_text() if INSTRUCTIONS_PATH.exists() else "You are a Race Engineer."
llm  = LLM_RaceEngineer(host="http://ollama-server:11434", instructions_md=INSTRUCTIONS)
adapter = STGAdapter(instructions=INSTRUCTIONS)
nlp  = NLP_Analysis()
telem_service = TelemetryService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print(f"--- INITIALIZING FASTAPI IN [{APP_MODE.upper()}] MODE ---")
    
    if APP_MODE == "edge":
        print("Connecting to local Mosquitto MQTT Broker...")
        loop = asyncio.get_running_loop()
        telem_service.initialize_mqtt(loop=loop)
        
        print("Loading pre-baked STG graphs into memory...")
        sessions_db.update(TelemUtils.load_all_session_graphs(DATA_DIR))
    
    yield # <--- The App runs here
    
    # --- SHUTDOWN ---
    print("Shutting down and cleaning memory...")
    if APP_MODE == "edge":
        telem_service.mqtt_client.loop_stop()
        telem_service.mqtt_client.disconnect()
        
    sessions_db.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)
APP_MODE = os.getenv("APP_MODE", "edge") # Defaults to local edge mode

if APP_MODE == "edge":
    print("Running in EDGE Mode (Base Station Laptop)")
    # 1. Initialize local MQTT subscriber to update NetworkX graphs mid-race
    # 2. Register local Ollama trigger endpoints
    # Start background loop tracking local data
    
elif APP_MODE == "vps":
    print("Running in VPS Mode (Hostinger Cloud)")
    # 1. Open endpoints to accept bulk ZIP/CSV historical syncs from outbox
    # 2. Register query endpoints for InfluxDB historical graphs
    @app.post("/api/sync-session")
    async def receive_historical_session():
        return {"status": "synchronized"}

if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def serve_ui():
    return FileResponse('frontend/index.html')


# ─────────────────────────────────────────────────────────
#  TELEMETRY WEBSOCKET
# ─────────────────────────────────────────────────────────
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected successfully.")
    try:
        while True:
            # Native non-blocking wait. Automatically sleeps/yields control
            # to the event loop until a new packet arrives in the queue.
            data = await telem_service.queue.get()
            await websocket.send_json(data)
            telem_service.queue.task_done()
    except WebSocketDisconnect:
        print("Client disconnected from telemetry feed.")
    except Exception as e:
        print(f"Error occurred in WebSocket loop: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


# ─────────────────────────────────────────────────────────
#  STATUS / SESSION CONTROL
# ─────────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    return {
        "status": "ONLINE",
        "app_mode": APP_MODE,
        "session_active": telem_service.session_active,
        "session_start_time": telem_service.session_start_time
    }

@app.post("/api/session/start")
async def start_session(payload: dict):
    telem_service.start_session(payload.get("session_name", "Session"))
    return {"message": "Session started"}

@app.post("/api/session/stop")
async def stop_session():
    telem_service.stop_session()
    return {"message": "Session stopped"}


# ─────────────────────────────────────────────────────────
#  HISTORICAL SESSIONS
#  FIX 1: Single route, returns a plain JSON array (not a dict)
# ─────────────────────────────────────────────────────────
@app.get("/api/debug")
async def debug_paths():
    """Hit this in your browser to instantly see what paths the backend resolves."""
    #outbox_files = []
    archive_files = [] #ARCHIVE_PATH
    #if os.path.exists(OUTBOX_PATH):
    if os.path.exists(ARCHIVE_PATH):
        #outbox_files = os.listdir(OUTBOX_PATH)
        archive_files = os.listdir(ARCHIVE_PATH)
    return { #left as outbox for now as will have to change index.html or api routing
        "base_dir":      BASE_DIR,
        "outbox_path":   ARCHIVE_PATH,
        "outbox_exists": os.path.exists(ARCHIVE_PATH),
        "outbox_files":  archive_files,
        "cwd":           os.getcwd(),
    }
    '''
    return {
        "base_dir":      BASE_DIR,
        "outbox_path":   OUTBOX_PATH,
        "outbox_exists": os.path.exists(OUTBOX_PATH),
        "outbox_files":  outbox_files,
        "cwd":           os.getcwd(),
    }
    '''

@app.get("/api/sessions")
async def list_sessions():
    """Returns a JSON array of CSV filenames in the outbox, newest first."""
    #if not os.path.exists(OUTBOX_PATH):
    if not os.path.exists(ARCHIVE_PATH):
        return []
    #files = [f for f in os.listdir(OUTBOX_PATH) if f.endswith(".csv")]
    files = [f for f in os.listdir(ARCHIVE_PATH) if f.endswith(".csv")]
    return sorted(files, reverse=True)   # plain list, not {"sessions": [...]}


# ─────────────────────────────────────────────────────────
#  SESSION DATA  — query-param version so frontend URL matches
#  FIX 2: /api/session-data?file=<filename>  (was /api/session/{filename})
#  FIX 3: Column names use lowercase keys matching the telemetry service
# ─────────────────────────────────────────────────────────
@app.get("/api/session-data")
async def get_session_data(file: str):
    """
    Returns the raw CSV as plain text so the frontend can parse it.
    The frontend handles column detection generically, so we just serve the file.
    """
    #filepath = os.path.join(OUTBOX_PATH, os.path.basename(file))  # basename prevents path traversal
    filepath = os.path.join(ARCHIVE_PATH, os.path.basename(file))
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Session file not found: {file}")
    return PlainTextResponse(open(filepath).read())


# ─────────────────────────────────────────────────────────
#  LLM RACE ENGINEER CHAT
# ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    all_rich_tokens = {}
    query = request.query
    print(f"User Query: {query}")
    if not query:
        return {"error": "No query provided"}

    if telem_service.session_active:
        active_short_name = telem_service.session_name.split('_')[0]
        for requested_session in intent.get('sessions', []):
            if str(active_short_name) in str(requested_session):
                return {
                    "query": str(query),
                    "insight": "AI Analysis Unavailable: The requested session is currently live. Please stop the session to allow graph processing before running LLM queries."
                }
    
    intent     = nlp.extract_intent(query)
    domain_key = intent.get('domain', 'default')
    tokeniser  = STGTokeniser(intent=intent)

    for s_id in intent.get('sessions', []):
        lookup_id = f"session{s_id}" if not s_id.startswith("session") else s_id
        G = sessions_db.get(lookup_id)
        if G:
            all_rich_tokens[lookup_id] = tokeniser.tokenize_from_graph(G, max_tokens=10000)

    flat_tokens = [tok for tokens in all_rich_tokens.values() for tok in tokens]
    final_prompt = adapter.encode(flat_tokens, query)
    final_prompt = adapter.cot_encode(final_prompt, query, domain_key)

    start_time = time.perf_counter()
    llm_output = llm.generate(final_prompt)
    inference_duration = time.perf_counter() - start_time

    return {
        "query": str(query),
        "time_to_infer_seconds": round(inference_duration, 2),
        "sessions_analyzed": list(all_rich_tokens.keys()),
        "intent": intent,
        "insight": llm_output
    }

@app.get("/analyze/{session_id}")
async def analyze_telemetry(session_id: str, query: str):
    G = sessions_db.get(session_id)
    if not G:
        return {"error": "Session graph not found"}