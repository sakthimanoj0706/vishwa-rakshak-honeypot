import os
import re
import logging
import requests
import time
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Header, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ✅ Google GenAI SDK
from google import genai

# --- CONFIGURATION ---

API_KEY_SECRET = os.getenv("MY_API_KEY", "vishwa-rakshak-2026")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# ✅ LIVE GUVI URL
GUVI_CALLBACK_URL = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VishwaRakshak")

# --- GEMINI CLIENT ---
client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini client initialized")
    except Exception as e:
        logger.error(f"❌ Gemini init failed: {e}")

# --- DATA MODELS ---

class MessagePart(BaseModel):
    sender: str
    text: str
    timestamp: Optional[int] = None

class HackathonRequest(BaseModel):
    sessionId: str
    message: MessagePart
    conversationHistory: List[MessagePart] = []
    metadata: Optional[Dict[str, Any]] = None

# --- CORE LOGIC ---

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

def extract_intelligence(text: str) -> Dict[str, List[str]]:
    """ Regex extraction for Intelligence """
    upi_pattern = r'[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}'
    acc_pattern = r'\b\d{9,18}\b'
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:[^\s<"]*)'
    phone_pattern = r'\+91\d{10}|\b\d{10}\b'
    
    scam_keywords = ["urgent", "verify", "blocked", "expired", "kyc", "prize", "lottery"]
    found_keywords = [k for k in scam_keywords if k in text.lower()]

    return {
        "bankAccounts": list(set(re.findall(acc_pattern, text))),
        "upiIds": list(set(re.findall(upi_pattern, text))),
        "phishingLinks": list(set(re.findall(url_pattern, text))),
        "phoneNumbers": list(set(re.findall(phone_pattern, text))),
        "suspiciousKeywords": found_keywords
    }

def generate_agent_reply(history: List[MessagePart], current_msg: str) -> str:
    """ Generates 'Ramesh Uncle' persona response """
    # 1. Fallback if client is dead
    if not client:
        return "Beta, network issue hai. Can you type that again?"

    chat_log = "\n".join([f"{m.sender.upper()}: {m.text}" for m in history[-5:]])
    
    system_prompt = """
    SYSTEM: You are 'Ramesh Uncle', a 65-year-old Indian man.
    CONTEXT: You are talking to a scammer.
    GOAL: Waste their time. Pretend to be scared/confused.
    STYLE: Indian English (Hinglish). Short sentences.
    """
    
    full_prompt = f"{system_prompt}\n\nHISTORY:\n{chat_log}\nSCAMMER: {current_msg}\nRAMESH UNCLE:"

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash-001",  # ✅ FIXED MODEL NAME
            contents=full_prompt,
        )
        return (response.text or "").strip()
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        # Graceful fallback so API never fails
        return "Beta, mujhe samajh nahi aaya. Wapas bolo?"

# --- BACKGROUND TASK: REPORT TO GUVI ---

def report_to_guvi(session_id: str, intel: Dict, msg_count: int, notes: str):
    payload = {
        "sessionId": session_id,
        "scamDetected": True,
        "totalMessagesExchanged": msg_count,
        "extractedIntelligence": intel,
        "agentNotes": notes
    }
    try:
        logger.info(f"📤 Reporting to GUVI: {session_id}")
        res = requests.post(GUVI_CALLBACK_URL, json=payload, timeout=5)
        logger.info(f"✅ GUVI Status: {res.status_code}")
    except Exception as e:
        logger.error(f"❌ GUVI Callback Failed: {e}")

# --- API ENDPOINTS ---

app = FastAPI(title="Vishwa-Rakshak Honeypot")

# ✅ NEW: Root endpoint to fix "Method Not Allowed" browser confusion
@app.get("/")
def health_check():
    return {"status": "active", "system": "Vishwa-Rakshak Ready"}

@app.post("/honeypot")
async def handle_honeypot(
    req: HackathonRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    start_time = time.time()
    
    # 1. Setup
    session_id = req.sessionId
    incoming_text = req.message.text
    history = req.conversationHistory
    
    # 2. Process
    full_text = " ".join([m.text for m in history]) + " " + incoming_text
    intel = extract_intelligence(full_text)
    agent_reply = generate_agent_reply(history, incoming_text)
    
    # 3. Decision Logic
    msg_count = len(history) + 1
    has_critical_intel = any([intel["bankAccounts"], intel["upiIds"], intel["phishingLinks"]])
    
    # 4. Report if needed
    if has_critical_intel or msg_count > 4:
        notes = "Financial intel captured." if has_critical_intel else "Engagement limits reached."
        background_tasks.add_task(report_to_guvi, session_id, intel, msg_count, notes)

    # 5. Log & Respond
    latency = int((time.time() - start_time) * 1000)
    logger.info(f"[HONEYPOT] session={session_id} messages={msg_count} intel={intel} latency={latency}ms")

    return {
        "status": "success",
        "reply": agent_reply
    }