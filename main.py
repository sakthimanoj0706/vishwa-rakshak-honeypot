import os
import time
import uuid
import re
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import redis

# --- CONFIGURATION ---
API_KEY_SECRET = os.getenv("MY_API_KEY", "vishwa-rakshak-2026")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VishwaRakshak")

# Setup Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logger.warning("⚠️ GEMINI_API_KEY missing. Agent will be brain-dead.")
    model = None

# Setup Redis (Safely)
redis_client = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        logger.info("✅ Redis Connected")
    except Exception as e:
        logger.error(f"❌ Redis Connection Failed: {e}")

# Fallback Memory (if Redis fails)
in_memory_store: Dict[str, List[str]] = {}

app = FastAPI(title="Vishwa-Rakshak: Agentic Honey-Pot")

# --- DATA MODELS ---
class Message(BaseModel):
    text: str
    sender_type: Optional[str] = "scammer"
    timestamp: Optional[str] = None

class HoneypotRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: Message
    history: Optional[List[Message]] = None

# --- CORE LOGIC ---

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

def get_history(conv_id: str) -> List[str]:
    """Retrieve chat history from Redis or Memory"""
    if redis_client:
        try:
            return redis_client.lrange(f"chat:{conv_id}", 0, -1)
        except:
            return []
    return in_memory_store.get(conv_id, [])

def save_history(conv_id: str, user_msg: str, agent_msg: str):
    """Save to Redis with 1-hour TTL (Privacy/Storage fix)"""
    if redis_client:
        try:
            redis_client.rpush(f"chat:{conv_id}", f"Scammer: {user_msg}")
            redis_client.rpush(f"chat:{conv_id}", f"Ramesh_Uncle: {agent_msg}")
            redis_client.expire(f"chat:{conv_id}", 3600) # 1 hour TTL
        except:
            pass
    else:
        # Fallback
        if conv_id not in in_memory_store:
            in_memory_store[conv_id] = []
        in_memory_store[conv_id].extend([f"Scammer: {user_msg}", f"Ramesh_Uncle: {agent_msg}"])

def extract_intelligence(text: str) -> Dict[str, List[str]]:
    """
    STRICT REGEX EXTRACTION. 
    We do NOT ask the LLM to extract data, because LLMs hallucinate.
    We only trust what we can grep from the raw text.
    """
    # UPI IDs (e.g., name@okaxis)
    upi_pattern = r'[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}'
    # Bank Accounts (9-18 digits, avoiding simple 4-6 digit OTPs)
    acc_pattern = r'\b\d{9,18}\b'
    # URLs
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

    return {
        "upi_ids": list(set(re.findall(upi_pattern, text))),
        "bank_accounts": list(set(re.findall(acc_pattern, text))),
        "phishing_urls": list(set(re.findall(url_pattern, text)))
    }

def fast_scam_filter(text: str) -> float:
    """Cheap keyword filter to save LLM costs"""
    keywords = ["lottery", "winner", "prize", "kyc", "block", "verify", "pay", "urgent", "expired"]
    text_lower = text.lower()
    score = 0.0
    for k in keywords:
        if k in text_lower:
            score += 0.2
    
    # Boost score if UPI pattern found
    if "@" in text and ("ok" in text or "pay" in text):
        score += 0.5
        
    return min(score, 0.99)

def generate_agent_reply(history: List[str], current_msg: str) -> str:
    """The Brain: Gemini 1.5 Flash"""
    if not model:
        return "Beta, my internet is down. Can you message later?"

    # The "Sentinel" System Prompt
    prompt = """
    SYSTEM: You are 'Ramesh Uncle', a 65-year-old retired clerk living in Pune.
    
    YOUR MISSION:
    1. A scammer is messaging you. Pretend to be gullible.
    2. Your SECRET GOAL is to get their UPI ID or Bank Account Number.
    3. Stall them. Say your app is not working, or you are confused.
    4. Speak in Indian English ("Hinglish"). Use words like "Beta", "Arre", "Network slow".
    5. SAFETY: If they ask if you are a bot, say "What is bot? I am just an old man."
    
    Keep response SHORT (max 2 sentences).
    """
    
    context = "\n".join(history[-6:]) # Only last 6 turns to save tokens
    full_prompt = f"{prompt}\n\nCHAT HISTORY:\n{context}\n\nSCAMMER: {current_msg}\nRAMESH UNCLE:"
    
    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return "Beta, I didn't understand. Can you send the payment details again?"

# --- ENDPOINTS ---

@app.get("/")
def keep_alive():
    return {"status": "Vishwa-Rakshak Active", "engine": "Gemini-Flash"}

@app.post("/honeypot")
async def handle_honeypot(req: HoneypotRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()
    
    # 1. Detect Scam
    confidence = fast_scam_filter(req.message.text)
    is_scam = confidence > 0.4 # Threshold
    
    # 2. Get History
    conv_id = req.conversation_id or str(uuid.uuid4())
    history_list = get_history(conv_id)
    
    # 3. Agent Logic
    agent_reply = None
    extracted_intel = { "upi_ids": [], "bank_accounts": [], "phishing_urls": [] }
    
    if is_scam:
        # Generate Reply
        agent_reply = generate_agent_reply(history_list, req.message.text)
        
        # Save Interaction
        save_history(conv_id, req.message.text, agent_reply)
        
        # Extract Intelligence (From FULL history to catch things mentioned earlier)
        full_text = " ".join(history_list) + " " + req.message.text
        extracted_intel = extract_intelligence(full_text)
    
    # 4. Response Construction (Strict JSON for Hackathon)
    return JSONResponse(content={
        "scam_detected": is_scam,
        "confidence_score": confidence,
        "agent_action": {
            "should_respond": is_scam,
            "response_text": agent_reply,
            "persona": "Ramesh_Uncle_v1"
        },
        "extracted_intelligence": extracted_intel,
        "conversation_metrics": {
            "turn_count": len(history_list) // 2,
            "latency_ms": int((time.time() - start_time) * 1000)
        }
    })