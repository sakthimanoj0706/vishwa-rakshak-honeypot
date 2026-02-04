import os
import re
import time
import logging
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, Header, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

# Gemini (google-genai) SDK
from google import genai

# ---------------- CONFIG ----------------

API_KEY_SECRET = os.getenv("MY_API_KEY", "vishwa-rakshak-2026")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Mandatory final result callback from problem statement
GUVI_CALLBACK_URL = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VishwaRakshak")

# --------------- GEMINI CLIENT ----------

client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini (google-genai) client initialized")
    except Exception as e:
        logger.error(f"❌ Gemini init failed: {e}")
else:
    logger.warning("⚠️ GEMINI_API_KEY not set. Using fallback replies only.")

# --------------- DATA MODELS ------------


class MessagePart(BaseModel):
    sender: str
    text: str
    timestamp: Optional[int] = None


class HackathonRequest(BaseModel):
    """
    Matches the request body from the hackathon email:

    {
      "sessionId": "uuid",
      "message": { "sender": "scammer", "text": "...", "timestamp": 1769... },
      "conversationHistory": [ {sender, text, timestamp}, ... ],
      "metadata": { "channel": "SMS", "language": "English", "locale": "IN" }
    }
    """
    sessionId: str
    message: MessagePart
    conversationHistory: List[MessagePart] = []
    metadata: Optional[Dict[str, Any]] = None


# --------------- HELPERS ----------------


def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


def extract_intelligence(text: str) -> Dict[str, List[str]]:
    """Regex-only extraction. No hallucinations."""
    upi_pattern = r"[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}"
    acc_pattern = r"\b\d{9,18}\b"
    url_pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:[^\s<\"]*)"
    phone_pattern = r"\+91\d{10}|\b\d{10}\b"

    scam_keywords = ["urgent", "verify", "blocked", "expired", "kyc", "prize", "lottery"]

    found_keywords = [k for k in scam_keywords if k in text.lower()]

    return {
        "bankAccounts": list(set(re.findall(acc_pattern, text))),
        "upiIds": list(set(re.findall(upi_pattern, text))),
        "phishingLinks": list(set(re.findall(url_pattern, text))),
        "phoneNumbers": list(set(re.findall(phone_pattern, text))),
        "suspiciousKeywords": found_keywords,
    }


def generate_agent_reply(history: List[MessagePart], current_msg: str) -> str:
    """‘Ramesh Uncle’ Hinglish reply, using Gemini if available."""
    if not client:
        # Fallback when Gemini key missing / broken
        return "Beta, mera phone thoda slow hai. Aap payment details fir se bhejoge?"

    chat_log = "\n".join(
        f"{m.sender.upper()}: {m.text}" for m in history[-5:]
    )

    system_prompt = """
SYSTEM: You are "Ramesh Uncle", a confused 65-year-old Indian man.
CONTEXT: You are talking to a scammer trying to steal money.
GOAL: Waste their time and gently push them to reveal UPI ID or bank account.
STYLE: Indian English (Hinglish). Short sentences. Use words like "Beta", "Arre", "network slow".
AVOID: Never say you are a bot or AI. Stay in character.
RESPONSE: Max 2 sentences.
"""

    full_prompt = (
        f"{system_prompt}\n\n"
        f"HISTORY:\n{chat_log}\n\n"
        f"SCAMMER: {current_msg}\n"
        f"RAMESH UNCLE:"
    )

    try:
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=full_prompt,
        )
        # google-genai returns output_text on the response
        reply = (resp.output_text or "").strip()
        if not reply:
            reply = "Arre beta, clearly bolo na. Kahan bhejna hai paise?"
        return reply
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return "Hello beta, mujhe theek se samajh nahi aaya. Payment kahan bhejna hai?"


def report_to_guvi(session_id: str, intel: Dict[str, List[str]], msg_count: int, notes: str):
    """Background callback to GUVI with final scam intelligence."""
    payload = {
        "sessionId": session_id,
        "scamDetected": True,
        "totalMessagesExchanged": msg_count,
        "extractedIntelligence": intel,
        "agentNotes": notes,
    }

    try:
        logger.info(f"📤 Reporting to GUVI for session={session_id}")
        res = requests.post(GUVI_CALLBACK_URL, json=payload, timeout=5)
        logger.info(f"✅ GUVI callback status={res.status_code} body={res.text}")
    except Exception as e:
        logger.error(f"❌ GUVI callback failed: {e}")


# --------------- FASTAPI APP ------------

app = FastAPI(title="Vishwa-Rakshak Honeypot")


@app.get("/")
def root():
    return {"status": "ok", "message": "Vishwa-Rakshak Honeypot live"}


@app.post("/honeypot")
async def honeypot_endpoint(
    req: HackathonRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Main Honey-Pot endpoint used by GUVI.

    ✅ Input: HackathonRequest (sessionId, message, conversationHistory, metadata)
    ✅ Output: { "status": "success", "reply": "<agent text>" }
    ✅ Side-effect: Background callback to GUVI when enough intel is collected.
    """
    start = time.time()

    session_id = req.sessionId
    incoming_text = req.message.text
    history = req.conversationHistory

    # 1) Extract intelligence from full text (history + current message)
    full_text_parts = [m.text for m in history]
    full_text_parts.append(incoming_text)
    full_text = " ".join(full_text_parts)

    intel = extract_intelligence(full_text)

    # 2) Generate reply from Ramesh Uncle
    agent_reply = generate_agent_reply(history, incoming_text)

    # 3) Decide whether to trigger the mandatory callback
    has_critical_intel = (
        len(intel["bankAccounts"]) > 0
        or len(intel["upiIds"]) > 0
        or len(intel["phishingLinks"]) > 0
    )

    msg_count = len(history) + 1

    if has_critical_intel or msg_count > 5:
        # build agent notes
        if has_critical_intel:
            notes = "Scam detected. Financial / link intelligence extracted."
        else:
            notes = "Scam pattern detected via language/urgency. Limited explicit intel."

        # run callback in background (does NOT block this response)
        background_tasks.add_task(
            report_to_guvi,
            session_id,
            intel,
            msg_count,
            notes,
        )

    latency = int((time.time() - start) * 1000)
    logger.info(
        f"[HONEYPOT] session={session_id} messages={msg_count} "
        f"intel={intel} latency={latency}ms"
    )

    # 🔴 IMPORTANT: EXACT RESPONSE FORMAT REQUIRED BY GUVI
    return {
        "status": "success",
        "reply": agent_reply,
    }
