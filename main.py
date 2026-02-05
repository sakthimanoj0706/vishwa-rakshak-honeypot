import os
import re
import time
import logging
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Header, HTTPException, Depends, BackgroundTasks, Request

# ✅ Google GenAI SDK (google-genai)
from google import genai

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

API_KEY_SECRET = os.getenv("MY_API_KEY", "vishwa-rakshak-2026")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# GUVI mandatory callback URL
GUVI_CALLBACK_URL = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VishwaRakshak")

# ---------------------------------------------------------------------
# GEMINI CLIENT
# ---------------------------------------------------------------------

client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini client initialized")
    except Exception as e:
        logger.error(f"❌ Gemini init failed: {e}")
else:
    logger.warning("⚠️ GEMINI_API_KEY missing – using fallback replies only.")

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def verify_api_key(x_api_key: str = Header(None)):
    """Simple API key check on header `x-api-key`."""
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


def normalize_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make the request format very flexible so we never 422.

    Supports exactly the hackathon spec:

        {
          "sessionId": "...",
          "message": {"sender": "scammer", "text": "...", "timestamp": ...},
          "conversationHistory": [...]
        }

    and also tolerates older field names like conversation_id, sender_type, history.
    """
    # Session / conversation ID
    session_id = (
        body.get("sessionId")
        or body.get("conversation_id")
        or body.get("session_id")
        or "session-unknown"
    )

    # Current message text
    msg_obj = body.get("message") or {}
    text = (
        msg_obj.get("text")
        or msg_obj.get("message")
        or body.get("text")
        or ""
    )

    # Sender (optional; default scammer)
    sender = (
        msg_obj.get("sender")
        or msg_obj.get("sender_type")
        or "scammer"
    )

    # History can be "conversationHistory", "history", or None
    raw_history = (
        body.get("conversationHistory")
        or body.get("history")
        or []
    )

    history: List[Dict[str, str]] = []
    if isinstance(raw_history, list):
        for m in raw_history:
            if not isinstance(m, dict):
                continue
            h_sender = (
                m.get("sender")
                or m.get("sender_type")
                or "scammer"
            )
            h_text = m.get("text") or m.get("message") or ""
            if h_text:
                history.append({"sender": h_sender, "text": h_text})

    return {
        "session_id": session_id,
        "current_sender": sender,
        "current_text": text,
        "history": history,
    }


def extract_intelligence(text: str) -> Dict[str, List[str]]:
    """Regex-only intelligence extraction (no hallucinations)."""
    upi_pattern = r"[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}"
    acc_pattern = r"\b\d{9,18}\b"
    url_pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s<\"]*"
    phone_pattern = r"\+91\d{10}|\b\d{10}\b"

    scam_keywords = ["urgent", "verify", "blocked", "expired", "kyc", "prize", "lottery"]
    lowered = text.lower()
    found_keywords = [k for k in scam_keywords if k in lowered]

    return {
        "bankAccounts": list(set(re.findall(acc_pattern, text))),
        "upiIds": list(set(re.findall(upi_pattern, text))),
        "phishingLinks": list(set(re.findall(url_pattern, text))),
        "phoneNumbers": list(set(re.findall(phone_pattern, text))),
        "suspiciousKeywords": found_keywords,
    }


def generate_agent_reply(history: List[Dict[str, str]], current_msg: str) -> str:
    """Generate 'Ramesh Uncle' reply using Gemini; has safe fallback."""
    # Fallback if no Gemini client
    if not client:
        return "Beta, network issue hai. Thoda clearly firse likho na?"

    # Last few turns as plain text
    chat_log_lines = [
        f"{m['sender'].upper()}: {m['text']}"
        for m in history[-5:]
        if "sender" in m and "text" in m
    ]
    chat_log = "\n".join(chat_log_lines)

    system_prompt = """
SYSTEM: You are 'Ramesh Uncle', a 65-year-old Indian man.
CONTEXT: You are talking to a scammer trying to steal money.
GOAL: Waste their time and keep them engaged. Act confused & a bit scared.
STYLE: Indian English (Hinglish), short sentences, friendly tone.
DO NOT give any real personal details.
RESPONSE: 1-2 short sentences only.
"""

    prompt = (
        f"{system_prompt}\n\n"
        f"HISTORY:\n{chat_log}\n\n"
        f"SCAMMER: {current_msg}\n"
        f"RAMESH UNCLE:"
    )

    try:
        resp = client.models.generate_content(
            model="gemini-1.5-flash-001",  # ✅ correct model name for new SDK
            contents=prompt,
        )
        text = (resp.text or "").strip()
        if not text:
            raise ValueError("Empty model response")
        return text
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        # Graceful fallback so API still works
        return "Beta, mujhe samajh nahi aaya. Zara dheere se fir bolo?"


def report_to_guvi(session_id: str, intel: Dict[str, Any], msg_count: int, notes: str):
    """Background callback to GUVI – REQUIRED by problem statement."""
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
        logger.info(f"✅ GUVI callback status={res.status_code}")
    except Exception as e:
        logger.error(f"❌ GUVI callback failed: {e}")


# ---------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------

app = FastAPI(title="Vishwa-Rakshak Honeypot")


@app.get("/")
def health_check():
    """Simple health endpoint for browser / Render."""
    return {"status": "ok", "message": "Vishwa-Rakshak Honeypot live"}


@app.post("/honeypot")
async def honeypot_handler(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Main honeypot endpoint.

    IMPORTANT: Must always return:
        {"status": "success", "reply": "<string>"}
    """

    start = time.time()

    # 1) Read raw JSON, handle any format → avoid 422
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("JSON root must be object")
    except Exception as e:
        logger.error(f"Invalid JSON body: {e}")
        # Still return a valid 'success' so tester doesn't break
        return {"status": "success", "reply": "Beta, message thik se nahi aaya. Fir bhejo?"}

    norm = normalize_body(body)
    session_id = norm["session_id"]
    current_text = norm["current_text"]
    history = norm["history"]

    # If somehow no text, still reply safely
    if not current_text:
        return {
            "status": "success",
            "reply": "Beta, aapne kya likha? Mujhe kuch dikh nahi raha.",
        }

    # 2) Extract intelligence from full conversation
    full_text = " ".join([m["text"] for m in history] + [current_text])
    intel = extract_intelligence(full_text)

    # 3) Generate uncle reply
    reply = generate_agent_reply(history, current_text)

    # 4) Decide when to send callback
    msg_count = len(history) + 1
    has_critical_intel = bool(
        intel["bankAccounts"] or intel["upiIds"] or intel["phishingLinks"]
    )

    if has_critical_intel or msg_count > 4:
        notes = (
            "Scam detected. Financial / contact details extracted."
            if has_critical_intel
            else "Scam pattern observed. Conversation length threshold reached."
        )
        background_tasks.add_task(
            report_to_guvi,
            session_id=session_id,
            intel=intel,
            msg_count=msg_count,
            notes=notes,
        )

    latency_ms = int((time.time() - start) * 1000)
    logger.info(
        f"[HONEYPOT] session={session_id} messages={msg_count} "
        f"intel={intel} latency={latency_ms}ms"
    )

    # 5) FINAL required response format
    return {"status": "success", "reply": reply}
