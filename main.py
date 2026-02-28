"""
ThoughtSort API — FastAPI backend
"""

import os
import json
import re
import logging
from datetime import datetime, timezone
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore, auth
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ── LOGGING ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("thoughtsort")

# ── FIREBASE INIT ─────────────────────────────────────────────
_cred_json = os.environ.get("FIREBASE_CREDENTIALS")
if _cred_json:
    _cred_dict = json.loads(_cred_json)
    cred = credentials.Certificate(_cred_dict)
else:
    cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred)
db = firestore.client()

# ── GEMINI INIT ───────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
genai.configure(api_key=GEMINI_API_KEY)

# ── FASTAPI APP ───────────────────────────────────────────────
app = FastAPI(title="ThoughtSort API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── AUTH DEPENDENCY ───────────────────────────────────────────
async def get_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except Exception as e:
        log.warning(f"Auth failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ── PYDANTIC MODELS ───────────────────────────────────────────
class AppendRequest(BaseModel):
    text: str
    timestamp: str

class SettingsRequest(BaseModel):
    known_tags: list[str] = []

class AnnotateRequest(BaseModel):
    text: str
    known_tags: list[str] = []

class AmalgamateRequest(BaseModel):
    tag: str
    notes: list[str]
    known_tags: list[str] = []

# ── HEALTH CHECK ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "ThoughtSort API"}

# ── ANNOTATE ─────────────────────────────────────────────────
@app.post("/annotate")
async def annotate_note(body: AnnotateRequest, uid: str = Depends(get_user_id)):
    """Annotate a single note — return tags only."""
    known_tags_str = ""
    if body.known_tags:
        known_tags_str = f"KNOWN TAGS (prefer these): {', '.join('#' + t for t in body.known_tags)}\n\n"

    prompt = (
        f"Tag this personal note with 1-4 topic tags.\n\n"
        f"{known_tags_str}"
        f"RULES:\n"
        f"- Prefer known tags. Only create new tags (lowercase, hyphens) if nothing fits.\n"
        f"- No # prefix. Reflect the actual topic, not generic words like 'note' or 'text'.\n"
        f"- Return a JSON object with a single 'tags' array of strings.\n\n"
        f"NOTE: {body.text}"
    )

    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=256,
                response_mime_type="application/json"
            )
        )
        raw = response.text.strip()
        log.info(f"annotate raw response: {raw[:200]}")
        result = json.loads(raw)
        # Normalise — ensure tags is a list of strings
        tags = result.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).lower().strip().lstrip("#") for t in tags if t]
        return {"tags": tags, "aiNote": ""}
    except Exception as e:
        log.error(f"annotate error: {e}")
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")


# ── AMALGAMATE ────────────────────────────────────────────────
@app.post("/amalgamate")
async def amalgamate(body: AmalgamateRequest, uid: str = Depends(get_user_id)):
    """Synthesize all notes in a tag into a single summary."""
    notes_text = "\n\n---\n\n".join(body.notes)
    known_tags_str = ""
    if body.known_tags:
        known_tags_str = f"KNOWN TAGS: {', '.join('#' + t for t in body.known_tags)}\n\n"

    prompt = f"""You are synthesizing a person's thoughts on the topic "#{body.tag}".

{known_tags_str}Read all the notes below and write a clear, coherent synthesis that:
- Identifies the main themes and recurring ideas
- Notes any contradictions or evolution of thinking
- Highlights any actionable insights or decisions
- Is written in second person ("You seem to be thinking about...")
- Is 2-4 paragraphs long

NOTES:
{notes_text}"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                max_output_tokens=1024
            )
        )
        return {"summary": response.text.strip()}
    except Exception as e:
        log.error(f"amalgamate error: {e}")
        raise HTTPException(status_code=500, detail=f"Amalgamate error: {e}")


# ── LEGACY ENDPOINTS (kept for compatibility) ─────────────────
@app.post("/inbox/append")
async def append_to_inbox(body: AppendRequest, uid: str = Depends(get_user_id)):
    entry = {
        "text":      body.text,
        "timestamp": body.timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    db.collection("users").document(uid).collection("inbox").add(entry)
    return {"status": "ok"}

@app.get("/notes")
async def get_notes(uid: str = Depends(get_user_id)):
    docs = (
        db.collection("users").document(uid).collection("notes")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .stream()
    )
    notes = []
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        notes.append(d)
    return {"notes": notes}

@app.get("/notes/{note_id}")
async def get_note(note_id: str, uid: str = Depends(get_user_id)):
    doc = db.collection("users").document(uid).collection("notes").document(note_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Note not found")
    d = doc.to_dict()
    d["id"] = doc.id
    return d

@app.post("/settings")
async def save_settings(body: SettingsRequest, uid: str = Depends(get_user_id)):
    db.collection("users").document(uid).set({"known_tags": body.known_tags}, merge=True)
    return {"status": "ok"}

@app.get("/settings")
async def get_settings(uid: str = Depends(get_user_id)):
    doc = db.collection("users").document(uid).get()
    if not doc.exists:
        return {"known_tags": []}
    return doc.to_dict()

@app.post("/sort")
async def sort_notes(uid: str = Depends(get_user_id)):
    inbox_ref  = db.collection("users").document(uid).collection("inbox")
    inbox_docs = list(inbox_ref.stream())
    if not inbox_docs:
        return {"status": "ok", "message": "Inbox is empty", "count": 0}
    entries = [{"id": d.id, "text": d.to_dict().get("text",""), "timestamp": d.to_dict().get("timestamp","")} for d in inbox_docs]
    inbox_text = "\n\n".join(f"### {e['timestamp']}\n{e['text']}" for e in entries)
    user_doc   = db.collection("users").document(uid).get()
    known_tags = user_doc.to_dict().get("known_tags", []) if user_doc.exists else []
    sorted_notes = await call_gemini(inbox_text, known_tags)
    notes_ref = db.collection("users").document(uid).collection("notes")
    sort_run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    for note in sorted_notes:
        note["sort_run_id"] = sort_run_id
        note["created_at"]  = datetime.now(timezone.utc).isoformat()
        notes_ref.add(note)
    batch = db.batch()
    for doc in inbox_docs:
        batch.delete(inbox_ref.document(doc.id))
    batch.commit()
    return {"status": "ok", "count": len(sorted_notes)}

async def call_gemini(inbox_text: str, known_tags: list[str]) -> list[dict]:
    known_tags_str = f"KNOWN TAGS: {', '.join('#' + t for t in known_tags)}\n" if known_tags else ""
    prompt = f"""You are an intelligent personal thought organizer.

{known_tags_str}
Return ONLY a valid JSON array. Each object must have:
- originalText: string
- timestamp: string
- filename: string
- tags: array of strings (no # prefix)
- aiNote: string

INBOX:
{inbox_text}"""
    model    = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=8192,
            response_mime_type="application/json"
        )
    )
    raw = response.text.strip()
    try:
        notes = json.loads(raw)
        if not isinstance(notes, list):
            raise ValueError("Expected array")
        return notes
    except Exception as e:
        log.error(f"call_gemini parse error: {e}\nRaw: {raw[:500]}")
        raise HTTPException(status_code=500, detail=f"Gemini response parse error: {e}")
