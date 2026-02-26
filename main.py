"""
ThoughtSort API — FastAPI backend
Endpoints:
  POST /inbox/append   — add a raw thought to the user's inbox
  POST /sort           — trigger AI sort for the user
  GET  /notes          — fetch all sorted notes
  GET  /notes/{id}     — fetch a single note
  POST /settings       — save user settings (known tags etc)
  GET  /settings       — get user settings
  GET  /health         — health check
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
import httpx

# ── LOGGING ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("thoughtsort")

# ── FIREBASE INIT ─────────────────────────────────────────────
# On Railway, FIREBASE_CREDENTIALS env var holds the JSON as a string
_cred_json = os.environ.get("FIREBASE_CREDENTIALS")
if _cred_json:
    _cred_dict = json.loads(_cred_json)
    cred = credentials.Certificate(_cred_dict)
else:
    # Local dev: point to the downloaded JSON file
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
    """Verify Firebase Auth token and return the user's UID."""
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
    timestamp: str  # ISO format: "2026-02-25T15:58:00"

class SettingsRequest(BaseModel):
    known_tags: list[str] = []

# ── HEALTH CHECK ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "ThoughtSort API"}

# ── APPEND TO INBOX ───────────────────────────────────────────
@app.post("/inbox/append")
async def append_to_inbox(body: AppendRequest, uid: str = Depends(get_user_id)):
    """Add a new raw thought to the user's Firestore inbox."""
    entry = {
        "text":      body.text,
        "timestamp": body.timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    db.collection("users").document(uid).collection("inbox").add(entry)
    log.info(f"Appended entry for user {uid}")
    return {"status": "ok"}

# ── GET NOTES ─────────────────────────────────────────────────
@app.get("/notes")
async def get_notes(uid: str = Depends(get_user_id)):
    """Return all sorted notes for the user, newest first."""
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
    """Return a single note by ID."""
    doc = db.collection("users").document(uid).collection("notes").document(note_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Note not found")
    d = doc.to_dict()
    d["id"] = doc.id
    return d

# ── SETTINGS ──────────────────────────────────────────────────
@app.post("/settings")
async def save_settings(body: SettingsRequest, uid: str = Depends(get_user_id)):
    """Save user settings (known tags etc)."""
    db.collection("users").document(uid).set(
        {"known_tags": body.known_tags},
        merge=True
    )
    return {"status": "ok"}

@app.get("/settings")
async def get_settings(uid: str = Depends(get_user_id)):
    """Get user settings."""
    doc = db.collection("users").document(uid).get()
    if not doc.exists:
        return {"known_tags": []}
    return doc.to_dict()

# ── SORT ──────────────────────────────────────────────────────
@app.post("/sort")
async def sort_notes(uid: str = Depends(get_user_id)):
    """
    Trigger AI sort for the user:
    1. Read inbox from Firestore
    2. Archive inbox entries
    3. Call Gemini
    4. Parse response into individual notes
    5. Save notes to Firestore
    6. Clear inbox
    """
    # 1. Read inbox
    inbox_ref  = db.collection("users").document(uid).collection("inbox")
    inbox_docs = list(inbox_ref.stream())

    if not inbox_docs:
        return {"status": "ok", "message": "Inbox is empty — nothing to sort.", "count": 0}

    entries = []
    for doc in inbox_docs:
        d = doc.to_dict()
        entries.append({"id": doc.id, "text": d.get("text",""), "timestamp": d.get("timestamp","")})

    log.info(f"Sorting {len(entries)} entries for user {uid}")

    # 2. Archive inbox
    archive_ref  = db.collection("users").document(uid).collection("inbox_archive")
    sort_run_id  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    archive_ref.add({
        "sort_run":  sort_run_id,
        "entries":   entries,
        "archived_at": datetime.now(timezone.utc).isoformat(),
    })

    # 3. Get user's known tags
    user_doc   = db.collection("users").document(uid).get()
    known_tags = []
    if user_doc.exists:
        known_tags = user_doc.to_dict().get("known_tags", [])

    # 4. Call Gemini
    inbox_text = "\n\n".join(
        f"### {e['timestamp']}\n{e['text']}" for e in entries
    )
    sorted_notes = await call_gemini(inbox_text, known_tags)

    # 5. Save notes to Firestore
    notes_ref = db.collection("users").document(uid).collection("notes")
    saved = 0
    for note in sorted_notes:
        note["sort_run_id"]  = sort_run_id
        note["created_at"]   = datetime.now(timezone.utc).isoformat()
        notes_ref.add(note)
        saved += 1

    # 6. Clear inbox
    batch = db.batch()
    for doc in inbox_docs:
        batch.delete(inbox_ref.document(doc.id))
    batch.commit()

    log.info(f"Sort complete for {uid}: {saved} notes saved")
    return {"status": "ok", "message": f"Sorted {saved} entries successfully.", "count": saved}

# ── GEMINI CALL ───────────────────────────────────────────────
async def call_gemini(inbox_text: str, known_tags: list[str]) -> list[dict]:
    """Call Gemini and parse response into a list of note dicts."""

    known_tags_str = ""
    if known_tags:
        tag_list = ", ".join(f"#{t}" for t in known_tags)
        known_tags_str = f"""
KNOWN TAGS (prefer these — only create new ones if nothing fits):
{tag_list}
"""

    prompt = f"""You are an intelligent personal thought organizer. Read the user's inbox and organize each entry into a structured note.

{known_tags_str}

RULES:
1. Assign 1-4 tags per entry. Use known tags when possible. Create new tags (lowercase-hyphenated) only when nothing fits.
2. Preserve the original text of each entry EXACTLY — never rewrite or paraphrase.
3. Write a brief AI note (1-2 sentences) adding context or connecting to related themes.
4. Generate a descriptive filename: 3-6 words, lowercase, hyphens, based on the core idea. No dates, no tag names.
5. Tags should NOT have # prefix in the JSON output — just the word itself.

OUTPUT: Return ONLY a valid JSON array. No markdown, no code blocks, no explanation. Each object must have exactly these fields:
- originalText: string (verbatim from inbox)
- timestamp: string (from the ### header)
- filename: string (e.g. "dark-mode-website-idea")
- tags: array of strings (e.g. ["website", "design"])
- aiNote: string

INBOX TO SORT:

{inbox_text}"""

    model    = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=8192)
    )

    raw = response.text.strip()

    # Strip markdown code fences if Gemini wraps in them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        notes = json.loads(raw)
        if not isinstance(notes, list):
            raise ValueError("Expected a JSON array")
        return notes
    except Exception as e:
        log.error(f"Failed to parse Gemini response: {e}\nRaw: {raw[:500]}")
        raise HTTPException(status_code=500, detail=f"Gemini response parse error: {e}")
