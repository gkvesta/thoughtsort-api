# ThoughtSort API

FastAPI backend for ThoughtSort — AI-powered thought capture and organization.

## Local Development

1. Place your Firebase service account JSON as `serviceAccountKey.json` in this folder
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   uvicorn main:app --reload
   ```
4. API docs at: http://localhost:8000/docs

## Railway Deployment

Set these environment variables in Railway:

| Variable | Value |
|----------|-------|
| `FIREBASE_CREDENTIALS` | Contents of your serviceAccountKey.json (the whole JSON as a string) |
| `GEMINI_API_KEY` | Your Gemini API key from aistudio.google.com |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /inbox/append | Add thought to inbox |
| POST | /sort | Trigger AI sort |
| GET | /notes | Get all sorted notes |
| GET | /notes/{id} | Get single note |
| GET | /settings | Get user settings |
| POST | /settings | Save user settings |

All endpoints except /health require Firebase Auth Bearer token.

## Data Model (Firestore)

```
users/{uid}/
  inbox/          — raw captured thoughts (cleared after sort)
  inbox_archive/  — permanent log of all past inbox entries
  notes/          — individual AI-sorted notes
  (document)      — user settings (known_tags etc)
```
