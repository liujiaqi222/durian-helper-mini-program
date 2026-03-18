# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is the "AI Durian Picker" backend. Currently only the **cv-service** (Python FastAPI + YOLO) is implemented. The planned NestJS main backend (`server/`) has design docs only — no runnable code.

### cv-service (Python FastAPI)

- **Location:** `cv-service/`
- **Run dev server:** `cd cv-service && source .venv/bin/activate && uvicorn app.main:app --reload --port 8010`
- **Lint:** `cd cv-service && source .venv/bin/activate && ruff check app/ && ruff format --check app/`
- **Endpoints:** `GET /health`, `POST /detect` (multipart file upload)
- **Swagger UI:** `http://127.0.0.1:8010/docs`

### Model weights

The config (`app/config.py`) expects `models/durian-best.pt`. The repo ships `models/best.pt`. The update script creates a symlink `durian-best.pt -> best.pt` if the expected file is missing. If you retrain the model, place the new `best.pt` and re-run the symlink step or rename directly.

### Gotchas

- `python3.12-venv` apt package is required to create virtualenvs on this VM — the update script handles this.
- The YOLO model loads during FastAPI lifespan startup. If the model file is missing, the server will crash with a `RuntimeError` on start.
- `ultralytics` pulls in PyTorch, which takes ~90s to install on first run; subsequent `pip install -r requirements.txt` is fast.
