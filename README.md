## Setup and run
 
### Option A: with uv (recommended)
 
```bash
git clone <repo-url>
cd <project-folder>
 
uv sync                  # installs dependencies from pyproject.toml / uv.lock
uv run fastapi dev       # starts the server with auto-reload
```
 
### Option B: with pip
 
```bash
git clone <repo-url>
cd <project-folder>
 
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
                                   # Windows Git Bash:   source .venv/Scripts/activate
 
pip install -e . -r requirements-dev.txt
uvicorn task_processing_system.main:app --reload
```
 
The server starts at **http://127.0.0.1:8000**.
 
- Interactive API docs (Swagger UI): **http://127.0.0.1:8000/docs**
- Health check: http://127.0.0.1:8000/health
The SQLite file `tasks.db` is created automatically in the folder the server is started from.
 
---
 
## Run the tests
 
```bash
uv run pytest -v          # or simply: pytest -v   (with the venv activated)
```
