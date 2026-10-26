# with uv
uv sync
uv run fastapi dev
uv run pytest -v

# with pip
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e . -r requirements-dev.txt
fastapi dev
pytest -v