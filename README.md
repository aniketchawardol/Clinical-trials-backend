# Clinical Trials Backend

FastAPI server powering the clinical trial chatbot and interactive map. It exposes a REST API consumed by the frontend React app.

---

## Architecture

The backend has two main responsibilities:

1. **Data layer** — loads `mock_trials.csv` into a hybrid BM25 + ChromaDB index for retrieval-augmented generation (RAG).
2. **Agent layer** — a LangGraph pipeline that processes chat messages: extracts intent, applies guardrails, geocodes the location, retrieves matching trials, and generates a conversational reply via Google Gemini.

### LangGraph Pipeline

```
User message
    │
    ▼
extract_intent  ──► (Gemini 2.5 Flash JSON → falls back to regex)
    │
    ▼
guardrails  ──► off-topic / medical-advice → immediate reply (short-circuit)
    │
    ▼
geocode  ──► MapTiler API → (lat, lon)
    │
    ├── no location extracted ──────────────────────┐
    ▼                                               │
rag_retrieve  ──► hybrid BM25 + ChromaDB            │
    │                                               │
    ▼                                               │
filter  ──► geodesic distance ≤ 500 km              │
    │                                               │
    ▼  ◄────────────────────────────────────────────┘
build_reply  ──► Gemini summary → fallback template reply
    │
    ▼
ChatResponse { reply, geojson }
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/trials` | GeoJSON FeatureCollection of all trials; filter by `location` and `cancer_type` query params |
| `GET` | `/api/meta` | Distinct values for UI dropdowns: `statuses`, `provinces`, `cancer_types` |
| `POST` | `/api/chat` | Run the LangGraph agent; returns `{ reply, geojson }` |
| `GET` | `/health` | Liveness check; reports BM25 and ChromaDB index status |

### `POST /api/chat`

**Request body:**
```json
{ "message": "Find breast cancer trials near Toronto" }
```

**Response:**
```json
{
  "reply": "Found **4** trial(s) near **Toronto** for **Breast Cancer**:...",
  "geojson": {
    "type": "FeatureCollection",
    "user_center": [-79.38, 43.65],
    "features": [...]
  }
}
```

### `GET /api/trials`

Query params (both optional):

| Param | Example | Notes |
|-------|---------|-------|
| `location` | `Vancouver` | Geocoded via MapTiler; returns trials within 500 km |
| `cancer_type` | `Lung Cancer` | Filtered via hybrid RAG (supports synonyms: "NSCLC", "blood cancer", etc.) |

---

## Project Structure

```
backend/
├── bin/
│   └── api.py              # FastAPI app + LangGraph agent (single entry point)
├── data/
│   ├── mock_trials.csv     # Trial data (trial_id, facility, lat/lon, status, etc.)
│   ├── load_mock_csv.py    # One-time pipeline: CSV → BM25 index + ChromaDB + GeoJSON
│   ├── bm25_index.pkl      # Serialised BM25Okapi index (generated)
│   └── trials.geojson      # GeoJSON snapshot (generated)
├── chroma_db/              # ChromaDB persistent store (generated)
├── .env                    # API keys (not committed)
├── pyproject.toml          # Poetry dependencies
└── requirements.txt        # Pinned pip-installable dependencies
```

---

## Setup

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/) (recommended) or pip

### 1. Install dependencies

**With Poetry:**
```powershell
cd backend
poetry install
```

**With pip:**
```powershell
cd backend
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the template and fill in your keys:
```powershell
copy .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes (for LLM) | Google Gemini API key — get one at [aistudio.google.com](https://aistudio.google.com) |
| `MAPTILER_API_KEY` | Yes (for geocoding) | MapTiler API key — get one at [maptiler.com](https://maptiler.com) |

> **Offline fallback:** If `GOOGLE_API_KEY` is not set or quota is exhausted, intent extraction falls back to regex patterns and replies fall back to a template. If `MAPTILER_API_KEY` is not set, location-based filtering is unavailable.

### 3. Build the search indexes

Run once (or any time `mock_trials.csv` changes):
```powershell
poetry run python -m data.load_mock_csv
```

This writes:
- `data/bm25_index.pkl` — keyword search index
- `data/trials.geojson` — static GeoJSON snapshot
- `chroma_db/` — ChromaDB vector embeddings

### 4. Start the server

```powershell
poetry run uvicorn bin.api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

---

## Data Format

`mock_trials.csv` must have the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `trial_id` | string | Unique identifier |
| `cancer_type` | string | e.g. `Breast Cancer`, `Lung Cancer` |
| `facility` | string | Hospital / clinic name |
| `address` | string | Street address |
| `city` | string | City name |
| `province` | string | Two-letter province code |
| `latitude` | float | Decimal degrees |
| `longitude` | float | Decimal degrees |
| `status` | string | e.g. `RECRUITING`, `COMPLETED` |
| `principal_investigator` | string | PI name |
| `url` | string | Link to trial details |

---

## RAG: Hybrid Retrieval

The search pipeline combines two strategies and fuses their scores:

- **BM25** (`rank-bm25`) — keyword overlap; fast, works offline
- **ChromaDB** — semantic vector search using the default embedding function

Supported cancer type synonyms (resolved by Gemini or regex):

| User says | Normalised to |
|-----------|---------------|
| `NSCLC`, `SCLC`, `lung` | Lung Cancer |
| `blood cancer`, `leukemia` | Leukemia |
| `bowel`, `colorectal`, `colon` | Colorectal Cancer |
| `GBM`, `glioma`, `brain` | Brain Cancer |
| …and more | see `_CANCER_KEYWORDS` in `api.py` |

---

## Development Notes

- The Gemini client is instantiated once at startup and **auto-disabled for the process lifetime** on a `429 quota exhausted` error, switching transparently to regex/template fallback.
- `MAX_RADIUS_KM = 500` — all distance filtering uses geodesic distance via `geopy`.
- CORS is open (`allow_origins=["*"]`) for local development; tighten for production.
