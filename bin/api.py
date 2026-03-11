"""
FastAPI server — Clinical Trials backend.

Endpoints
---------
GET  /api/trials   → GeoJSON FeatureCollection (optionally filtered by location/cancer_type)
GET  /api/meta     → distinct filter values for UI dropdowns
POST /api/chat     → LangGraph agent: intent → guardrails → geocode → RAG → reply + GeoJSON
GET  /health       → liveness check
"""
import json
import math
import os
import pickle
from typing import Any, Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from geopy.distance import geodesic
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from typing_extensions import TypedDict

import google.genai as genai
from google.genai import errors as genai_errors

# ---------------------------------------------------------------------------
# Gemini client (created once; disabled automatically on quota exhaustion)
# ---------------------------------------------------------------------------

_gemini_client: Optional[genai.Client] = None
_gemini_disabled = False  # set to True on 429 to stop retrying for this process lifetime


def _get_gemini_client() -> Optional[genai.Client]:
    global _gemini_client, _gemini_disabled
    if _gemini_disabled:
        return None
    if _gemini_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

# ---------------------------------------------------------------------------
# Config / startup
# ---------------------------------------------------------------------------

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_FILE = os.path.join(DATA_DIR, "mock_trials.csv")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "trials"
MAX_RADIUS_KM = 500.0

# Load CSV into memory
df_trials = pd.read_csv(DATA_FILE)
df_trials = df_trials.dropna(subset=["latitude", "longitude"])

# ---------------------------------------------------------------------------
# RAG indexes (loaded once at startup; fall back gracefully if not built yet)
# ---------------------------------------------------------------------------

_bm25_index: Optional[BM25Okapi] = None
_bm25_ids: list[str] = []
_chroma_collection = None


def _load_indexes() -> None:
    global _bm25_index, _bm25_ids, _chroma_collection

    # BM25
    if os.path.exists(BM25_PATH):
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        _bm25_index = data["index"]
        _bm25_ids = data["ids"]
        print(f"[bm25]     loaded {len(_bm25_ids)} docs")
    else:
        print("[bm25]     index not found — run 'python -m data.load_mock_csv' first")

    # ChromaDB
    if os.path.exists(CHROMA_DIR):
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            ef = DefaultEmbeddingFunction()
            _chroma_collection = client.get_collection(
                name=COLLECTION_NAME, embedding_function=ef
            )
            print(f"[chromadb] loaded collection '{COLLECTION_NAME}'")
        except Exception as e:
            print(f"[chromadb] could not load collection: {e}")
    else:
        print("[chromadb] chroma_db directory not found — run load_mock_csv first")


_load_indexes()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Clinical Trials API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers: geocoding, filtering, GeoJSON
# ---------------------------------------------------------------------------

STATUS_LABELS: dict[str, str] = {
    "RECRUITING": "Recruiting",
    "NOT_YET_RECRUITING": "Not Yet Recruiting",
    "ACTIVE_NOT_RECRUITING": "Active, Not Recruiting",
    "ACTIVE, NOT RECRUITING": "Active, Not Recruiting",
    "COMPLETED": "Completed",
    "TERMINATED": "Terminated",
    "SUSPENDED": "Suspended",
    "WITHDRAWN": "Withdrawn",
    "ENROLLING_BY_INVITATION": "Enrolling by Invitation",
    "ENROLLING BY INVITATION": "Enrolling by Invitation",
    "NOT YET RECRUITING": "Not Yet Recruiting",
}


def normalize_status(status: Any) -> str:
    if not status or not isinstance(status, str):
        return "Unknown"
    return STATUS_LABELS.get(status.strip().upper(), status.strip())


def _clean(value: Any) -> Any:
    """Convert NaN to None so JSONResponse doesn't emit invalid JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def get_coordinates(location_str: str) -> Optional[tuple[float, float]]:
    """MapTiler geocoding → (lat, lon). Returns None on failure."""
    token = os.environ.get("MAPTILER_API_KEY")
    if not token:
        print("MAPTILER_API_KEY not set")
        return None
    url = f"https://api.maptiler.com/geocoding/{requests.utils.quote(location_str)}.json"
    try:
        r = requests.get(url, params={"key": token, "country": "ca", "limit": 1}, timeout=5)
        r.raise_for_status()
        features = r.json().get("features", [])
        if features:
            lon, lat = features[0]["center"]
            return (lat, lon)
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None


def rag_search(query: str, n_results: int = 500) -> list[str]:
    """
    Hybrid BM25 + ChromaDB retrieval.
    Returns a ranked list of trial_ids (best matches first).
    Falls back to an empty list if indexes are not available.
    """
    if not query:
        return []

    scores: dict[str, float] = {}

    # BM25 — keyword overlap
    if _bm25_index is not None:
        tokens = query.lower().split()
        bm25_scores = _bm25_index.get_scores(tokens)
        for idx, score in enumerate(bm25_scores):
            if score > 0:
                tid = _bm25_ids[idx]
                scores[tid] = scores.get(tid, 0.0) + score

    # ChromaDB — semantic similarity
    if _chroma_collection is not None:
        try:
            results = _chroma_collection.query(
                query_texts=[query], n_results=min(n_results, 500)
            )
            for tid, dist in zip(results["ids"][0], results["distances"][0]):
                # Convert distance to a positive similarity score
                scores[tid] = scores.get(tid, 0.0) + 1.0 / (1.0 + dist)
        except Exception as e:
            print(f"ChromaDB query error: {e}")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in ranked[:n_results]]


def filter_trials(
    lat: float,
    lon: float,
    cancer_type: Optional[str] = None,
    radius_km: float = MAX_RADIUS_KM,
) -> list[dict]:
    """
    Distance filter combined with optional RAG-based cancer type matching.
    When cancer_type is given, hybrid BM25+ChromaDB retrieval is used so that
    synonyms ('blood cancer' → Leukemia, 'NSCLC' → Lung Cancer) are handled.
    """
    rag_id_set: Optional[set[str]] = None
    if cancer_type:
        ranked = rag_search(cancer_type)
        if ranked:
            rag_id_set = set(ranked)

    results = []
    for _, row in df_trials.iterrows():
        tid = str(row["trial_id"])
        if rag_id_set is not None and tid not in rag_id_set:
            continue
        dist = geodesic((lat, lon), (row["latitude"], row["longitude"])).kilometers
        if dist <= radius_km:
            d = row.to_dict()
            d["distance_km"] = round(dist, 1)
            results.append(d)

    results.sort(key=lambda x: x["distance_km"])
    return results


def build_geojson(
    trials: list[dict], user_center: Optional[tuple[float, float]] = None
) -> dict:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [t["longitude"], t["latitude"]]},
            "properties": {
                "trial_id": _clean(t.get("trial_id")),
                "cancer_type": _clean(t.get("cancer_type")),
                "status": normalize_status(_clean(t.get("status"))),
                "facility": _clean(t.get("facility")),
                "city": _clean(t.get("city")),
                "province": _clean(t.get("province")),
                "principal_investigator": _clean(t.get("principal_investigator")),
                "url": _clean(t.get("url")),
                "distance_km": _clean(t.get("distance_km")),
            },
        }
        for t in trials
    ]
    geojson: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if user_center:
        geojson["user_center"] = [user_center[1], user_center[0]]  # [lon, lat] for Mapbox
    return geojson


import re

# ---------------------------------------------------------------------------
# LangGraph agent
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Shared state passed through every node in the graph."""
    message: str
    # Intent
    location: Optional[str]
    cancer_type: Optional[str]
    is_medical_advice: bool
    is_off_topic: bool
    # Geocoding
    coords: Optional[tuple[float, float]]
    geocode_failed: bool
    # RAG + filtering
    rag_ids: list[str]
    trials: list[dict]
    # Output
    reply: str
    geojson: Optional[dict]


_INTENT_PROMPT = """\
You are a routing assistant for a Canadian cancer clinical trial finder.
Extract structured intent from the user message and reply in JSON only.

Keys:
  location          – city, address, or postal code mentioned (string or null)
  cancer_type       – specific cancer type mentioned (string or null); normalise to common name
  is_medical_advice – true if the user asks for diagnosis, prognosis, or treatment advice
  is_off_topic      – true if the query has nothing to do with clinical trials or cancer

Examples:
  "Find breast cancer trials near Toronto"
    → {"location":"Toronto","cancer_type":"Breast Cancer","is_medical_advice":false,"is_off_topic":false}
  "blood cancer clinics in Vancouver"
    → {"location":"Vancouver","cancer_type":"Leukemia","is_medical_advice":false,"is_off_topic":false}
  "NSCLC trials in Calgary"
    → {"location":"Calgary","cancer_type":"Lung Cancer","is_medical_advice":false,"is_off_topic":false}
  "Am I in remission?"
    → {"location":null,"cancer_type":null,"is_medical_advice":true,"is_off_topic":false}
  "What is the capital of France?"
    → {"location":null,"cancer_type":null,"is_medical_advice":false,"is_off_topic":true}
"""


_CANCER_KEYWORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(breast)\b", re.I), "Breast Cancer"),
    (re.compile(r"\b(lung|nsclc|sclc)\b", re.I), "Lung Cancer"),
    (re.compile(r"\b(colorect|colon|rectal|bowel)\b", re.I), "Colorectal Cancer"),
    (re.compile(r"\b(prostate)\b", re.I), "Prostate Cancer"),
    (re.compile(r"\b(melanoma|skin)\b", re.I), "Melanoma"),
    (re.compile(r"\b(leukemi|leukaemi|blood cancer)\b", re.I), "Leukemia"),
    (re.compile(r"\b(lymphoma)\b", re.I), "Lymphoma"),
    (re.compile(r"\b(brain|glioma|glioblastoma|gbm)\b", re.I), "Brain Cancer"),
    (re.compile(r"\b(pancrea)\b", re.I), "Pancreatic Cancer"),
    (re.compile(r"\b(ovarian|ovary)\b", re.I), "Ovarian Cancer"),
    (re.compile(r"\b(bladder)\b", re.I), "Bladder Cancer"),
    (re.compile(r"\b(kidney|renal)\b", re.I), "Kidney Cancer"),
    (re.compile(r"\b(thyroid)\b", re.I), "Thyroid Cancer"),
    (re.compile(r"\b(liver|hepat)\b", re.I), "Liver Cancer"),
    (re.compile(r"\b(stomach|gastric)\b", re.I), "Gastric Cancer"),
    (re.compile(r"\b(cervic)\b", re.I), "Cervical Cancer"),
    (re.compile(r"\b(uterine|endometri)\b", re.I), "Uterine Cancer"),
    (re.compile(r"\b(myeloma|multiple myeloma)\b", re.I), "Multiple Myeloma"),
]

_LOCATION_PATTERN = re.compile(
    r"\b(?:near|in|around|close to|at)\s+([A-Za-z][A-Za-z\s\-'\.]{1,40}?)(?:\s*$|[,.])",
    re.I,
)

_MEDICAL_ADVICE_PATTERN = re.compile(
    r"\b(am i|do i have|is this|what stage|my diagnosis|my prognosis|should i take|treatment for me)\b",
    re.I,
)


def _regex_extract(message: str) -> dict:
    """
    Fast offline fallback when Gemini is unavailable.
    Extracts location, cancer_type, is_medical_advice from the message using patterns.
    """
    cancer_type = None
    for pattern, name in _CANCER_KEYWORDS:
        if pattern.search(message):
            cancer_type = name
            break

    location = None
    m = _LOCATION_PATTERN.search(message)
    if m:
        location = m.group(1).strip()

    is_medical_advice = bool(_MEDICAL_ADVICE_PATTERN.search(message))

    return {
        "location": location,
        "cancer_type": cancer_type,
        "is_medical_advice": is_medical_advice,
        "is_off_topic": False,
    }


def node_extract_intent(state: AgentState) -> AgentState:
    global _gemini_disabled
    parsed = {}
    client = _get_gemini_client()
    if client is not None:
        prompt = _INTENT_PROMPT + f'\n\nUser: "{state["message"]}"'
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            print(f"[intent] gemini: {resp.text!r}")
            parsed = json.loads(resp.text)
        except genai_errors.ClientError as e:
            if e.code == 429:
                print("[intent] quota exhausted — disabling Gemini for this session, using regex fallback")
                _gemini_disabled = True
            else:
                print(f"[intent] Gemini error {e.code}: {e}")
        except Exception as e:
            print(f"[intent] unexpected error: {type(e).__name__}: {e}")

    # Fall back to regex if Gemini gave nothing useful or is unavailable
    if not parsed.get("location") and not parsed.get("cancer_type") and not parsed.get("is_medical_advice") and not parsed.get("is_off_topic"):
        fallback = _regex_extract(state["message"])
        print(f"[intent] regex fallback: {fallback}")
        parsed = fallback

    return {
        **state,
        "location": parsed.get("location"),
        "cancer_type": parsed.get("cancer_type"),
        "is_medical_advice": bool(parsed.get("is_medical_advice", False)),
        "is_off_topic": bool(parsed.get("is_off_topic", False)),
    }


def node_guardrails(state: AgentState) -> AgentState:
    if state["is_medical_advice"]:
        return {
            **state,
            "reply": (
                "I can't provide medical advice, diagnoses, or treatment recommendations. "
                "Please consult your oncologist or healthcare provider. "
                "I can only help you find clinical trials by location and cancer type."
            ),
            "geojson": None,
        }
    if state["is_off_topic"]:
        return {
            **state,
            "reply": (
                "I'm specialised in finding clinical cancer trials in Canada. "
                'Try asking: _"Find lung cancer trials near Vancouver"_.'
            ),
            "geojson": None,
        }
    return state


def node_geocode(state: AgentState) -> AgentState:
    location = state.get("location")
    if not location:
        return {**state, "coords": None, "geocode_failed": False}
    coords = get_coordinates(location)
    return {**state, "coords": coords, "geocode_failed": coords is None}


def node_rag_retrieve(state: AgentState) -> AgentState:
    cancer_type = state.get("cancer_type")
    return {**state, "rag_ids": rag_search(cancer_type) if cancer_type else []}


def node_filter(state: AgentState) -> AgentState:
    coords = state.get("coords")
    if not coords:
        return {**state, "trials": []}

    rag_id_set = set(state["rag_ids"]) if state.get("rag_ids") else None
    results = []
    for _, row in df_trials.iterrows():
        tid = str(row["trial_id"])
        if rag_id_set is not None and tid not in rag_id_set:
            continue
        dist = geodesic(coords, (row["latitude"], row["longitude"])).kilometers
        if dist <= MAX_RADIUS_KM:
            d = row.to_dict()
            d["distance_km"] = round(dist, 1)
            results.append(d)

    results.sort(key=lambda x: x["distance_km"])
    return {**state, "trials": results}


_REPLY_PROMPT = """\
You are a helpful, empathetic clinical trial finder assistant for cancer patients in Canada.
The user asked a question, and we retrieved some matching clinical trials based on their location and condition.
Your job is to write a conversational, encouraging response summarizing the best matching trials.

Rules:
1. Be concise but empathetic.
2. Mention the number of trials found near their location.
3. Highlight 2-3 of the most relevant trials by name, status, and distance.
4. Use Markdown formatting (bolding facility names).
5. IMPORTANT: ALWAYS include a disclaimer at the end stating: "This information is for reference only. Please consult your physician or oncologist before making any medical decisions."
6. Do NOT invent or hallucinate trials. ONLY use the context provided.
"""

def node_build_reply(state: AgentState) -> AgentState:
    global _gemini_disabled
    location = state.get("location")
    cancer_type = state.get("cancer_type")
    trials = state.get("trials", [])

    if not location:
        if cancer_type:
            reply = (
                f"I can see you're looking for **{cancer_type}** trials. "
                "Could you share your city or postal code so I can find ones near you?"
            )
        else:
            reply = (
                "Could you tell me your city and what type of cancer trials you're looking for? "
                'For example: _"Find breast cancer trials near Toronto"_.'
            )
        return {**state, "reply": reply, "geojson": None}

    if state.get("geocode_failed"):
        return {
            **state,
            "reply": (
                f"I couldn't locate **{location}**. "
                "Please try a more specific Canadian city name or postal code."
            ),
            "geojson": None,
        }

    geojson = build_geojson(trials, state["coords"])

    if not trials:
        reply = f"No trials found near **{location}**"
        reply += f" for **{cancer_type}**." if cancer_type else "."
        reply += "\n\nTry broadening your search — I already looked within 500 km. Please consult your physician for more options."
        return {**state, "reply": reply, "geojson": geojson}

    # If we have trials and LLM is available, generate a response
    client = _get_gemini_client()
    if client is not None:
        context_lines = []
        for t in trials[:5]:  # Send top 5 to context
            status = normalize_status(str(t.get("status", "")))
            dist = t.get("distance_km", "Unknown")
            context_lines.append(f"- Trial ID: {t.get('trial_id')} | Facility: {t.get('facility')} | Cancer: {t.get('cancer_type')} | Status: {status} | Distance: {dist} km")
        
        context_str = "\n".join(context_lines)
        
        prompt = (
            f"{_REPLY_PROMPT}\n\n"
            f"User Location: {location}\n"
            f"Cancer Type: {cancer_type or 'Any'}\n"
            f"Total Trials Found: {len(trials)}\n"
            f"Top Matches Context:\n{context_str}\n\n"
            f"User Message: \"{state['message']}\"\n\n"
            "Response:"
        )
        
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            print("[build_reply] gemini generated response successfully.")
            return {**state, "reply": resp.text, "geojson": geojson}
        except genai_errors.ClientError as e:
            if e.code == 429:
                print("[build_reply] quota exhausted — disabling Gemini for this session, using fallback")
                _gemini_disabled = True
            else:
                print(f"[build_reply] Gemini error {e.code}: {e}")
        except Exception as e:
            print(f"[build_reply] unexpected error: {type(e).__name__}: {e}")

    # Fallback to the old hardcoded reply if Gemini fails or is disabled
    reply = f"Found **{len(trials)}** trial(s) near **{location}**"
    if cancer_type:
        reply += f" for **{cancer_type}**"
    reply += ":\n"
    for t in trials[:3]:
        status = normalize_status(str(t.get("status", "")))
        reply += f"\n• **{t['facility']}** — {status} ({t['distance_km']} km)"
    if len(trials) > 3:
        reply += f"\n\n_…and {len(trials) - 3} more. Click map pins for details._"
    reply += "\n\n*Disclaimer: This information is for reference only. Please consult your physician or oncologist before making any medical decisions.*"

    return {**state, "reply": reply, "geojson": geojson}


def _route_after_guardrails(state: AgentState) -> str:
    """Short-circuit to END when a guardrail already set a reply."""
    return END if state.get("reply") else "geocode"


def _route_after_geocode(state: AgentState) -> str:
    """Skip RAG+filter when no location was extracted."""
    return "build_reply" if not state.get("location") else "rag_retrieve"


_builder = StateGraph(AgentState)
_builder.add_node("extract_intent", node_extract_intent)
_builder.add_node("guardrails", node_guardrails)
_builder.add_node("geocode", node_geocode)
_builder.add_node("rag_retrieve", node_rag_retrieve)
_builder.add_node("filter", node_filter)
_builder.add_node("build_reply", node_build_reply)

_builder.set_entry_point("extract_intent")
_builder.add_edge("extract_intent", "guardrails")
_builder.add_conditional_edges("guardrails", _route_after_guardrails)
_builder.add_conditional_edges("geocode", _route_after_geocode)
_builder.add_edge("rag_retrieve", "filter")
_builder.add_edge("filter", "build_reply")
_builder.add_edge("build_reply", END)

agent = _builder.compile()


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/trials")
def get_trials(location: Optional[str] = None, cancer_type: Optional[str] = None):
    """
    Return GeoJSON FeatureCollection.
    With location: filter by proximity (500 km) + optional RAG cancer type match.
    Without location: return all trials (distance_km = null).
    """
    if location:
        coords = get_coordinates(location)
        if coords:
            trials = filter_trials(coords[0], coords[1], cancer_type)
            return JSONResponse(content=build_geojson(trials, coords))

    all_trials = df_trials.to_dict(orient="records")
    for t in all_trials:
        t["distance_km"] = None
    return JSONResponse(content=build_geojson(all_trials))


@app.get("/api/meta")
def get_meta():
    """Return distinct values for UI filter dropdowns."""
    statuses = sorted({normalize_status(str(s)) for s in df_trials["status"].dropna().unique()})
    provinces = sorted(
        {str(p).strip() for p in df_trials["province"].dropna().unique() if str(p).strip()}
    )
    cancer_set: set[str] = set()
    for val in df_trials["cancer_type"].dropna().unique():
        for part in str(val).split("|"):
            t = part.strip()
            if t:
                cancer_set.add(t)
    return JSONResponse(content={
        "statuses": statuses,
        "provinces": provinces,
        "cancer_types": sorted(cancer_set),
    })


@app.get("/health")
def health():
    return {
        "status": "ok",
        "bm25": _bm25_index is not None,
        "chromadb": _chroma_collection is not None,
    }


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    geojson: Optional[dict] = None


@app.post("/api/chat")
def chat(req: ChatRequest) -> ChatResponse:
    """
    Run the LangGraph agent pipeline:
      extract_intent → guardrails → geocode → rag_retrieve → filter → build_reply
    """
    initial_state: AgentState = {
        "message": req.message,
        "location": None,
        "cancer_type": None,
        "is_medical_advice": False,
        "is_off_topic": False,
        "coords": None,
        "geocode_failed": False,
        "rag_ids": [],
        "trials": [],
        "reply": "",
        "geojson": None,
    }
    try:
        final_state = agent.invoke(initial_state)
    except Exception as e:
        print(f"Agent error: {e}")
        return ChatResponse(reply="Sorry, something went wrong. Please try again.")

    return ChatResponse(reply=final_state["reply"], geojson=final_state.get("geojson"))
