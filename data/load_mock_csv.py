"""
Data pipeline: CSV → ChromaDB embeddings + BM25 index + GeoJSON snapshot.

Run once (or whenever mock_trials.csv changes):
    poetry run python -m data.load_mock_csv
"""
import os
import json
import pickle

import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(DATA_DIR, "mock_trials.csv")
GEOJSON_PATH = os.path.join(DATA_DIR, "trials.geojson")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
CHROMA_DIR = os.path.join(DATA_DIR, "..", "chroma_db")
COLLECTION_NAME = "trials"


def _row_to_document(row: pd.Series) -> str:
    """Build a plain-text document from a trial row for indexing."""
    parts = [
        str(row.get("trial_id", "")),
        str(row.get("cancer_type", "")),
        str(row.get("facility", "")),
        str(row.get("city", "")),
        str(row.get("province", "")),
        str(row.get("status", "")),
        str(row.get("principal_investigator", "")),
    ]
    return " | ".join(p for p in parts if p and p != "nan")


def build_geojson(df: pd.DataFrame) -> None:
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]],
            },
            "properties": {
                "trial_id": row["trial_id"],
                "cancer_type": row["cancer_type"],
                "status": row["status"],
                "facility": row["facility"],
                "address": row["address"],
                "city": row["city"],
                "province": row["province"],
                "principal_investigator": row["principal_investigator"],
                "url": row["url"],
            },
        })
    geojson = {"type": "FeatureCollection", "features": features}
    with open(GEOJSON_PATH, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    print(f"[geojson]  wrote {len(features)} features → {GEOJSON_PATH}")


def build_chromadb(df: pd.DataFrame) -> None:
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop and recreate so re-runs are idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    ef = DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    # Deduplicate by trial_id — keep first occurrence
    seen_ids: set[str] = set()
    documents, ids, metadatas = [], [], []
    for _, row in df.iterrows():
        tid = str(row["trial_id"])
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        doc = _row_to_document(row)
        documents.append(doc)
        ids.append(tid)
        metadatas.append({
            "trial_id": tid,
            "cancer_type": str(row.get("cancer_type", "")),
            "city": str(row.get("city", "")),
            "province": str(row.get("province", "")),
            "status": str(row.get("status", "")),
            "facility": str(row.get("facility", "")),
        })

    # ChromaDB has a batch-size limit; chunk to be safe
    BATCH = 500
    for i in range(0, len(documents), BATCH):
        collection.add(
            documents=documents[i : i + BATCH],
            ids=ids[i : i + BATCH],
            metadatas=metadatas[i : i + BATCH],
        )
    print(f"[chromadb] embedded {len(documents)} docs → {CHROMA_DIR}")


def build_bm25(df: pd.DataFrame) -> None:
    df = df.drop_duplicates(subset=["trial_id"], keep="first")
    corpus = [_row_to_document(row).lower().split() for _, row in df.iterrows()]
    trial_ids = list(df["trial_id"].astype(str))
    index = BM25Okapi(corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"index": index, "ids": trial_ids}, f)
    print(f"[bm25]     indexed {len(corpus)} docs → {BM25_PATH}")


def load_data() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["latitude", "longitude"])
    print(f"[csv]      loaded {len(df)} rows from {CSV_PATH}")

    build_geojson(df)
    build_chromadb(df)
    build_bm25(df)
    print("Done — backend is ready to serve.")


if __name__ == "__main__":
    load_data()
