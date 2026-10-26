"""
BGE-M3 sentence embeddings for every question, computed once and cached.

Used by the BGE-M3 classifier (RQ1) and by hop 1 of retrieval (RQ3: the most
similar Q&As). BGE-M3 is multilingual, so Arabic and English questions share one
vector space.

Usage (from the repo root; the first run downloads the model, about 2.3 GB):
    python src/models/embed.py

Input:  data/clean/questions_clean.csv (+ data/labelled/haqqi_50.csv if present)
Output: data/embeddings/bge_m3.npy      (one row per question, unit length)
        data/embeddings/bge_m3_ids.csv  (question_id for each row)
Only questions not yet in the cache are embedded, so re-running is cheap.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.label.build_queue import all_questions  # noqa: E402

MODEL = "BAAI/bge-m3"
VECS = "data/embeddings/bge_m3.npy"
IDS = "data/embeddings/bge_m3_ids.csv"
MAX_TOKENS = 512  # longer questions are cut here; most are far shorter


def _load_cache() -> tuple[list, np.ndarray]:
    if not os.path.exists(VECS):
        return [], np.zeros((0, 0), dtype=np.float32)
    return pd.read_csv(IDS, dtype=str)["question_id"].tolist(), np.load(VECS)


def update_cache() -> None:
    """Embed every question that is not cached yet."""
    import torch
    from sentence_transformers import SentenceTransformer

    qs = all_questions()
    ids, vecs = _load_cache()
    todo = qs[~qs["question_id"].isin(set(ids))]
    print(f"{len(ids)} cached, {len(todo)} to embed")
    if todo.empty:
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple GPU if present
    model = SentenceTransformer(MODEL, device=device)
    model.max_seq_length = MAX_TOKENS
    new = model.encode(todo["text_masked"].tolist(), batch_size=16, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    vecs = new if len(ids) == 0 else np.vstack([vecs, new])
    ids = ids + todo["question_id"].tolist()
    os.makedirs(os.path.dirname(VECS), exist_ok=True)
    np.save(VECS, vecs)
    pd.DataFrame({"question_id": ids}).to_csv(IDS, index=False)
    print(f"-> {VECS} {vecs.shape}")


def get_embeddings(question_ids: list) -> np.ndarray:
    """Embeddings in the order of question_ids (run update_cache first)."""
    ids, vecs = _load_cache()
    pos = {q: i for i, q in enumerate(ids)}
    missing = [q for q in question_ids if q not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no embedding; run python src/models/embed.py")
    return vecs[[pos[q] for q in question_ids]]


if __name__ == "__main__":
    update_cache()
