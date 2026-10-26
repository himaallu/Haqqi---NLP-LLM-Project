"""
RQ3 hop 1: for each question, the 10 most similar Q&As of the same problem type.

Usage (from the repo root):
    python src/rag/hop1.py                         # problem type from the labels
    python src/rag/hop1.py --pred results/rq1_bge_m3_haqqi_predictions.csv
                                                   # problem type from a classifier (end-to-end)
    python src/rag/hop1.py --current-only          # only Q&As asked under the current law

Q&A pool: scraped questions that have at least one lawyer answer and a problem
type (silver label). Similarity: cosine between BGE-M3 embeddings
(data/embeddings/, from src/models/embed.py). BGE-M3 is multilingual, so an
Arabic question can retrieve English Q&As and the other way round.

Never returned for a question: the question itself, and copies of it - the same
question posted twice, sometimes under another title or with one name changed.
A copy is a cosine >= DUPLICATE, or question bodies that are >= SAME_TEXT alike
character by character (difflib ratio).
Queries: every Q&A in the pool (leave-one-out, so retrieval can be tested now)
plus your Haqqi cases once data/labelled/haqqi_50.csv has problem_type.

Output: data/rag/hop1_neighbors.csv
        (query_id, query_lang, problem_type, rank, neighbor_id, neighbor_lang,
         neighbor_law_era, similarity)
"""
import argparse
import difflib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.common import EXCLUDE, load_haqqi, load_silver  # noqa: E402
from src.models.embed import get_embeddings, update_cache  # noqa: E402

CLEAN = "data/clean/questions_clean.csv"
OUT = "data/rag/hop1_neighbors.csv"
K = 10
DUPLICATE = 0.98  # cosine at or above this = the same question posted again
SAME_TEXT = 0.9   # question bodies this alike = the same question (catches a changed title)


def load_pool(current_only: bool = False) -> pd.DataFrame:
    """Scraped Q&As with at least one answer, with their silver problem type."""
    clean = pd.read_csv(CLEAN, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    clean = clean[clean["answers"].map(lambda a: len(json.loads(a)) > 0) & ~clean["question_id"].isin(EXCLUDE)]
    clean = clean.assign(body=clean["question"].str.strip())
    pool = clean[["question_id", "lang", "law_era", "body"]].merge(
        load_silver()[["question_id", "problem_type"]], on="question_id")
    if current_only:
        pool = pool[pool["law_era"] == "current"]
    return pool.reset_index(drop=True)


def is_copy(a: str, b: str) -> bool:
    m = difflib.SequenceMatcher(None, a, b)
    return m.quick_ratio() >= SAME_TEXT and m.ratio() >= SAME_TEXT  # quick_ratio is a cheap upper bound


def neighbours(query_vec: np.ndarray, query_id: str, problem_type: str, pool: pd.DataFrame,
               pool_vecs: np.ndarray, k: int = K, query_body: str = "") -> list[tuple[int, float]]:
    """(pool row, cosine) of the k most similar same-type Q&As, excluding the query and its copies."""
    same = np.flatnonzero(pool["problem_type"].to_numpy() == problem_type)
    sims = pool_vecs[same] @ query_vec  # vectors are unit length, so dot product = cosine
    order = np.argsort(-sims)
    out = []
    for j in order:
        row, sim = same[j], float(sims[j])
        if pool.at[row, "question_id"] == query_id or sim >= DUPLICATE:
            continue
        if query_body and is_copy(query_body, pool.at[row, "body"]):
            continue
        out.append((row, sim))
        if len(out) == k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="classifier predictions CSV (question_id, pred) to take the problem type from")
    ap.add_argument("--current-only", action="store_true", help="pool: only questions asked under the current law")
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()

    update_cache()  # embeds any new questions (e.g. Haqqi cases)
    pool = load_pool(args.current_only)
    pool_vecs = get_embeddings(pool["question_id"].tolist())

    queries = pool[["question_id", "lang", "problem_type"]]
    haqqi = load_haqqi()
    if len(haqqi):
        queries = pd.concat([queries, haqqi[["question_id", "lang", "problem_type"]]], ignore_index=True)
    if args.pred:  # use the classifier's problem type where it has one
        pred = pd.read_csv(args.pred, encoding="utf-8-sig", dtype=str).set_index("question_id")["pred"]
        queries = queries[queries["question_id"].isin(pred.index)].copy()
        queries["problem_type"] = queries["question_id"].map(pred)
    query_vecs = get_embeddings(queries["question_id"].tolist())
    bodies = dict(zip(pool["question_id"], pool["body"]))
    bodies.update(dict(zip(haqqi["question_id"], haqqi["text"])) if len(haqqi) else {})

    rows = []
    for q, vec in zip(queries.itertuples(), query_vecs):
        found = neighbours(vec, q.question_id, q.problem_type, pool, pool_vecs, args.k, bodies.get(q.question_id, ""))
        for rank, (row, sim) in enumerate(found, 1):
            rows.append({"query_id": q.question_id, "query_lang": q.lang, "problem_type": q.problem_type,
                         "rank": rank, "neighbor_id": pool.at[row, "question_id"],
                         "neighbor_lang": pool.at[row, "lang"], "neighbor_law_era": pool.at[row, "law_era"],
                         "similarity": round(sim, 4)})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    per_q = out.groupby("query_id").size().reindex(queries["question_id"], fill_value=0)
    print(f"pool: {len(pool)} Q&As | queries: {len(queries)} -> {OUT}")
    print(f"neighbours per query: median {per_q.median():.0f}, fewer than {args.k}: {(per_q < args.k).sum()} queries")
    print("fewer than k, by problem type:",
          queries.assign(n=per_q.values).query(f"n < {args.k}").groupby("problem_type").size().to_dict())
    print(f"mean similarity: rank 1 {out[out['rank'] == 1]['similarity'].mean():.3f}, "
          f"rank {args.k} {out[out['rank'] == args.k]['similarity'].mean():.3f}")
    cross = (out["query_lang"] != out["neighbor_lang"]).groupby(out["query_lang"]).mean().round(2).to_dict()
    print("share of neighbours in the other language:", cross)
    print("share of neighbours asked before 2022:", round((out["neighbor_law_era"] == "pre2022").mean(), 2))


if __name__ == "__main__":
    main()
