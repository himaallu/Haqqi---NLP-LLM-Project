"""
RQ3 hop 2: score each law article by keyword matches and keep the top 3,
one hop (the question alone) vs two hops (the question + its 10 hop-1 Q&As).

Usage (from the repo root, after src/rag/hop1.py):
    python src/rag/hop2.py              # IDF-weighted keyword scores
    python src/rag/hop2.py --no-idf     # plain count of matched keywords

Input:  data/law/article_keywords.csv   (English + Arabic keywords per article)
        data/rag/hop1_neighbors.csv     (each question's 10 similar Q&As)
        data/clean/questions_clean.csv  (Q&A text: question + lawyers' answers)
        data/law/type_article_map.csv   (optional: problem_type -> correct articles)
Output: data/rag/hop2_articles.csv      (query_id, query_lang, problem_type, hops,
                                         rank, article, score, matched)
        results/rq3_recall.csv          (if the map exists: recall@3 and hit@3,
                                         one hop vs two hops, per language)

Matching: text and keywords are normalised the same way as everywhere else
(lower case; Arabic alef / yeh / teh marbuta forms unified, diacritics removed).
A keyword matches as a whole word or phrase; Arabic keywords also match with an
attached prefix (و ف ب ل ك, e.g. "والإجازة"), as Arabic writes them joined.

Score of an article = sum over its keywords found, each keyword counted once
per document and weighted by idf = log(N / df), where df is the number of Q&As
(out of N) containing it: rare, specific keywords count more than common ones
and ties are rare. One hop: the documents are the question. Two hops: the
question and each of its Q&As (title, question and answers) count as one
document each. Top 3 articles by score; articles with score 0 are not returned.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.common import load_haqqi  # noqa: E402
from src.preprocess.preprocess import normalize_text  # noqa: E402

KEYWORDS = "data/law/article_keywords.csv"
HOP1 = "data/rag/hop1_neighbors.csv"
CLEAN = "data/clean/questions_clean.csv"
TYPE_MAP = "data/law/type_article_map.csv"
OUT = "data/rag/hop2_articles.csv"
RECALL = "results/rq3_recall.csv"
TOP = 3
AR_PREFIX = "[وفبلك]{0,2}"  # attached Arabic prefixes: and / so / with / for / like


def load_terms() -> list[tuple[int, str, re.Pattern]]:
    """(article, keyword as written, compiled pattern) for every keyword."""
    kw = pd.read_csv(KEYWORDS, encoding="utf-8-sig", keep_default_na=False)
    terms = []
    for r in kw.itertuples():
        for col in ("keywords_en", "keywords_ar"):
            for term in getattr(r, col).split(";"):
                norm = normalize_text(term.strip())
                if not norm:
                    continue
                prefix = AR_PREFIX if col == "keywords_ar" else ""
                pattern = re.compile(r"(?<!\w)" + prefix + re.escape(norm) + r"(?!\w)")
                terms.append((int(r.article), term.strip(), pattern))
    return terms


def matched_terms(text: str, terms) -> set[int]:
    """Indexes of the keywords found in text."""
    norm = normalize_text(text)
    return {i for i, (_, _, pattern) in enumerate(terms) if pattern.search(norm)}


def qa_text(row) -> str:
    """A Q&A as one document: title, question and all lawyers' answers."""
    return " ".join([row.title, row.question, *json.loads(row.answers)])


def rank_articles(docs: list[set[int]], terms, idf: dict) -> list[tuple[int, float, list[str]]]:
    """Top articles for a set of documents: (article, score, matched keywords)."""
    score, found = defaultdict(float), defaultdict(set)
    for doc in docs:
        for i in doc:
            article, term, _ = terms[i]
            score[article] += idf[i]
            found[article].add(term)
    ranked = sorted(score, key=lambda a: (-score[a], a))[:TOP]  # ties: lower article number
    return [(a, round(score[a], 3), sorted(found[a])) for a in ranked]


def evaluate(out: pd.DataFrame) -> None:
    """recall@3 and hit@3 against the problem-type-to-article map, if it exists."""
    if not os.path.exists(TYPE_MAP):
        print(f"\nNo {TYPE_MAP} yet: fill in the template to get recall@3.")
        return
    m = pd.read_csv(TYPE_MAP, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    m = m[m["article"].str.strip() != ""]
    gold = m.groupby("problem_type")["article"].apply(lambda s: {int(a) for a in s}).to_dict()
    queries = out.drop_duplicates("query_id")[["query_id", "query_lang", "problem_type"]]
    queries = queries[queries["problem_type"].isin(gold)]
    rows = []
    for hops in ("one_hop", "two_hop"):
        got = out[out["hops"] == hops].groupby("query_id")["article"].apply(set).to_dict()
        for q in queries.itertuples():
            g, top = gold[q.problem_type], got.get(q.query_id, set())
            rows.append({"hops": hops, "lang": q.query_lang, "problem_type": q.problem_type,
                         "recall_at_3": len(g & top) / min(len(g), TOP), "hit_at_3": int(bool(g & top))})
    r = pd.DataFrame(rows)
    table = pd.concat([r.groupby(["hops", "lang"])[["recall_at_3", "hit_at_3"]].mean(),
                       r.groupby("hops")[["recall_at_3", "hit_at_3"]].mean().assign(lang="all")
                        .set_index("lang", append=True)]).round(3)
    table["n"] = r.groupby(["hops", "lang"]).size().reindex(table.index).fillna(len(queries)).astype(int)
    os.makedirs(os.path.dirname(RECALL), exist_ok=True)
    table.reset_index().to_csv(RECALL, index=False, encoding="utf-8-sig")
    print(f"\nrecall@3 (relevant articles found / min(relevant, 3)) and hit@3 (at least one found):")
    print(table.to_string())
    print("by problem type:")
    print(r.groupby(["problem_type", "hops"])["recall_at_3"].mean().unstack().round(3).to_string())
    skipped = out.drop_duplicates("query_id")["problem_type"].value_counts().drop(list(gold), errors="ignore")
    print(f"not scored (no articles in the map): {skipped.to_dict()} -> {RECALL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-idf", action="store_true", help="count matched keywords instead of idf weights")
    args = ap.parse_args()

    terms = load_terms()
    hop1 = pd.read_csv(HOP1, encoding="utf-8-sig", dtype=str)
    clean = pd.read_csv(CLEAN, encoding="utf-8-sig", dtype=str, keep_default_na=False).set_index("question_id")

    # keywords found in each Q&A document (computed once, reused by every query)
    pool_ids = sorted((set(hop1["neighbor_id"]) | set(hop1["query_id"])) & set(clean.index))
    qa_docs = {qid: matched_terms(qa_text(clean.loc[qid]), terms) for qid in pool_ids}
    n = len(qa_docs)
    df = defaultdict(int)
    for doc in qa_docs.values():
        for i in doc:
            df[i] += 1
    idf = {i: 1.0 if args.no_idf else math.log(n / df[i]) if df[i] else math.log(n)
           for i in range(len(terms))}

    # the query is the question only (no answer: that is what the system must supply)
    haqqi = load_haqqi()
    haqqi_text = dict(zip(haqqi["question_id"], haqqi["text"])) if len(haqqi) else {}
    def query_text(qid):
        return haqqi_text[qid] if qid in haqqi_text else f"{clean.at[qid, 'title']} {clean.at[qid, 'question']}"

    rows = []
    for qid, g in hop1.groupby("query_id", sort=False):
        q_doc = matched_terms(query_text(qid), terms)
        lang, ptype = g["query_lang"].iat[0], g["problem_type"].iat[0]
        neighbour_docs = [qa_docs[nid] for nid in g.sort_values("rank", key=lambda s: s.astype(int))["neighbor_id"]]
        for hops, docs in (("one_hop", [q_doc]), ("two_hop", [q_doc] + neighbour_docs)):
            for rank, (article, score, found) in enumerate(rank_articles(docs, terms, idf), 1):
                rows.append({"query_id": qid, "query_lang": lang, "problem_type": ptype, "hops": hops,
                             "rank": rank, "article": article, "score": score, "matched": "; ".join(found)})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    nq = hop1["query_id"].nunique()
    print(f"{len(terms)} keywords, {n} Q&A documents, {nq} queries -> {OUT}")
    for hops in ("one_hop", "two_hop"):
        got = out[out["hops"] == hops].groupby("query_id").size()
        print(f"{hops}: queries with >=1 article {len(got) / nq:.0%}, with 3 articles {(got == TOP).sum() / nq:.0%}")
    evaluate(out)


if __name__ == "__main__":
    main()
