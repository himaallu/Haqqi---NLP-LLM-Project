# Project context for Claude Code

MAAI 902 (NLP & LLMs), University of Wollongong in Dubai. Assessment 2, individual, due week 6 (assumed Mon 26 Oct 2026). Deliverables: IEEE-style paper (3,000-4,000 words), working prototype (notebook or web UI), this repository with README, and a demonstration.

## What we are building
A pipeline that reads a UAE worker's labour-problem message in Arabic or English and:
1. preprocesses it (language detection, Arabic normalisation, PII masking, tokenisation),
2. classifies the problem type,
3. extracts facts (amount, dates, duration, employer),
4. retrieves the relevant articles of UAE Federal Decree-Law No. 33 of 2021,
5. generates a plain-language answer that cites those articles.

## Research questions
- RQ1: TF-IDF + logistic regression vs BGE-M3 embeddings + classifier vs LLM prompting, macro-F1 per language.
- RQ2: rules vs LLM for fact extraction, precision/recall per fact type.
- RQ3 (updated design): two-hop retrieval.
  - Hop 1: for a question, the 10 most similar Q&As of the same problem type (BGE-M3), never the question itself.
  - Hop 2: score each law article by keyword matches (English + Arabic keyword table per article) against the question alone (one hop) and against the question plus the 10 Q&As (two hops); keep the top 3.
  - Answer: an LLM answers from the question, facts, Q&As and articles, citing article numbers.
  - Evaluate: recall@3 against the student's problem-type-to-article map, one hop vs two hops, per language; answer quality vs the real lawyer answer for model alone / + Q&As / + Q&As and articles.

## Problem types
unpaid_wages, passport_withheld, gratuity, unfair_termination, recruitment_fees, court_enforcement, other

## Data
- Legal Advice Middle East employment questions (legaladviceme.com/questions): 3,618 English, 2,386 Arabic. Scrape about 500 per language, newest first, pre-2022 dropped.
- Scraping rules: 12 s delay between requests, never use ?lang= URLs (disallowed by robots.txt), save raw HTML then parse offline.
- Test set: real data only, includes the student's 50 Haqqi cases. Synthetic data, if any, is training only and tagged.
- Law: English PDF from MoHRE, Arabic PDF from u.ae; split into articles with their numbers.

## Conventions
- Python 3.10+, scripts run from the repo root, outputs in data/ and results/.
- CSV files saved as utf-8-sig so Arabic opens correctly in Excel.
- Keep code simple and commented; this is graded on clarity.
- Never put API keys in code; use environment variables.
- The student writes the paper in their own words. Help with code, analysis and explanations, and flag anything that needs checking.
