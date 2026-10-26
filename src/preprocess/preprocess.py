"""
Preprocessing for worker messages (Arabic and English).

Steps
  1. detect_language: 'ar', 'en' or 'mixed', from the share of Arabic letters.
  2. mask_pii: replaces emails, phone numbers, Emirates ID numbers and URLs with tags.
     (Names are NOT masked automatically - check by eye while labelling.)
  3. normalize_arabic: removes diacritics and tatweel, unifies alef / alef maksura /
     teh marbuta forms, converts Arabic-Indic digits to 0-9.
  4. normalize_text: lower-cases English, normalises Arabic, collapses spaces.

Usage as a script (cleans the scraped CSV):
    python src/preprocess/preprocess.py --in data/raw/questions.csv --out data/clean/questions_clean.csv

Usage in code:
    from src.preprocess.preprocess import preprocess
    preprocess("My boss owes me AED 5,000, call 050 123 4567")
"""
import argparse
import re

import pandas as pd

AR_LETTER = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL = re.compile(r"https?://\S+|www\.\S+")
EMIRATES_ID = re.compile(r"\b784[-\s]?\d{4}[-\s]?\d{7}[-\s]?\d\b")
# UAE mobiles/landlines (+971 / 00971 / 05x / 04) and other long digit runs
PHONE = re.compile(r"(?:\+|00)?971[\s-]?\d{1,2}[\s-]?\d{3}[\s-]?\d{4}\b|\b0\d{1,2}[\s-]?\d{3}[\s-]?\d{4}\b")


def detect_language(text: str, mixed_threshold: float = 0.2) -> str:
    ar = len(AR_LETTER.findall(text or ""))
    la = len(LATIN_LETTER.findall(text or ""))
    total = ar + la
    if total == 0:
        return "unknown"
    share_ar = ar / total
    if share_ar >= 1 - mixed_threshold:
        return "ar"
    if share_ar <= mixed_threshold:
        return "en"
    return "mixed"


def mask_pii(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = URL.sub("[URL]", text)
    text = EMIRATES_ID.sub("[EID]", text)
    text = PHONE.sub("[PHONE]", text)
    return text


def normalize_arabic(text: str) -> str:
    text = DIACRITICS.sub("", text)
    text = text.replace(TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    return text.translate(AR_DIGITS)


def normalize_text(text: str) -> str:
    text = normalize_arabic(text or "")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> dict:
    masked = mask_pii(text or "")
    return {
        "lang": detect_language(masked),
        "text_masked": masked,
        "text_norm": normalize_text(masked),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/questions.csv")
    ap.add_argument("--out", default="data/clean/questions_clean.csv")
    ap.add_argument("--min-year", type=int, default=2022, help="questions before this year get law_era = pre2022 (old labour law)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, encoding="utf-8-sig")
    n0 = len(df)
    df["full_text"] = (df["title"].fillna("") + ". " + df["question"].fillna("")).str.strip(". ")
    df = df.drop_duplicates(subset=["full_text"])
    # format="mixed": without it pandas guesses one format from the first row and
    # silently turns every date in another format into NaT (which we keep)
    years = pd.to_datetime(df["date"], errors="coerce", utc=True, format="mixed").dt.year
    print(f"{years.isna().sum()} rows have no readable date (law_era = unknown)")
    # keep every row, but tag which labour law applied when it was asked
    df["law_era"] = "unknown"
    df.loc[years < args.min_year, "law_era"] = "pre2022"
    df.loc[years >= args.min_year, "law_era"] = "current"

    out = df["full_text"].apply(preprocess).apply(pd.Series)
    df = pd.concat([df, out], axis=1)
    df = df[df["text_norm"].str.len() > 20]

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"{n0} rows in, {len(df)} rows out -> {args.out}")
    print(df["lang"].value_counts().to_string())


if __name__ == "__main__":
    main()
