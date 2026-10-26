"""
Split Federal Decree-Law No. 33 of 2021 (with its amendments) into articles,
in English and Arabic, for RQ3 retrieval and answer generation.

Usage (from the repo root):
    python src/law/split_articles.py

Input:  data/law/decree_law_33_2021_en.pdf  (MoHRE, consolidated with amendments)
        data/law/decree_law_33_2021_ar.pdf  (u.ae, consolidated with amendments)
Output: data/law/articles.csv  (article, lang, title, amended, text, pages), utf-8-sig
        amended = the PDF marks the article with "*" (changed by a later decree-law)
        prints a check: which of articles 1-74 were found in each language

Both PDFs also contain the executive regulations (Cabinet Resolution No. 1 of
2022). Only the Decree-Law's own articles are kept: the English PDF pages
before the resolution's cover (LAW_PAGES_EN), and in Arabic the articles up to
the first repeat of "Article (1)", where the resolution starts.

English: text is read directly. Each PDF page holds two printed pages side by
side, so each half is read top to bottom. The PDF lost the "h" of the "Th"
ligature ("Te worker"); the affected words are fixed with a fixed list.
Arabic: the PDF's embedded text is unusable (broken letter shapes), so every page
is rendered at 300 dpi and read with Tesseract OCR (Arabic model). OCR text is
cached in data/law/ocr_ar/, so only the first run is slow (~2 minutes).
OCR makes small errors, mostly in tables (e.g. the definitions in Article 1).
Known issue: the Arabic PDF gives the original fines in Article 60, while the
English PDF gives the amended ones; OCR also lost items 1-3 of Arabic Article 60.
"""
import os
import re
import subprocess

import pandas as pd
import pymupdf

EN_PDF = "data/law/decree_law_33_2021_en.pdf"
AR_PDF = "data/law/decree_law_33_2021_ar.pdf"
OUT = "data/law/articles.csv"
OCR_DIR = "data/law/ocr_ar"
TESSDATA = ".venv/share/tessdata"  # holds ara.traineddata (Tesseract's Arabic model)
N_ARTICLES = 74
LAW_PAGES_EN = range(3, 23)  # 0-based: pages 4-23; page 24 starts the Cabinet Resolution

# words whose "Th" ligature lost its "h" in the English PDF
LOST_H = {"Te": "The", "Tat": "That", "Tis": "This", "Tey": "They", "Terein": "Therein",
          "Tereof": "Thereof", "Tirty": "Thirty", "Ten": "Then"}
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}

EN_HEAD = re.compile(r"Article \((\d+)\)\s*(\*?)")  # "*" marks an amended article
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
# "المادة (12)" - OCR sometimes turns the brackets round or drops them
# (OCR sometimes drops the final letter: "الماد (33)")
#  or the "ل": "اماد (66)")
AR_HEAD = re.compile(r"^\s*ال?ماد[ةه]?\s*[\(\)]?\s*(\d{1,2})\s*[\(\)]?\s*(\*?)\s*$")
BIDI_MARKS = re.compile("[\u200e\u200f\u202a-\u202e]")  # invisible direction marks from OCR
# the law ends with the President's signature; the Cabinet Resolution follows
END_EN, END_AR = "Khalifa bin Zayed", "خليفة بن زايد"
MAX_TITLE_CHARS = 55  # Arabic titles are short; penalty articles (59-63) have none
MAX_SKIP = 2  # accept a heading up to 2 numbers ahead if one was unreadable (reported as missing)


def clean_en(text: str) -> str:
    for a, b in LIGATURES.items():
        text = text.replace(a, b)
    return re.sub(r"\b(" + "|".join(LOST_H) + r")\b", lambda m: LOST_H[m.group(1)], text)


def en_lines() -> list[tuple[int, str]]:
    """(page, line) for the law pages, each half page read top to bottom."""
    doc = pymupdf.open(EN_PDF)
    out = []
    for i in LAW_PAGES_EN:
        page = doc[i]
        mid = page.rect.width / 2
        blocks = sorted(page.get_text("blocks"), key=lambda b: (b[0] >= mid, b[1]))
        for b in blocks:
            for line in clean_en(b[4]).splitlines():
                line = line.strip()
                if line and not re.fullmatch(r"[\d\s]+", line):  # drop printed page numbers
                    out.append((i + 1, line))
    return out


def split_en() -> list[dict]:
    articles, cur, pending_title = [], None, ""
    lines = en_lines()
    for k, (page, line) in enumerate(lines):
        if cur and cur["article"] == N_ARTICLES and END_EN in line:
            break
        head = EN_HEAD.search(line)
        title_here = line.lstrip("►").replace(head.group(0), "").strip() if head and line.startswith("►") else ""
        if head and (line == head.group(0).strip() or line.startswith("►")):
            cur = {"article": int(head.group(1)), "lang": "en", "title": title_here or pending_title,
                   "amended": head.group(2) == "*", "lines": [], "pages": {page}}
            articles.append(cur)
            pending_title = ""
            continue
        if line.startswith("►"):
            nxt = lines[k + 1][1] if k + 1 < len(lines) else ""
            if EN_HEAD.fullmatch(nxt.strip()):  # title printed just above its heading
                pending_title = line.lstrip("►").strip()
                continue
            if cur and not cur["title"]:
                cur["title"] = line.lstrip("►").strip()
                continue
        if cur:
            cur["lines"].append(line)
            cur["pages"].add(page)
    return articles


def ocr_page(doc, i: int) -> str:
    path = f"{OCR_DIR}/page_{i + 1:03d}.txt"
    if not os.path.exists(path):
        png = f"{OCR_DIR}/page_{i + 1:03d}.png"
        doc[i].get_pixmap(dpi=300).save(png)
        text = subprocess.run(["tesseract", png, "-", "-l", "ara", "--psm", "6"], capture_output=True,
                              text=True, env={**os.environ, "TESSDATA_PREFIX": TESSDATA}).stdout
        os.remove(png)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    with open(path, encoding="utf-8") as f:
        return f.read()


def is_ocr_noise(line: str) -> bool:
    """Page numbers and the decorative page header that OCR turns into symbols."""
    chars = [c for c in line if not c.isspace()]
    if not chars or re.fullmatch(r"[\d\s]+", line):
        return True
    arabic = sum("\u0600" <= c <= "\u06ff" for c in chars)
    return len(chars) > 4 and arabic / len(chars) < 0.5 and not re.search(r"\d", line[:3])


def split_ar() -> list[dict]:
    os.makedirs(OCR_DIR, exist_ok=True)
    doc = pymupdf.open(AR_PDF)
    articles, cur, expect_title = [], None, False
    for i in range(len(doc)):
        for line in ocr_page(doc, i).translate(AR_DIGITS).splitlines():
            line = BIDI_MARKS.sub("", line).strip()
            if cur and cur["article"] == N_ARTICLES and END_AR in line:
                return articles
            head = AR_HEAD.match(line)
            if head:
                n = int(head.group(1))
                if articles and n == 1:  # second "Article (1)": the Cabinet Resolution starts
                    return articles
                expected = cur["article"] + 1 if cur else 1
                if expected <= n <= expected + MAX_SKIP:  # not a cross-reference to an earlier article
                    cur = {"article": n, "lang": "ar", "title": "", "amended": head.group(2) == "*",
                           "lines": [], "pages": {i + 1}}
                    articles.append(cur)
                    expect_title = True
                    continue
            if not cur or not line or is_ocr_noise(line):
                continue
            if expect_title:  # the line after the heading is the title, if it is short
                expect_title = False
                if len(line) <= MAX_TITLE_CHARS:
                    cur["title"] = re.sub(r"^[4►]\s+", "", line)  # OCR reads "►" as "4"
                    continue
            cur["lines"].append(line)
            cur["pages"].add(i + 1)
    return articles


def main():
    rows = []
    for arts in (split_en(), split_ar()):
        for a in arts:
            rows.append({"article": a["article"], "lang": a["lang"], "title": a["title"], "amended": a["amended"],
                         "text": "\n".join(a["lines"]).strip(),
                         "pages": ",".join(map(str, sorted(a["pages"])))})
    df = pd.DataFrame(rows)
    # "amended" belongs to the article, so take it from either language (OCR can miss the "*")
    amended = set(df.loc[df["amended"], "article"])
    df["amended"] = df["article"].isin(amended)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"-> {OUT}")
    for lang, g in df.groupby("lang"):
        nums = g["article"].tolist()
        missing = sorted(set(range(1, N_ARTICLES + 1)) - set(nums))
        dups = sorted({n for n in nums if nums.count(n) > 1})
        print(f"{lang}: {len(g)} articles | missing {missing or 'none'} | duplicates {dups or 'none'} "
              f"| empty text {int((g['text'] == '').sum())} | median chars {int(g['text'].str.len().median())} "
              f"| marked amended {sorted(g.loc[g['amended'], 'article'].tolist())}")


if __name__ == "__main__":
    main()
