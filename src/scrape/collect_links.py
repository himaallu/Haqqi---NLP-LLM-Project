"""
Step 1 of scraping: collect question links from Legal Advice Middle East listing pages.

How to use
----------
1. Open https://legaladviceme.com/questions in your browser, apply the
   "Employment" area-of-law filter, and copy the URL from the address bar.
   (Do NOT add a ?lang= parameter: robots.txt disallows it. We detect
   the language ourselves later.)
2. Run:
       python src/scrape/collect_links.py --start-url "<that URL>" --max-pages 70

   Each listing page has about 16 questions, so 70 pages ~ 1,100 links.

Output: data/raw/links.csv
        (url, question_id, listing_lang, slug_lang, listing_page, collected_at)
listing_lang comes from the site's own language filter (language=41 en, 7 ar).
slug_lang is 'ar' if the URL slug is written in Arabic letters, else 'en'.
The Arabic filter also lists English questions that have an Arabic translation;
those have an English slug. Questions written in Arabic have an Arabic slug, so
slug_lang lets fetch_pages.py keep only Arabic originals. We still detect the
language from the text itself in preprocessing.

Politeness: waits --delay seconds between pages (default 12, robots.txt asks
named bots for 10-15 s). The script resumes: links already in links.csv are kept.

Optional: --use-agentql uses AgentQL (as Professor Asma did) instead of a plain
CSS/regex search for links. Needs `pip install agentql` and AGENTQL_API_KEY set.
"""
import argparse
import csv
import os
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse, unquote

from playwright.sync_api import sync_playwright

QUESTION_RE = re.compile(r"^https?://(www\.)?legaladviceme\.com/questions/(\d+)/[^/?#]+")
OUT = "data/raw/links.csv"
FIELDS = ["url", "question_id", "listing_lang", "slug_lang", "listing_page", "collected_at"]
ARABIC_LETTER = re.compile(r"[\u0600-\u06FF]")
# language ids used by the site's own filter
SITE_LANGS = {"41": "en", "7": "ar"}

AGENTQL_QUERY = """
{
    questions[] {
        title
        link(the URL of the question)
    }
}
"""


def page_url(start_url: str, page_no: int) -> str:
    """Add or replace ?page=N on the filtered listing URL."""
    parts = urlparse(start_url)
    qs = parse_qs(parts.query)
    if page_no > 1:
        qs["page"] = [str(page_no)]
    else:
        qs.pop("page", None)
    return urlunparse(parts._replace(query=urlencode(qs, doseq=True)))


def slug_lang(url: str) -> str:
    """'ar' if the last part of the URL (the slug) contains Arabic letters."""
    slug = unquote(url.rstrip("/").split("/")[-1])
    return "ar" if ARABIC_LETTER.search(slug) else "en"


def links_plain(page) -> list[str]:
    hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
    return [h for h in hrefs if QUESTION_RE.match(h)]


def links_agentql(page) -> list[str]:
    import agentql  # imported only when asked for

    wrapped = agentql.wrap(page)
    data = wrapped.query_data(AGENTQL_QUERY)
    out = []
    for q in data.get("questions", []):
        link = q.get("link") or ""
        if link.startswith("/"):
            link = "https://legaladviceme.com" + link
        if QUESTION_RE.match(link):
            out.append(link)
    return out


def load_existing() -> dict[str, dict]:
    if not os.path.exists(OUT):
        return {}
    with open(OUT, newline="", encoding="utf-8") as f:
        rows = {r["url"]: r for r in csv.DictReader(f)}
    for r in rows.values():  # fill the column for files made before it existed
        r["slug_lang"] = r.get("slug_lang") or slug_lang(r["url"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-url", required=True, help="Employment-filtered listing URL")
    ap.add_argument("--first-page", type=int, default=1)
    ap.add_argument("--max-pages", type=int, default=70)
    ap.add_argument("--delay", type=float, default=12.0)
    ap.add_argument("--use-agentql", action="store_true")
    ap.add_argument("--headful", action="store_true", help="Show the browser window")
    args = ap.parse_args()

    if "lang=" in args.start_url:
        raise SystemExit("Remove the lang= parameter from the URL (robots.txt disallows it).")

    lang_id = parse_qs(urlparse(args.start_url).query).get("language", [""])[0]
    listing_lang = SITE_LANGS.get(lang_id, "unknown")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    seen = load_existing()
    print(f"Already have {len(seen)} links")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        # contact address for the site owner, kept out of the code: export SCRAPER_CONTACT=you@example.com
        contact = os.environ.get("SCRAPER_CONTACT", "see README")
        page = browser.new_page(user_agent=f"UOWD MAAI902 student research (contact: {contact})")
        for n in range(args.first_page, args.first_page + args.max_pages):
            url = page_url(args.start_url, n)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                page.wait_for_timeout(1500)
                found = links_agentql(page) if args.use_agentql else links_plain(page)
            except Exception as e:  # keep going on a bad page
                print(f"page {n}: error {e}")
                time.sleep(args.delay)
                continue

            new = 0
            for link in dict.fromkeys(found):  # de-duplicate, keep order
                clean = QUESTION_RE.match(link).group(0)
                if clean not in seen:
                    seen[clean] = {
                        "url": clean,
                        "question_id": QUESTION_RE.match(clean).group(2),
                        "listing_lang": listing_lang,
                        "slug_lang": slug_lang(clean),
                        "listing_page": n,
                        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
                    new += 1
            print(f"page {n}: {len(found)} links, {new} new, total {len(seen)}")

            # write after every page so a crash loses nothing
            with open(OUT, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
                w.writeheader()
                w.writerows(seen.values())

            if not found:
                print("No links found on this page; stopping.")
                break
            time.sleep(args.delay)
        browser.close()


if __name__ == "__main__":
    main()
