"""
Step 2 of scraping: open each question link with Playwright and save the raw HTML.

We save the whole page first and parse it later (parse_pages.py). That way, if
the parser needs fixing, you never have to download the pages again.

Usage:
    python src/scrape/fetch_pages.py --limit 1000

Input:  data/raw/links.csv
Output: data/raw/html/<question_id>.html   (skips files that already exist)

Links from the Arabic listing with an English slug are skipped: they are English
questions that the site has translated, not questions written in Arabic.
Use --include-translations to fetch them anyway.
"""
import argparse
import csv
import os
import time
from itertools import zip_longest

from playwright.sync_api import sync_playwright

LINKS = "data/raw/links.csv"
HTML_DIR = "data/raw/html"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--delay", type=float, default=12.0)
    ap.add_argument("--include-translations", action="store_true",
                    help="also fetch Arabic-listing links that have an English slug")
    args = ap.parse_args()

    os.makedirs(HTML_DIR, exist_ok=True)
    with open(LINKS, newline="", encoding="utf-8") as f:
        links = list(csv.DictReader(f))

    if not args.include_translations:
        translated = [r for r in links if r.get("listing_lang") == "ar" and r.get("slug_lang") == "en"]
        links = [r for r in links if r not in translated]
        print(f"skipping {len(translated)} translated questions from the Arabic listing")

    todo = [r for r in links if not os.path.exists(f"{HTML_DIR}/{r['question_id']}.html")]
    # alternate en / ar so --limit does not use up the whole budget on the
    # English links, which come first in links.csv
    by_lang = {}
    for r in todo:
        by_lang.setdefault(r.get("listing_lang", ""), []).append(r)
    todo = [r for group in zip_longest(*by_lang.values()) for r in group if r][: args.limit]
    print(f"{len(links)} links, {len(todo)} to fetch")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # contact address for the site owner, kept out of the code: export SCRAPER_CONTACT=you@example.com
        contact = os.environ.get("SCRAPER_CONTACT", "see README")
        page = browser.new_page(user_agent=f"UOWD MAAI902 student research (contact: {contact})")
        for i, r in enumerate(todo, 1):
            try:
                page.goto(r["url"], wait_until="domcontentloaded", timeout=60_000)
                page.wait_for_timeout(1500)
                html = page.content()
                with open(f"{HTML_DIR}/{r['question_id']}.html", "w", encoding="utf-8") as out:
                    out.write(html)
                print(f"[{i}/{len(todo)}] saved {r['question_id']}")
            except Exception as e:
                print(f"[{i}/{len(todo)}] error {r['url']}: {e}")
            time.sleep(args.delay)
        browser.close()


if __name__ == "__main__":
    main()
