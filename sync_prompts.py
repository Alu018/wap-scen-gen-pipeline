"""
Sync prompts from Google Sheets → local CSV.

The WAP prompt sheet is published to the web as CSV; this script downloads it
to wap_prompts.csv (the canonical seed/few-shot source for the generation
pipeline) and reports which prompt IDs are new since the last sync.

Setup (one-time, already done for the current sheet):
1. In Google Sheets: File → Share → Publish to web
2. Choose "Comma-separated values (.csv)" and copy the URL
3. Paste it below as GOOGLE_SHEETS_URL

Usage:
    python sync_prompts.py
"""

import csv
import os
import time

import requests

# Published-CSV URL of the WAP prompts Google Sheet.
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTNXht2JAIcJykneWCm4OnEeQe3nYCsumjis2d1jq_SmC9mzMAf0fmTjfB0ALcHHqkKhpSow7JKFcTs/pub?output=csv"
LOCAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wap_prompts.csv")


def get_existing_ids() -> set[str]:
    """Read prompt IDs from the current local CSV before overwriting it."""
    if not os.path.exists(LOCAL_CSV):
        return set()
    with open(LOCAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["id"] for row in reader}


def print_new_prompts(old_ids: set[str]) -> None:
    """Compare the freshly synced CSV against old_ids and print any new rows."""
    new_rows = []
    with open(LOCAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] not in old_ids:
                new_rows.append(row)

    print("\n" + "=" * 60)
    if not new_rows:
        print("ℹ️  No new prompts were added in this sync.")
    else:
        print(f"🆕 {len(new_rows)} new prompt(s) added this sync:")
        print("=" * 60)
        for row in new_rows:
            p = row["prompt"]
            print(
                f"  ID {row['id']} | {row.get('context', '')} | "
                f"{p[:120]}{'...' if len(p) > 120 else ''}"
            )
    print("=" * 60)


def download_from_google_sheets() -> bool:
    """Download the published Google Sheet as CSV to LOCAL_CSV."""
    print("📥 Downloading from Google Sheets...")
    try:
        # Google's publish-to-web endpoint caches responses for ~5 minutes;
        # a unique query param + no-cache header forces a fresh copy.
        response = requests.get(
            f"{GOOGLE_SHEETS_URL}&cachebust={int(time.time())}",
            headers={"Cache-Control": "no-cache"},
            allow_redirects=True,
        )
        response.raise_for_status()
        with open(LOCAL_CSV, "wb") as f:
            f.write(response.content)
        with open(LOCAL_CSV, "r", encoding="utf-8") as f:
            num_prompts = len(f.readlines()) - 1  # -1 for header
        print(f"✅ Downloaded {num_prompts} prompts to {LOCAL_CSV}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def main() -> None:
    print("=" * 60)
    print("Google Sheets → Local CSV Sync")
    print("=" * 60)

    old_ids = get_existing_ids()

    if not download_from_google_sheets():
        return

    print_new_prompts(old_ids)

    print("\n✅ Sync complete!")
    print(f"   • Local: {LOCAL_CSV}")
    print("\nNext time you update the Google Sheet, just run:")
    print("   python sync_prompts.py")


if __name__ == "__main__":
    main()
