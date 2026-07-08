"""Download Atlas Academy nice/export JSON dumps needed for Phase 0.

Usage:
    python scripts/fetch_atlas.py [--force]

Fetches (cached under data/raw/, skipped if already present unless --force):
  - {STRUCTURED_REGION} nice_servant_lang_en.json -> full servant stats/skills/NPs/traits
  - {STRUCTURED_REGION} nice_gacha_lang_en.json    -> banner/pickup history (supplementary)
  - {NAME_REGION} nice_servant.json                -> servant names in NAME_REGION's game
                                                       language, used to match Mooncell wiki
                                                       page titles (which are in Chinese).
"""

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ATLAS_API_BASE, ATLAS_NAME_REGION, ATLAS_STRUCTURED_REGION, RAW_DIR

FILES = [
    (ATLAS_STRUCTURED_REGION, "nice_servant_lang_en", f"{ATLAS_STRUCTURED_REGION.lower()}_nice_servant_lang_en.json"),
    (ATLAS_STRUCTURED_REGION, "nice_gacha_lang_en", f"{ATLAS_STRUCTURED_REGION.lower()}_nice_gacha_lang_en.json"),
    (ATLAS_NAME_REGION, "nice_servant", f"{ATLAS_NAME_REGION.lower()}_nice_servant.json"),
]


def fetch(region: str, export_name: str, out_name: str, force: bool) -> None:
    out_path = RAW_DIR / out_name
    if out_path.exists() and not force:
        print(f"skip (cached): {out_path}")
        return
    url = f"{ATLAS_API_BASE}/export/{region}/{export_name}.json"
    print(f"downloading: {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"saved: {out_path} ({len(resp.content) / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    for region, export_name, out_name in FILES:
        fetch(region, export_name, out_name, args.force)


if __name__ == "__main__":
    main()
