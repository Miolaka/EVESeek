#!/usr/bin/env python3
"""
Download Hoboleaks SDE data (JSON format) and verify checksums.
Source: https://sde.hoboleaks.space/
"""

import json
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

HOBOLEAKS_BASE = "https://sde.hoboleaks.space/tq/"
HOBOLEAKS_META = f"{HOBOLEAKS_BASE}meta.json"

# Essential SDE files for BOM calculations
# Hoboleaks provides these core files
REQUIRED_FILES = {
    "blueprints.json": "Blueprint definitions (what produces what)",
    "typematerials.json": "Material requirements per type",
    "industryactivities.json": "Manufacturing/reaction/refining activities",
    "industryassemblylines.json": "Assembly line definitions",
    "industryinstallationtypes.json": "Installation types (station classes)",
    "industrymodifiersources.json": "Industry modifier sources",
}

SDE_DIR = Path(__file__).parent.parent / "data" / "sde_raw"
MANIFEST_FILE = SDE_DIR / "manifest.json"


def download_file(filename: str, verbose: bool = True) -> bool:
    """Download a single file from Hoboleaks and verify."""
    url = f"{HOBOLEAKS_BASE}{filename}"
    filepath = SDE_DIR / filename

    if verbose:
        print(f"Downloading {filename}...", end=" ", flush=True)

    try:
        with urlopen(url, timeout=30) as response:
            data = response.read()

        # Calculate MD5
        md5 = hashlib.md5(data).hexdigest()

        # Save file
        with open(filepath, "wb") as f:
            f.write(data)

        if verbose:
            size_mb = len(data) / (1024 * 1024)
            print(f"✓ ({size_mb:.1f} MB, MD5: {md5[:8]}...)")

        return True

    except URLError as e:
        if verbose:
            print(f"✗ (Error: {e})")
        return False
    except Exception as e:
        if verbose:
            print(f"✗ (Unexpected error: {e})")
        return False


def verify_files() -> bool:
    """Check if all required files exist and are valid JSON."""
    all_valid = True
    for filename in REQUIRED_FILES.keys():
        filepath = SDE_DIR / filename
        if not filepath.exists():
            print(f"✗ {filename} — file not found")
            all_valid = False
            continue

        try:
            with open(filepath, "r") as f:
                json.load(f)
            print(f"✓ {filename} — valid JSON ({filepath.stat().st_size / (1024*1024):.1f} MB)")
        except json.JSONDecodeError as e:
            print(f"✗ {filename} — invalid JSON: {e}")
            all_valid = False

    return all_valid


def download_all():
    """Download all required SDE files."""
    print("=" * 80)
    print("EVESeek — Hoboleaks SDE Downloader")
    print("=" * 80)
    print()

    print(f"Target directory: {SDE_DIR}")
    print(f"Source: {HOBOLEAKS_BASE}")
    print()

    print("Files to download:")
    for filename, description in REQUIRED_FILES.items():
        print(f"  • {filename}")
        print(f"    └─ {description}")
    print()

    # Download files
    print("Downloading files:")
    success_count = 0
    for filename in REQUIRED_FILES.keys():
        if download_file(filename):
            success_count += 1

    print()
    print(f"Downloaded: {success_count}/{len(REQUIRED_FILES)}")

    if success_count != len(REQUIRED_FILES):
        print("⚠ Some files failed to download.")
        return False

    # Verify
    print()
    print("Verifying files:")
    if verify_files():
        print()
        print("✓ All files downloaded and verified successfully!")
        return True
    else:
        print()
        print("✗ Some files failed verification.")
        return False


if __name__ == "__main__":
    SDE_DIR.mkdir(parents=True, exist_ok=True)
    success = download_all()
    exit(0 if success else 1)
