#!/usr/bin/env python3
"""
Download Fuzzwork SDE SQLite database.
Source: https://www.fuzzwork.co.uk/dump/
"""

import hashlib
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
import bz2

FUZZWORK_BASE = "https://www.fuzzwork.co.uk/dump/latest/"
FUZZWORK_DB = "eve.db.bz2"
DATA_DIR = Path(__file__).parent.parent / "data"
COMPRESSED_FILE = DATA_DIR / FUZZWORK_DB
EXTRACTED_FILE = DATA_DIR / "eve.db"


def download_file(url: str, output_path: Path, verbose: bool = True) -> bool:
    """Download file with progress tracking."""
    if verbose:
        print(f"Downloading {url}...", flush=True)

    try:
        req = Request(url, headers={'User-Agent': 'EVESeek/1.0'})
        with urlopen(req, timeout=300) as response:
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024 * 1024  # 1MB chunks
            downloaded = 0

            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size and verbose:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r  {mb_downloaded:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)",
                              end='', flush=True)

            if verbose:
                print()  # newline after progress
        return True

    except URLError as e:
        if verbose:
            print(f"✗ Download failed: {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"✗ Unexpected error: {e}")
        return False


def extract_bz2(compressed_path: Path, extracted_path: Path, verbose: bool = True) -> bool:
    """Extract bz2 file."""
    if verbose:
        print(f"Extracting {compressed_path.name}...", end=" ", flush=True)

    try:
        with bz2.open(compressed_path, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                f_out.write(f_in.read())

        if verbose:
            size_mb = extracted_path.stat().st_size / (1024 * 1024)
            print(f"✓ ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        if verbose:
            print(f"✗ {e}")
        return False


def verify_database(db_path: Path, verbose: bool = True) -> bool:
    """Verify SQLite database integrity."""
    if verbose:
        print(f"Verifying database...", end=" ", flush=True)

    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Quick integrity check
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        conn.close()

        if result == "ok":
            if verbose:
                print(f"✓ Database is valid")
            return True
        else:
            if verbose:
                print(f"✗ {result}")
            return False

    except Exception as e:
        if verbose:
            print(f"✗ {e}")
        return False


def download_all():
    """Download and extract Fuzzwork SDE."""
    print("=" * 80)
    print("EVESeek — Fuzzwork SDE Downloader")
    print("=" * 80)
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Target: {EXTRACTED_FILE}")
    print(f"Source: {FUZZWORK_BASE}{FUZZWORK_DB}")
    print()

    # Check if already extracted
    if EXTRACTED_FILE.exists():
        size_mb = EXTRACTED_FILE.stat().st_size / (1024 * 1024)
        print(f"ℹ Database already exists ({size_mb:.1f} MB)")
        if verify_database(EXTRACTED_FILE):
            print("✓ Database is valid and ready to use")
            return True
        else:
            print("✗ Existing database is corrupted, re-downloading...")
            EXTRACTED_FILE.unlink()

    # Download
    url = f"{FUZZWORK_BASE}{FUZZWORK_DB}"
    if not download_file(url, COMPRESSED_FILE):
        print("✗ Download failed")
        return False

    # Extract
    if not extract_bz2(COMPRESSED_FILE, EXTRACTED_FILE):
        print("✗ Extraction failed")
        return False

    # Verify
    if not verify_database(EXTRACTED_FILE):
        print("✗ Database verification failed")
        return False

    # Cleanup compressed file
    COMPRESSED_FILE.unlink()
    print()
    print("✓ Fuzzwork SDE downloaded and ready!")
    return True


if __name__ == "__main__":
    success = download_all()
    exit(0 if success else 1)
