"""Verify delivered bytes BEFORE running analysis or regenerating timestamped PDFs."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    failures = []
    checked = 0
    for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
        checked += 1
    if failures:
        raise SystemExit("Delivery-byte checksum mismatch (regenerated outputs may differ): " + ", ".join(failures))
    print(f"PASS: {checked} delivered files match their checksums. Run this before regeneration.")


if __name__ == "__main__":
    main()
