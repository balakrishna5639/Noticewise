"""Knowledge base indexing script for NoticeWise (F4).

Parses official markdown documents in kb/, extracts title/URL metadata,
chunks content logically by statutory sections, and saves pre-indexed
chunks to data/kb_index.json so the runtime app starts instantly with zero API calls.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_index")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "kb"
DATA_DIR = PROJECT_ROOT / "data"
INDEX_OUTPUT_PATH = DATA_DIR / "kb_index.json"


def parse_header_metadata(lines: List[str]) -> Dict[str, str]:
    """Extract metadata header lines from top of markdown file."""
    meta = {
        "title": "Official Administrative Procedure",
        "source_url": "https://www.india.gov.in/",
        "date_accessed": "2026-11-19",
        "state": "National / Telangana",
    }
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        lower = stripped.lower()
        if "title:" in lower:
            meta["title"] = stripped.split(":", 1)[1].strip()
        elif "source url:" in lower or "url:" in lower:
            meta["source_url"] = stripped.split(":", 1)[1].strip()
        elif "date accessed:" in lower:
            meta["date_accessed"] = stripped.split(":", 1)[1].strip()
        elif "state:" in lower or "scope:" in lower:
            meta["state"] = stripped.split(":", 1)[1].strip()
    return meta


def chunk_document(file_path: Path) -> List[Dict[str, Any]]:
    """Parse and chunk a single KB markdown document into structured sections."""
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    meta = parse_header_metadata(lines)
    stem = file_path.stem

    # Separate header lines from content body
    body_lines = []
    in_header = True
    for line in lines:
        stripped = line.strip()
        if in_header and stripped.startswith("#") and any(k in stripped.lower() for k in ["title:", "source url:", "url:", "date accessed:", "state:", "scope:"]):
            continue
        in_header = False
        body_lines.append(line)

    body = "\n".join(body_lines).strip()
    if not body:
        return []

    # Map file stem to target document type
    doc_type_mapping = {
        "01_rti": "rti",
        "02_rti": "rti",
        "03_rti": "rti",
        "04_echallan": "challan",
        "05_telangana_echallan": "challan",
        "06_tsspdcl": "electricity",
        "07_tserc": "electricity",
        "08_electricity": "electricity",
        "09_ghmc": "tax",
        "10_ghmc": "tax",
        "11_nfsa": "scheme_rejection",
        "12_telangana_aasara": "scheme_rejection",
    }
    assigned_doc_type = "other"
    for prefix, dtype in doc_type_mapping.items():
        if stem.startswith(prefix):
            assigned_doc_type = dtype
            break

    # Split document by markdown section headers (## ) or numbered sections (\n\n(?=\d+\.\s+))
    if re.search(r"\n(?=##\s+)", body):
        sections = re.split(r"\n(?=##\s+)", body)
    elif re.search(r"\n\n(?=\d+\.\s+)", body):
        sections = re.split(r"\n\n(?=\d+\.\s+)", body)
    else:
        sections = [body]

    chunks: List[Dict[str, Any]] = []
    chunk_idx = 1
    for sec in sections:
        cleaned_sec = sec.strip()
        if not cleaned_sec:
            continue

        chunk_id = f"{stem.upper()}_C{chunk_idx}"
        chunks.append({
            "chunk_id": chunk_id,
            "title": meta["title"],
            "source_url": meta["source_url"],
            "date_accessed": meta["date_accessed"],
            "doc_type": assigned_doc_type,
            "content": cleaned_sec,
        })
        chunk_idx += 1

    return chunks


def build_kb_index() -> List[Dict[str, Any]]:
    """Build knowledge base index across all files in kb/ and write to data/kb_index.json."""
    if not KB_DIR.exists():
        raise FileNotFoundError(f"Knowledge base directory '{KB_DIR}' does not exist.")

    kb_files = sorted(KB_DIR.glob("*.md"))
    if not kb_files:
        raise ValueError(f"No markdown files found in '{KB_DIR}'.")

    logger.info("Found %d official documents in %s", len(kb_files), KB_DIR)
    all_chunks: List[Dict[str, Any]] = []

    for f in kb_files:
        chunks = chunk_document(f)
        all_chunks.extend(chunks)
        logger.info("Indexed '%s': %d chunks", f.name, len(chunks))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_OUTPUT_PATH, "w", encoding="utf-8") as out:
        json.dump(all_chunks, out, indent=2)

    logger.info("Successfully built knowledge base index with %d chunks at %s", len(all_chunks), INDEX_OUTPUT_PATH)
    return all_chunks


if __name__ == "__main__":
    build_kb_index()
