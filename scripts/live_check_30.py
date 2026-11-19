"""Live verification script running extraction with Gemini API on all 30 synthetic notices.

Measures:
- doc_type accuracy
- deadline_date accuracy
- amount_inr accuracy
- reference_number accuracy
- Complete list of mismatches and failures with notice ID, expected vs. actual, and error messages.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extract import extract_notice, get_cached_labels
from src.llm import is_live_gemini_available
from src.schemas import DocumentType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("live_check")


def normalize_str(val: Any) -> str:
    """Normalize string for robust comparison (strip whitespace, uppercase)."""
    if val is None:
        return ""
    return str(val).strip().upper()


def normalize_amount(val: Any) -> Any:
    """Normalize numeric amount for comparison."""
    if val is None:
        return None
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return val


def run_live_check() -> Dict[str, Any]:
    if not is_live_gemini_available():
        logger.error("GEMINI_API_KEY is not available or is a placeholder!")
        sys.exit(1)

    labels_file = get_cached_labels()
    if not labels_file:
        logger.error("Failed to load labels.json")
        sys.exit(1)

    notices = labels_file.notices
    total_count = len(notices)
    logger.info("Starting live verification across %d synthetic notices...", total_count)

    doc_type_matches = 0
    deadline_matches = 0
    amount_matches = 0
    ref_matches = 0

    failures: List[Dict[str, Any]] = []
    results_summary: List[Dict[str, Any]] = []

    for idx, notice in enumerate(notices, 1):
        notice_id = notice.filename
        file_path = PROJECT_ROOT / "data" / "test_notices" / notice_id
        logger.info("[%d/%d] Extracting %s (doc_type: %s)...", idx, total_count, notice_id, notice.doc_type)

        start_t = time.time()
        try:
            res = extract_notice(file_path, force_mock=False)
            latency = round(time.time() - start_t, 2)
        except Exception as exc:
            logger.error("Extraction crashed for %s: %s", notice_id, exc)
            failures.append({
                "notice_id": notice_id,
                "field": "FATAL_CRASH",
                "expected": "Success",
                "actual": str(exc),
                "error": str(exc)
            })
            continue

        if res.is_mock:
            logger.warning("WARNING: Result for %s was marked is_mock=True! Live API must be used.", notice_id)

        ext = res.extraction
        
        # 1. doc_type check
        expected_doc_type = notice.doc_type.value if hasattr(notice.doc_type, "value") else str(notice.doc_type)
        actual_doc_type = ext.doc_type.value if hasattr(ext.doc_type, "value") else str(ext.doc_type)
        dt_match = (actual_doc_type == expected_doc_type)
        if dt_match:
            doc_type_matches += 1
        else:
            failures.append({
                "notice_id": notice_id,
                "field": "doc_type",
                "expected": expected_doc_type,
                "actual": actual_doc_type,
                "error": res.error or "Mismatch"
            })

        # 2. deadline_date check
        expected_deadline = str(notice.deadline_date) if notice.deadline_date else None
        actual_deadline = str(ext.deadline_date) if ext.deadline_date else None
        deadline_match = (actual_deadline == expected_deadline)
        if deadline_match:
            deadline_matches += 1
        else:
            failures.append({
                "notice_id": notice_id,
                "field": "deadline_date",
                "expected": expected_deadline,
                "actual": actual_deadline,
                "error": res.error or "Mismatch"
            })

        # 3. amount_inr check
        expected_amount = normalize_amount(notice.amount_inr)
        actual_amount = normalize_amount(ext.amount_inr)
        amount_match = (actual_amount == expected_amount)
        if amount_match:
            amount_matches += 1
        else:
            failures.append({
                "notice_id": notice_id,
                "field": "amount_inr",
                "expected": expected_amount,
                "actual": actual_amount,
                "error": res.error or "Mismatch"
            })

        # 4. reference_number check
        # Fuzzy / normalized check for reference number (strip whitespace / case)
        expected_ref = normalize_str(notice.reference_number)
        actual_ref = normalize_str(ext.reference_number)
        ref_match = (actual_ref == expected_ref) or (expected_ref and expected_ref in actual_ref) or (actual_ref and actual_ref in expected_ref)
        if ref_match:
            ref_matches += 1
        else:
            failures.append({
                "notice_id": notice_id,
                "field": "reference_number",
                "expected": notice.reference_number,
                "actual": ext.reference_number,
                "error": res.error or "Mismatch"
            })

        results_summary.append({
            "notice_id": notice_id,
            "category": str(notice.doc_type),
            "is_mock": res.is_mock,
            "latency": latency,
            "doc_type_match": dt_match,
            "deadline_match": deadline_match,
            "amount_match": amount_match,
            "ref_match": ref_match,
            "confidence": ext.confidence,
            "error": res.error
        })

        # Short pause between calls to respect rate limits
        time.sleep(1.0)

    stats = {
        "total_notices": total_count,
        "doc_type_accuracy": round(doc_type_matches / total_count, 4),
        "deadline_date_accuracy": round(deadline_matches / total_count, 4),
        "amount_inr_accuracy": round(amount_matches / total_count, 4),
        "reference_number_accuracy": round(ref_matches / total_count, 4),
        "doc_type_matches": doc_type_matches,
        "deadline_matches": deadline_matches,
        "amount_matches": amount_matches,
        "ref_matches": ref_matches,
        "failures_count": len(failures),
        "failures": failures,
        "details": results_summary
    }

    print("\n" + "="*60)
    print("LIVE VERIFICATION RESULTS (30 SYNTHETIC NOTICES)")
    print("="*60)
    print(f"Total Notices Evaluated: {total_count}")
    print(f"doc_type Accuracy:        {stats['doc_type_accuracy'] * 100:.1f}% ({doc_type_matches}/{total_count})")
    print(f"deadline_date Accuracy:   {stats['deadline_date_accuracy'] * 100:.1f}% ({deadline_matches}/{total_count})")
    print(f"amount_inr Accuracy:      {stats['amount_inr_accuracy'] * 100:.1f}% ({amount_matches}/{total_count})")
    print(f"reference_number Accuracy: {stats['reference_number_accuracy'] * 100:.1f}% ({ref_matches}/{total_count})")
    print(f"Total Field Mismatches:   {len(failures)}")
    print("="*60)

    # Save to JSON report
    out_path = PROJECT_ROOT / "live_verification_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Full details saved to {out_path}")

    return stats


if __name__ == "__main__":
    run_live_check()
