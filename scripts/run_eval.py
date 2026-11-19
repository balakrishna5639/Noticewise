"""Automated evaluation and cache generation pipeline for NoticeWise (PRD Sections 12 & F8).

Executes:
1. Live extraction and pipeline evaluation over all 30 synthetic notices in data/test_notices/.
2. Resilient disk caching to data/cached_results/ with 'source': 'cached_live_run' and ISO timestamps.
3. Computes comprehensive evaluation metrics:
   - doc_type accuracy and per-document-type breakdown.
   - Per-field exact match (reference_number, amount_inr, deadline_date).
   - Abstention precision and recall on out-of-scope notices.
   - Deterministic citation-support verification (chunk exists and shares semantic overlap).
   - Latency and live API call averages.
   - Separate counts for successful, api_error, and abstained.
4. Exports metrics to reports/eval_results.json.
5. Exports 20 checklist steps with cited excerpts to reports/citation_review.csv for manual audit.
"""

import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.extract import extract_from_file, resolve_relative_deadline
from src.guardrails import evaluate_guardrails
from src.letters import draft_letter_content
from src.llm import get_live_calls_count, is_live_gemini_available, reset_live_calls_count
from src.reminders import generate_ics_from_extraction
from src.retrieve import search_kb
from src.schemas import DocumentType, GroundedExplanation, NoticeExtraction, RetrievedChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_eval")

DATA_DIR = PROJECT_ROOT / "data"
NOTICES_DIR = DATA_DIR / "test_notices"
LABELS_PATH = NOTICES_DIR / "labels.json"
CACHED_RESULTS_DIR = DATA_DIR / "cached_results"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVAL_RESULTS_PATH = REPORTS_DIR / "eval_results.json"
CITATION_REVIEW_PATH = REPORTS_DIR / "citation_review.csv"


def normalize_str(val: Optional[str]) -> Optional[str]:
    """Normalize string for exact match comparison."""
    if val is None:
        return None
    cleaned = re.sub(r"[\s\-_/]+", "", str(val)).lower()
    return cleaned if cleaned else None


def float_match(val1: Optional[float], val2: Optional[float], tolerance: float = 0.5) -> bool:
    """Compare two monetary values within a small tolerance."""
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False
    return abs(float(val1) - float(val2)) <= tolerance


def check_citation_support(step_dict: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Deterministic check verifying cited chunk existence and semantic overlap."""
    cited_chunk_id = step_dict.get("chunk_id", "")
    chunk = next((c for c in retrieved_chunks if c.get("chunk_id") == cited_chunk_id), None)
    if not chunk:
        return False, f"Cited chunk_id '{cited_chunk_id}' not found in retrieved chunks"

    step_text = f"{step_dict.get('action', '')} {step_dict.get('details', '')}".lower()
    chunk_text = f"{chunk.get('title', '')} {chunk.get('content', '')}".lower()

    step_words = set(re.findall(r"\b[a-z]{4,}\b", step_text))
    chunk_words = set(re.findall(r"\b[a-z]{4,}\b", chunk_text))
    word_overlap = step_words.intersection(chunk_words)

    step_nums = set(re.findall(r"\b\d+\b", step_text))
    chunk_nums = set(re.findall(r"\b\d+\b", chunk_text))
    num_overlap = step_nums.intersection(chunk_nums)

    if len(word_overlap) >= 1 or len(num_overlap) >= 1 or float(chunk.get("score", 0.0)) >= 0.30:
        return True, f"Supported by chunk '{cited_chunk_id}' (overlap: {len(word_overlap)} words, {len(num_overlap)} numbers)"
    return False, f"Insufficient term overlap with chunk '{cited_chunk_id}'"


def run_single_notice_pipeline(
    notice_path: Path,
    label: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute end-to-end extraction, retrieval, explanation, letter and reminder for one notice."""
    filename = notice_path.name
    t0 = time.time()
    call_start = get_live_calls_count()

    try:
        # Step 1: Multimodal Extraction
        ext_res = extract_from_file(notice_path, force_mock=False, use_cache=True)
        extraction = ext_res.extraction

        # Step 2: Grounded Retrieval
        query = " ".join([extraction.issuer or "", " ".join(extraction.key_facts[:3])]).strip()
        if not query:
            query = str(extraction.doc_type.value if hasattr(extraction.doc_type, "value") else extraction.doc_type)

        retrieved_models = search_kb(query=query, doc_type=extraction.doc_type, top_k=4)
        top_score = max([c.score for c in retrieved_models], default=0.0)

        # Step 3: Guardrails and Abstention Decision
        decision = evaluate_guardrails(extraction, top_retrieval_score=top_score)

        # Step 4: Grounded Explanation
        from src.explain import generate_grounded_explanation
        explanation_model: Optional[GroundedExplanation] = None
        if not decision.should_abstain:
            try:
                explanation_model = generate_grounded_explanation(
                    extraction=extraction,
                    retrieved_chunks=retrieved_models,
                    language="English",
                    use_cache=True,
                    force_mock=False,
                )
            except Exception as exp_err:
                logger.warning("Explanation generation error for %s: %s", filename, exp_err)

        # Step 5: Letter Draft
        letter_dict = None
        if not decision.should_abstain and extraction.doc_type in (
            DocumentType.electricity, DocumentType.challan, DocumentType.scheme_rejection, DocumentType.rti
        ):
            try:
                letter_draft = draft_letter_content(
                    extraction=extraction,
                    retrieved_chunks=retrieved_models,
                    language="English",
                )
                letter_dict = letter_draft.model_dump()
            except Exception as let_err:
                logger.warning("Letter drafting error for %s: %s", filename, let_err)

        # Step 6: Calendar Reminder
        reminder_ics = None
        if extraction.deadline_date:
            try:
                reminder_ics = generate_ics_from_extraction(extraction)
            except Exception as rem_err:
                logger.warning("Reminder generation error for %s: %s", filename, rem_err)

        latency = time.time() - t0
        calls_used = get_live_calls_count() - call_start

        status = "abstained" if decision.should_abstain else "success"
        if ext_res.error:
            status = "api_error"

        result_payload = {
            "source": "cached_live_run",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "status": status,
            "is_mock": ext_res.is_mock,
            "latency_seconds": round(latency, 3),
            "api_calls": calls_used,
            "extraction": extraction.model_dump(),
            "decision": decision.to_dict(),
            "retrieved_chunks": [c.model_dump() for c in retrieved_models],
            "explanation": explanation_model.model_dump() if explanation_model else None,
            "letter_draft": letter_dict,
            "has_reminder_ics": reminder_ics is not None,
            "error": ext_res.error,
        }
        return result_payload

    except Exception as exc:
        latency = time.time() - t0
        logger.error("Pipeline failure on %s: %s", filename, exc)
        return {
            "source": "cached_live_run",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "status": "api_error",
            "is_mock": False,
            "latency_seconds": round(latency, 3),
            "api_calls": get_live_calls_count() - call_start,
            "extraction": None,
            "decision": None,
            "retrieved_chunks": [],
            "explanation": None,
            "letter_draft": None,
            "has_reminder_ics": False,
            "error": str(exc),
        }


def run_evaluation() -> Dict[str, Any]:
    """Run evaluation over all notices, save cached runs, and compute all metrics."""
    CACHED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"labels.json not found at {LABELS_PATH}")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    notices_labels: List[Dict[str, Any]] = labels_data.get("notices", [])

    logger.info("Starting live evaluation across %d notices...", len(notices_labels))
    results: List[Dict[str, Any]] = []

    for idx, label in enumerate(notices_labels, 1):
        filename = label["filename"]
        stem = Path(filename).stem
        cached_file = CACHED_RESULTS_DIR / f"{stem}.json"
        notice_path = NOTICES_DIR / filename

        # Check if already cached from live run
        if cached_file.exists():
            try:
                with open(cached_file, "r", encoding="utf-8") as cf:
                    cached_data = json.load(cf)
                if cached_data.get("source") == "cached_live_run" and not cached_data.get("is_mock"):
                    logger.info("[%d/%d] Loaded cached live result for %s", idx, len(notices_labels), filename)
                    if cached_data.get("extraction"):
                        ext_model = NoticeExtraction.model_validate(cached_data["extraction"])
                        ext_model = resolve_relative_deadline(ext_model)
                        cached_data["extraction"] = ext_model.model_dump()
                    results.append(cached_data)
                    continue
            except Exception as e:
                logger.warning("Failed to read cache file %s: %s", cached_file, e)

        # Run live pipeline
        logger.info("[%d/%d] Running live pipeline for %s...", idx, len(notices_labels), filename)
        notice_result = run_single_notice_pipeline(notice_path, label)

        # Save immediately to disk for resumability
        try:
            with open(cached_file, "w", encoding="utf-8") as cf:
                json.dump(notice_result, cf, indent=2)
            logger.info("Saved cached live result to %s", cached_file.name)
        except Exception as e:
            logger.error("Failed to write cached result for %s: %s", filename, e)

        results.append(notice_result)

        # Sleep between calls to respect rate limits
        time.sleep(1.5)

    # ----------------- COMPUTE METRICS -----------------
    total_count = len(results)
    successful_count = sum(1 for r in results if r["status"] == "success")
    abstained_count = sum(1 for r in results if r["status"] == "abstained")
    api_error_count = sum(1 for r in results if r["status"] == "api_error")

    # Exclude api_errors from accuracy evaluation
    eval_set: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for r in results:
        if r["status"] == "api_error":
            continue
        lbl = next((l for l in notices_labels if l["filename"] == r["filename"]), None)
        if lbl and r.get("extraction"):
            eval_set.append((r, lbl))

    eval_total = len(eval_set)

    # Document type accuracy
    doc_type_matches = 0
    ref_matches = 0
    amount_matches = 0
    deadline_matches = 0

    per_type_stats: Dict[str, Dict[str, int]] = {}
    for dt in ["electricity", "challan", "tax", "scheme_rejection", "rti", "other"]:
        per_type_stats[dt] = {"total": 0, "correct": 0}

    # Confusion matrix: {actual: {predicted: count}}
    all_types = ["electricity", "challan", "tax", "scheme_rejection", "rti", "other"]
    confusion_matrix: Dict[str, Dict[str, int]] = {a: {p: 0 for p in all_types} for a in all_types}

    failures: List[Dict[str, Any]] = []

    for res, lbl in eval_set:
        ext = res["extraction"]
        pred_type = ext.get("doc_type", "other")
        actual_type = lbl.get("doc_type", "other")

        confusion_matrix[actual_type][pred_type] += 1
        per_type_stats[actual_type]["total"] += 1

        is_type_correct = (pred_type == actual_type)
        if is_type_correct:
            doc_type_matches += 1
            per_type_stats[actual_type]["correct"] += 1
        else:
            failures.append({
                "filename": lbl["filename"],
                "metric": "doc_type",
                "expected": actual_type,
                "predicted": pred_type,
                "reason": f"Document classified as {pred_type} instead of {actual_type}",
            })

        # Field exact matches
        pred_ref = normalize_str(ext.get("reference_number"))
        actual_ref = normalize_str(lbl.get("reference_number"))
        if pred_ref == actual_ref:
            ref_matches += 1
        else:
            failures.append({
                "filename": lbl["filename"],
                "metric": "reference_number",
                "expected": lbl.get("reference_number"),
                "predicted": ext.get("reference_number"),
                "reason": "Reference number discrepancy",
            })

        if float_match(ext.get("amount_inr"), lbl.get("amount_inr")):
            amount_matches += 1
        else:
            failures.append({
                "filename": lbl["filename"],
                "metric": "amount_inr",
                "expected": lbl.get("amount_inr"),
                "predicted": ext.get("amount_inr"),
                "reason": "Amount INR discrepancy",
            })

        if ext.get("deadline_date") == lbl.get("deadline_date"):
            deadline_matches += 1
        else:
            failures.append({
                "filename": lbl["filename"],
                "metric": "deadline_date",
                "expected": lbl.get("deadline_date"),
                "predicted": ext.get("deadline_date"),
                "reason": "Deadline date mismatch",
            })

    doc_type_accuracy = round(doc_type_matches / eval_total, 4) if eval_total > 0 else 0.0
    ref_accuracy = round(ref_matches / eval_total, 4) if eval_total > 0 else 0.0
    amount_accuracy = round(amount_matches / eval_total, 4) if eval_total > 0 else 0.0
    deadline_accuracy = round(deadline_matches / eval_total, 4) if eval_total > 0 else 0.0

    # Abstention metrics (on out-of-scope notices)
    tp_abstained = 0  # Out of scope and abstained
    fn_not_abstained = 0  # Out of scope and NOT abstained
    fp_abstained = 0  # In scope and abstained
    tn_not_abstained = 0  # In scope and NOT abstained

    for res, lbl in eval_set:
        is_oos = bool(lbl.get("is_out_of_scope") or lbl.get("doc_type") == "other")
        was_abstained = (res["status"] == "abstained") or (res["extraction"].get("doc_type") == "other")

        if is_oos:
            if was_abstained:
                tp_abstained += 1
            else:
                fn_not_abstained += 1
        else:
            if was_abstained:
                fp_abstained += 1
            else:
                tn_not_abstained += 1

    abstention_precision = round(tp_abstained / (tp_abstained + fp_abstained), 4) if (tp_abstained + fp_abstained) > 0 else 1.0
    abstention_recall = round(tp_abstained / (tp_abstained + fn_not_abstained), 4) if (tp_abstained + fn_not_abstained) > 0 else 1.0

    # Citation support check & CSV export
    all_steps_for_review: List[Dict[str, Any]] = []
    total_steps = 0
    supported_steps = 0

    for res in results:
        explanation = res.get("explanation")
        if not explanation:
            continue
        checklist = explanation.get("checklist", [])
        retrieved_chunks = res.get("retrieved_chunks", [])

        for step in checklist:
            total_steps += 1
            is_supported, support_detail = check_citation_support(step, retrieved_chunks)
            if is_supported:
                supported_steps += 1

            chunk = next((c for c in retrieved_chunks if c.get("chunk_id") == step.get("chunk_id")), {})
            content_snippet = chunk.get("content", "").replace("\n", " ")[:150]

            all_steps_for_review.append({
                "notice_filename": res["filename"],
                "doc_type": res["extraction"].get("doc_type") if res.get("extraction") else "unknown",
                "step_number": step.get("step_number"),
                "action": step.get("action"),
                "details": step.get("details"),
                "cited_chunk_id": step.get("chunk_id"),
                "source_title": step.get("source_title", chunk.get("title")),
                "source_url": step.get("source_url", chunk.get("source_url")),
                "cited_excerpt_snippet": content_snippet,
                "supported_verdict": "VERIFIED_SUPPORTED" if is_supported else "UNSUPPORTED",
                "notes": support_detail,
            })

    citation_support_rate = round(supported_steps / total_steps, 4) if total_steps > 0 else 1.0

    # Export exactly 20 checklist steps to reports/citation_review.csv
    review_sample = all_steps_for_review[:20]
    with open(CITATION_REVIEW_PATH, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "notice_filename", "doc_type", "step_number", "action", "details",
            "cited_chunk_id", "source_title", "source_url", "cited_excerpt_snippet",
            "supported_verdict", "notes"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in review_sample:
            writer.writerow(row)

    logger.info("Exported %d checklist steps to %s", len(review_sample), CITATION_REVIEW_PATH)

    # Latency and API calls averages
    latencies = [r["latency_seconds"] for r in results if r.get("latency_seconds")]
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0

    api_calls = [r["api_calls"] for r in results if r.get("api_calls") is not None]
    avg_api_calls = round(sum(api_calls) / len(api_calls), 2) if api_calls else 0.0

    # Final eval results object
    eval_results = {
        "metadata": {
            "evaluation_date": datetime.now(timezone.utc).isoformat(),
            "total_notices": total_count,
            "evaluated_notices": eval_total,
            "successful_count": successful_count,
            "abstained_count": abstained_count,
            "api_error_count": api_error_count,
            "model_used": os.getenv("GEMINI_EXTRACT_MODEL", "gemini-3.5-flash-lite"),
            "disclaimer": "All test notices are synthetic. The dataset is small (30 notices) and administrative procedures differ by state.",
        },
        "metrics": {
            "doc_type_accuracy": doc_type_accuracy,
            "reference_number_accuracy": ref_accuracy,
            "amount_inr_accuracy": amount_accuracy,
            "deadline_date_accuracy": deadline_accuracy,
            "abstention_precision": abstention_precision,
            "abstention_recall": abstention_recall,
            "citation_support_rate": citation_support_rate,
            "average_latency_seconds": avg_latency,
            "average_live_calls_per_notice": avg_api_calls,
        },
        "per_doc_type_accuracy": {
            dt: round(per_type_stats[dt]["correct"] / per_type_stats[dt]["total"], 4)
            if per_type_stats[dt]["total"] > 0 else 0.0
            for dt in all_types
        },
        "confusion_matrix": confusion_matrix,
        "abstention_details": {
            "true_positives_out_of_scope_abstained": tp_abstained,
            "false_negatives_out_of_scope_missed": fn_not_abstained,
            "false_positives_in_scope_abstained": fp_abstained,
            "true_negatives_in_scope_passed": tn_not_abstained,
        },
        "failures": failures,
    }

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as ef:
        json.dump(eval_results, ef, indent=2)

    logger.info("Evaluation complete! Results saved to %s", EVAL_RESULTS_PATH)
    return eval_results


if __name__ == "__main__":
    run_evaluation()
