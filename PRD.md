# NoticeWise: PRD and Implementation Plan

Solo hackathon entry (NxtWave), Track 01: Civic Pulse, Digital Governance & Public Information. Submission: GitHub repo + deployed link. Deadline: Fri 20 Nov 2026, 11:59 pm. Target: submit by the evening of 20 Nov.

## 1. One-line pitch
NoticeWise turns the confusing government notice a citizen is holding into done paperwork: it reads the document, explains it in Telugu, Hindi or English, cites the official procedure for every step, and produces a ready-to-send reply letter and a deadline reminder. When it is not sure, it says so and sends the user to the right office.

## 2. Problem
Citizens receive electricity bill disputes, traffic e-challans, tax notices and scheme rejections written in dense jargon. Most people cannot tell what the notice means, what happens if they ignore it, or what to do next. They miss deadlines, pay wrongly, or give up on legitimate claims. Existing tools either explain schemes in general (chatbots) or fill forms for you, but none start from the actual document and carry the person through to a finished, sourced action.

## 3. Users
- **Citizen with low legal or bureaucratic literacy:** wants to know what this is, how urgent it is, and what to do, in their own language.
- **Family member or helper (e.g. a student helping parents):** needs a checklist and a letter they can send.
- **Judge or evaluator:** needs to see grounded answers, measured accuracy, and safe behaviour.

## 4. Goals, non-goals and success metrics
### Goals
1. End-to-end flow from photo or PDF to explanation, checklist, drafted letter (PDF) and reminder file (.ics).
2. Every procedural claim is backed by a cited source from a curated knowledge base.
3. The system abstains when confidence or evidence is low.
4. Measured quality on a labelled test set.

### Non-goals
- Legal advice, or anything that submits forms on government portals.
- Covering every notice type. Five types only.
- Training any model.
- Live scraping of government websites at runtime.

### Success metrics (report real numbers from real runs)
- Field extraction accuracy (document type, amount, deadline, reference number) on the test set.
- Deadline correctness rate.
- Citation correctness: share of cited sources that actually support the step.
- Abstention behaviour: share of out-of-scope or unreadable documents correctly refused.
- Telugu and Hindi output reviewed on a small sample.

## 5. Scope: five document types
1. Electricity bill dispute or disconnection notice (DISCOM).
2. Traffic e-challan.
3. Municipal or property tax notice.
4. Scheme application rejection letter (for example a ration card or pension scheme).
5. Generic RTI help: the user describes what they want to know and the app drafts an RTI application.

Anything else is classified `other` and handled with a safe abstention message.

## 6. User flow
1. User opens the app and picks a language (English, Telugu, Hindi).
2. User uploads a photo or PDF of the notice (or picks a sample notice).
3. App shows extracted fields for the user to confirm or correct.
4. App shows a plain-language explanation: what this is, what happens if ignored, how long they have, the amount involved.
5. App shows a step-by-step action checklist with a citation next to each step, plus the documents needed.
6. User clicks to generate a reply or complaint letter. App creates an editable draft, then a downloadable PDF.
7. User downloads a calendar reminder (.ics) for the deadline.
8. If confidence is low or no source is found, the app shows an abstention card: "I am not sure about this. Please visit or call your local office", with what to bring.

## 7. Features and acceptance criteria
- **F1. Document intake.** Upload JPG, PNG or PDF. Show a preview. Reject files over 10 MB. Acceptance: a blurry or non-document image gives a clear error, not a crash.
- **F2. Structured extraction.** Multimodal LLM returns strict JSON:
```json
{
  "doc_type": "electricity|challan|tax|scheme_rejection|rti|other",
  "issuer": string|null,
  "reference_number": string|null,
  "amount_inr": number|null,
  "issue_date": "YYYY-MM-DD"|null,
  "deadline_date": "YYYY-MM-DD"|null,
  "key_facts": [string],
  "language_of_document": string,
  "confidence": 0-1,
  "missing_or_unclear": [string]
}
```
Acceptance: output is validated with Pydantic. Invalid JSON triggers one retry, then falls back to a manual-entry form. Fields shown to the user are editable.
- **F3. Plain-language explanation.** Generated only from the confirmed fields plus retrieved sources. Outputs: what it is, what happens if ignored, days left (computed in code, not by the LLM), amount. Telugu, Hindi and English. Acceptance: the days-left number is always computed by Python from the deadline date.
- **F4. Grounded retrieval and citations.** A knowledge base of 10 to 15 official documents, split into chunks with source URL and title. Retrieval uses embeddings (Gemini embeddings or sentence-transformers) with a keyword fallback. The LLM must answer using only retrieved chunks and cite chunk IDs. Acceptance: each checklist step shows its source title and URL. A step with no supporting chunk is dropped, not invented.
- **F5. Abstention and safety.** Abstain when `doc_type` is `other`, extraction confidence is below 0.6, or the top retrieval score is below a threshold. Always show a "not legal advice, verify with the issuing office" note. Acceptance: a set of 5 or more out-of-scope or unreadable test documents all trigger abstention.
- **F6. Letter generator.** Template-driven by doc type, filled with confirmed fields and user inputs (name, address, grounds for dispute). The LLM writes only the "grounds" paragraph, based on retrieved sources. Output: editable text, then PDF (ReportLab or WeasyPrint). Acceptance: PDF opens correctly with all fields filled, and the placeholders the user has not filled are clearly marked.
- **F7. Deadline reminder.** Generate an .ics file with the deadline and a reminder 3 days before. Acceptance: the file imports into Google Calendar.
- **F8. Evaluation page.** A page in the app showing the test-set metrics, a confusion matrix for document type, and a few sample failures with an honest explanation. Acceptance: numbers come from `reports/eval_results.json`, which is generated by a script.

## 8. Data
- **Knowledge base (`kb/`):** 10 to 15 official pages or PDFs that you collect yourself and save as .md or .txt files, each starting with a header line containing the title, URL and date accessed. Suggested coverage: consumer complaint steps for electricity, how to contest an e-challan, how to appeal or dispute a property tax assessment, how to file an RTI application and its format, and how to appeal a scheme rejection. Use official government or DISCOM sites. Do not paste text you are unsure is official.
- **Synthetic test notices (`data/test_notices/`):** 25 to 30 notices you create. Generate them with a script or an LLM using realistic templates (fake names, fake reference numbers), render them as images and PDFs, and store the ground truth in `data/test_notices/labels.json`. Include: 5 per supported type, a few with a degraded scan look (blur, rotation), and 5 or more out-of-scope or unreadable ones. Label them as synthetic in the README and on the evaluation page.
- **Sample notices for the demo:** 5 of the synthetic notices, selectable in the UI.
- Never use real people's documents in the repo or in the deployed app.

## 9. AI design
- **Model:** Gemini (multimodal) via the `google-genai` SDK. Use structured output (JSON schema) for extraction. Keep temperature low (0 to 0.2).
- **Pipeline:** extract (F2), then validate and confirm, then retrieve (F4), then generate explanation and checklist (F3) with the retrieved chunks as the only allowed source, then a verification step.
- **Verification step:** a second LLM call or rule check that, for each checklist step, confirms that the cited chunk supports it. Remove unsupported steps.
- **Deterministic code does:** date arithmetic, amount formatting, template filling, file generation.
- **LLM does:** reading, translation, simplification, drafting the grounds paragraph.
- **Prompt rules:** do not give legal opinions, do not predict outcomes, do not invent section numbers, penalties or deadlines that are not in the sources, and say "not found in sources" when unsure.
- **Languages:** produce Telugu and Hindi directly from the verified English content to keep the facts identical across languages.

## 10. Architecture and repo layout
```text
noticewise/
  README.md
  PRD.md
  requirements.txt
  .env.example
  kb/                      # official docs as .md with source header
  data/test_notices/       # synthetic notices + labels.json
  src/
    schemas.py             # Pydantic models
    extract.py             # F2
    retrieve.py            # F4 (index build + search)
    explain.py             # F3, checklist, verification
    letters.py             # F6 templates + PDF
    reminders.py           # F7 .ics
    guardrails.py          # F5
    llm.py                 # Gemini client wrapper, retries
  scripts/
    build_index.py
    make_synthetic_notices.py
    run_eval.py
  app/streamlit_app.py
  reports/eval_results.json
  tests/                   # unit tests for dates, schemas, guardrails
```

## 11. Tech stack
Python 3.11, Streamlit, google-genai (Gemini), Pydantic, numpy, scikit-learn or sentence-transformers (embeddings and similarity), PyMuPDF (PDF to image), Pillow, ReportLab, ics, pytest. Deploy on Streamlit Community Cloud. Store the API key in Streamlit secrets and in a local .env (never committed).

## 12. Evaluation plan
1. Run `scripts/run_eval.py` over all test notices.
2. Compute: document type accuracy, per-field exact-match accuracy, deadline accuracy, abstention precision and recall on out-of-scope notices, citation support rate (manual check of 20 steps), and average latency.
3. Save results to `reports/eval_results.json` and show them on the Evaluation page.
4. Manually review 5 Telugu and 5 Hindi outputs and note any mistakes honestly.
5. Report limitations: synthetic test data, small test set, knowledge base covers selected procedures only, procedures differ by state and may change.

## 13. Implementation plan
### Wed 18 (today): foundation
- Set up the repo, Streamlit skeleton, Gemini client, and Pydantic schemas.
- Collect the 10 to 15 knowledge-base documents.
- Agent builds `make_synthetic_notices.py` and generates the test set with labels.
- Build and test extraction (F2) on 5 notices.
- Night checkpoint: extraction works end to end in the UI.

### Thu 19: the intelligence and the outputs
- Morning: retrieval index, grounded explanation and checklist with citations (F3, F4), verification step.
- Midday: guardrails and abstention (F5), language output.
- Afternoon: letter PDF (F6), reminder file (F7).
- Evening: first deployment to Streamlit Community Cloud and a smoke test of the full flow on the live link.

### Fri 20: prove it and ship
- Morning: run the evaluation (F8), fix the worst failures, add the Evaluation page.
- Noon: feature freeze.
- Afternoon: README, demo video of 2 to 3 minutes, final deployment test on fresh browser/phone.
- Evening: submit GitHub repo link and deployed link.
