# NoticeWise — Citizen Notice Simplification & Action Assistant

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35.0+-FF4B4B.svg)](https://streamlit.io/)
[![Google GenAI](https://img.shields.io/badge/google--genai-0.1.1+-4285F4.svg)](https://pypi.org/project/google-genai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **NoticeWise** transforms intimidating Indian administrative and government notices (electricity disconnection, traffic citations, welfare scheme rejections, and RTI inquiries) into clear plain-language explanations, cited action plans, formal reply letters, and deadline reminders.

---

## 1. Problem Statement

Every day, ordinary citizens receive intimidating bureaucratic notices from municipal bodies, state electricity distribution companies (DISCOMs), traffic police branches, and welfare departments. These documents often feature:
- Densely nested legal clauses and references to obscure statutory sections.
- Vague or harsh warnings regarding electricity disconnection or compounding penalties.
- Strict 7, 15, or 30-day statutory limitation deadlines.
- Little to no actionable guidance on which administrative office to visit or what documents to bring.

Lacking legal counsel, citizens frequently panic, miss critical appeal windows, or fall victim to unauthorized intermediaries. **NoticeWise** solves this by providing immediate, grounded, multilingual civic paperwork assistance with strict safety guardrails.

---

## 2. Architecture & How It Works

NoticeWise implements a multi-stage, grounded pipeline engineered to prioritize factual grounding and citizen safety:

```
[ Citizen Notice (PDF / Image / Sample) ]
                   │
                   ▼
  1. Multimodal Structured Extraction (Gemini 3.5 Flash Lite + Pydantic)
     - Fast-path selectable-text PDF extraction (zero vision tokens)
     - Vision compression & normalization for images (<=1500px, JPEG quality 85)
     - Resilient single-retry with JSON validation feedback
                   │
                   ▼
  2. Grounded TF-IDF Retrieval (Official Statutory Knowledge Base)
     - Queries 14 authoritative statutory chunks across Central & State Acts
     - Deterministic scoring: thresholds check semantic match (min score >= 0.20)
                   │
                   ▼
  3. Guardrails & Safe Abstention Engine (F5 Safety Filter)
     - Refuses out-of-scope documents (leases, court summons, invoices)
     - Refuses in-scope documents lacking verified statutory KB backing
     - Directs citizen to local physical office with required document checklist
                   │
                   ▼
  4. Grounded Action Generation (Plain Language, Citations & Paperwork)
     - Plain-Language Explanation (English, Telugu, Hindi)
     - Step-by-Step Action Checklist with verified official deep-link citations
     - Multilingual Formal Reply Letter & print-ready PDF (ReportLab)
     - RFC 5545 iCalendar (.ics) Deadline Reminder with 3-day advance alerts
```

### Zero-API-Key Out-of-the-Box Mode
To ensure immediate accessibility and zero friction during evaluation or deployment:
- All **30 pre-loaded synthetic notices** in the Sample Notice Picker load verified live results directly from `data/cached_results/`.
- Citizens and judges can test multimodal intake, plain explanations, verified citations, PDF letter drafting, and calendar reminders with **zero API calls** and **no API key required**.
- Live document uploads prompt the user when `GEMINI_API_KEY` is not configured.

---

## 3. Statutory Knowledge Base (Telangana & Central Sources)

NoticeWise enforces a strict statutory integrity standard: **no procedure, deadline, fee, or penalty is generated from memory**. Every entry in the knowledge base is taken verbatim from official government portals, gazetted Acts, or regulatory PDFs accessed on **19 November 2026**.

### Surviving Verified Sources (6 Documents, 14 Chunks in `kb/`):
| File | Topic | Official Deep-Link Source | Verbatim Sections Covered |
| :--- | :--- | :--- | :--- |
| `01_rti_application_procedure_sec6.md` | RTI Application Procedure | [cic.gov.in/sites/default/files/RTI-Act_English.pdf](https://cic.gov.in/sites/default/files/RTI-Act_English.pdf) | Sections 6(1), 6(2), 6(3), 7(1) |
| `02_rti_first_appeal_rules_sec19.md` | RTI First & Second Appeals | [cic.gov.in/sites/default/files/RTI-Act_English.pdf](https://cic.gov.in/sites/default/files/RTI-Act_English.pdf) | Section 19(1), 19(2), 19(6) |
| `03_rti_fee_and_cost_rules.md` | RTI Fees, Costs & Appeals | [pmindia.gov.in/wp-content/uploads/2017/04/RTIRules_2012_English_0.pdf](https://www.pmindia.gov.in/wp-content/uploads/2017/04/RTIRules_2012_English_0.pdf) | Rules 3, 4, 5, 6, 8 |
| `05_telangana_echallan_compounding_and_dispute.md` | Motor Vehicle Offence Compounding | [indiacode.nic.in/bitstream/123456789/9460/1/a1988-59.pdf](https://www.indiacode.nic.in/bitstream/123456789/9460/1/a1988-59.pdf) | Section 200 of MV Act, 1988 |
| `08_electricity_disconnection_notice_rules.md` | Electricity Disconnection Rules | [derc.gov.in/sites/default/files/SupplyCode.pdf](https://www.derc.gov.in/sites/default/files/SupplyCode.pdf) | Regulation 49 & Sec 56 Electricity Act |
| `11_nfsa_ration_card_grievance_redressal.md` | NFSA Ration Card Grievance Redressal | [dfpd.gov.in/WriteReadData/Other/nfsa_1.pdf](https://dfpd.gov.in/WriteReadData/Other/nfsa_1.pdf) | Sections 14, 15, 16 of NFSA, 2013 |

### Deleted Sources (Marked as MISSING):
The following 6 files were removed following the Phase 2 audit because their procedures could not be verified verbatim on official public portals without authentication:
1. `04_echallan_contest_and_grievance_procedure.md` (**MISSING**): Parivahan FAQ portal does not publish full statutory dispute guidelines.
2. `06_tsspdcl_electricity_complaint_and_billing_dispute.md` (**MISSING**): TSSPDCL portal (`tgsouthernpower.org`) suffered SSL handshake failures (`SSLEOFError`) and lacked verbatim Regulation 14 text.
3. `07_tserc_cgrf_and_ombudsman_regulations.md` (**MISSING**): TSERC portal (`tserc.gov.in`) timed out with SSL alert errors; regulations could not be fetched verbatim.
4. `09_ghmc_property_tax_revision_petition.md` (**MISSING**): GHMC portal (`ghmc.gov.in`) does not publish verbatim text of Section 282.
5. `10_ghmc_property_tax_appeal_sec286.md` (**MISSING**): GHMC portal does not publish verbatim text of Section 286.
6. `12_telangana_aasara_pension_rejection_appeal.md` (**MISSING**): Aasara portal (`aasara.telangana.gov.in`) is gated behind employee authentication.

> **NoticeWise Safe Handling:** Because GHMC property tax and Aasara pension rules are marked MISSING, NoticeWise safely abstains when processing these notices (top retrieval score = 0.0), instructing the citizen to visit their local municipal or Tahsildar office.

---

## 4. Evaluation Benchmark Results

The pipeline was evaluated live against all 30 synthetic notices in `data/test_notices/` using `scripts/run_eval.py`. Results are recorded in `reports/eval_results.json`.

### Key Metrics Summary (30 Synthetic Notices)
| Metric | Value | Description |
| :--- | :---: | :--- |
| **Total Test Notices** | `30` | Complete synthetic benchmark suite |
| **Evaluated Notices** | `30` (100%) | All notices evaluated live without quota crashes |
| **API Error Count** | `0` (0%) | Zero API dropouts on active model (`gemini-3.5-flash-lite`) |
| **Document Type Accuracy** | **`1.0000` (100%)** | 30 / 30 notices correctly classified into schema types |
| **Reference Number Accuracy** | **`1.0000` (100%)** | 30 / 30 notice reference numbers matched ground truth |
| **Amount INR Accuracy** | **`0.7333` (73.3%)** | 22 / 30 exact monetary demands identified |
| **Deadline Date Accuracy** | **`0.6667` (66.7%)** | 20 / 30 statutory dates matched ISO ground truth |
| **Abstention Recall (Safety)** | **`1.0000` (100%)** | 5 / 5 true out-of-scope documents safely refused |
| **Abstention Precision** | **`0.4167` (41.7%)** | 5 true OOS / 12 total abstained (conservative safety policy) |
| **Citation Support Rate** | **`1.0000` (100%)** | 100% of checklist steps verified against official KB |
| **Average Latency** | `6.561s` | End-to-end multimodal extraction, retrieval, and drafting |
| **Average Live Calls / Notice** | `1.50` | Fused extraction + explanation calls (cached retrieval) |

### Accuracy Breakdown by Document Type
| Category | Evaluated | Accuracy | Pipeline Status |
| :--- | :---: | :---: | :--- |
| **Electricity DISCOM Notices** | 5 | 100.0% | Supported (DERC Supply Code Reg 49 / Electricity Act Sec 56) |
| **Traffic e-Challan Citations** | 5 | 100.0% | Supported (Motor Vehicles Act Sec 200) |
| **Municipal Property Tax Demands** | 5 | 100.0% | Safely Abstained (GHMC portal unverified) |
| **Welfare Scheme Rejections** | 5 | 100.0% | Supported for NFSA Ration Cards (Sec 14-16) |
| **Right to Information (RTI)** | 5 | 100.0% | Supported (RTI Act 2005 Sec 6 & RTI Rules 2012) |
| **Out-of-Scope (Private/Court)** | 5 | 100.0% | Safely Abstained (F5 Safety Card) |

### Confusion Matrix (6x6)
```
Actual \ Predicted   ELEC   CHALLAN    TAX   SCHEME    RTI   OTHER
--------------------------------------------------------------
electricity             5         0      0        0      0       0
challan                 0         5      0        0      0       0
tax                     0         0      5        0      0       0
scheme_rejection        0         0      0        5      0       0
rti                     0         0      0        0      5       0
other                   0         0      0        0      0       5
```

### Real Failure Analysis (Discrepancies Observed)
1. **`notice_02_elec_meter_tamper.pdf` (Amount INR)**: Expected ₹12,800.00; Predicted `None`.
   - *Root Cause:* The document was a narrative vigilance inspection report where the compound assessment was in an unstructured clause rather than a tabular billing summary.
2. **`notice_02_elec_meter_tamper.pdf` (Deadline Date)**: Expected `2026-11-10`; Predicted `None`.
   - *Root Cause:* The notice used the relative phrase "within 7 days of service" rather than an explicit ISO calendar date.
3. **`notice_04_elec_tariff_revision.pdf` (Amount INR)**: Expected ₹1,850.00; Predicted `None`.
   - *Root Cause:* Regulatory tariff revision circular where the PPAC amount was an illustrative formula example rather than an individual bill demand.
4. **`notice_04_elec_tariff_revision.pdf` (Deadline Date)**: Expected `2026-11-25`; Predicted `None`.
   - *Root Cause:* Advisory communication notice without an imminent service cutoff date.
5. **`notice_07_challan_signal.pdf` (Amount INR)**: Expected ₹1,000.00; Predicted `None`.
   - *Root Cause:* PDF notice cited statutory compounding schedules rather than an explicit bold demand total block.

---

## 5. Limitations & Safe Use

1. **Synthetic Benchmark Dataset:** The 30 evaluation notices in `data/test_notices/` are synthetically generated artifacts designed to test OCR robustness, blurred scans, and classification boundaries.
2. **Small Dataset Size:** 30 notices provide representative coverage of the 6 target categories, but real-world document variations (regional dialects, stamps, handwriting) require broader field testing.
3. **Jurisdictional & State Variability:** Administrative rules, fee schedules, and appellate forums vary across Indian states. What holds true under Central Acts (RTI, Motor Vehicles, Electricity) may differ in state-specific municipal codes.
4. **Not Legal Advice:** NoticeWise is an automated civic empowerment tool. It does not provide legal representation or create an attorney-client relationship. Citizens are always directed to verify findings with the issuing department.

---

## 6. Local Setup & Execution

### Prerequisites
- Python 3.11+
- Git

### Installation
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/noticewise.git
cd noticewise

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration (Optional for Sample Testing)
To enable live multimodal vision extraction on custom uploaded notices, create a `.env` file in the root directory:
```env
GEMINI_API_KEY=AIzaSy...your-actual-gemini-api-key
GEMINI_EXTRACT_MODEL=gemini-3.5-flash-lite
GEMINI_EXPLAIN_MODEL=gemini-3.5-flash-lite
```
*(NoticeWise runs completely offline without this key if you use the pre-loaded synthetic notices).*

### Run Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### Run Test Suite
```bash
pytest -v
```

---

## 7. Streamlit Community Cloud Deployment Guide

NoticeWise is designed to deploy seamlessly to **Streamlit Community Cloud (Python 3.11)** with zero configuration overhead:

1. **Push to GitHub:**
   Ensure all changes are committed and pushed to your GitHub repository:
   ```bash
   git push origin master
   ```

2. **Sign in to Streamlit Community Cloud:**
   Go to [share.streamlit.io](https://share.streamlit.io/) and log in using your GitHub account.

3. **Create New App:**
   - Click **"New app"**.
   - Select your **Repository**: `<your-username>/noticewise`.
   - Select **Branch**: `master`.
   - Set **Main file path**: `app/streamlit_app.py`.
   - Set **App URL**: `noticewise` (or custom slug).

4. **Configure Secrets (Optional for Live Uploads):**
   - Click **"Advanced settings..."** before deploying.
   - Under **Secrets**, add your Gemini API key:
     ```toml
     GEMINI_API_KEY = "AIzaSy...your-gemini-key"
     ```
   - *Note:* If you omit this secret, the app still launches and runs cleanly. All 30 synthetic notices will function with pre-computed live run caches, and uploads will show a polite guidance message.

5. **Deploy:**
   - Click **"Deploy!"**.
   - Streamlit will automatically build dependencies from `requirements.txt` and launch the application.
