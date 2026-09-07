# HingGuard — Project Briefing for AI Agents & Collaborators

> Read this file first before touching any code or data in this repository.
> It tells you what this project is, what has been done, what is left, and how everything fits together.

---

## 1. What Is This Project?

**HingGuard** is an NLP research project focused on **PII (Personally Identifiable Information) detection and anonymization in Hindi-English code-mixed (Hinglish) text**.

Code-mixed text — where a speaker switches between Hindi and English mid-sentence — is extremely common in South Asian social media (Twitter, WhatsApp, Reddit). Existing PII detection tools (like Microsoft Presidio) are designed for clean, monolingual English text. HingGuard investigates how well these tools perform on Hinglish data, and whether they can be adapted or improved.

**Research question in one sentence:**
> How accurately can existing PII detection systems identify and anonymize personal information in Hindi-English code-mixed social media text?

---

## 2. Team Structure & Division of Work

| Week | Task | Owner |
|------|------|-------|
| Week 1 | Data collection & preprocessing | [Team member who ran the notebook] |
| Week 2 | Exploratory Data Analysis (EDA) | Next team member |
| Week 3+ | PII injection & anonymization testing | TBD |
| Week 3+ | Model fine-tuning / adaptation | TBD |

---

## 3. Project Architecture — What Exists So Far

### 3.1 Data Sources

| Source | Status | Format | Size | Notes |
|--------|--------|--------|------|-------|
| **L3Cube HingLID** | Collected | CoNLL (.txt) | ~32,000 sentences | Word-level HI/EN tags; pre-split |
| **LinCE** | Collected | Hugging Face dataset | Varies | LID + POS configs; pre-split |
| **GLUECoS** | Skipped | — | — | Only has Tweet IDs, not text; needs Twitter API |
| **Full HingCorpus** | Skipped | Google Drive (single file) | 53M sentences | No tags; too large for Colab free tier |

### 3.2 Notebook

**File:** HingGuard_Week1_Data_Preprocessing.ipynb

**What it does (in order):**
1. Installs dependencies (datasets, pandas, pyarrow, matplotlib, scikit-learn)
2. Downloads and parses L3Cube HingLID from GitHub (CoNLL format)
3. Loads LinCE from Hugging Face (with graceful fallback if it fails)
4. Attempts GLUECoS — skips cleanly if no Twitter API credentials found
5. Combines all sources into one unified pandas DataFrame
6. Cleans text: removes URLs, @mentions, duplicates, junk-length sentences
7. Detects script type per sentence: roman_only, devanagari_only, or mixed_script
8. Splits into train/validation/test (80/10/10), respecting pre-existing splits
9. Saves output as .parquet and .csv
10. Generates a summary table and a stacked bar chart (script distribution)

### 3.3 Output Files (produced by the notebook)

| File | Format | Description |
|------|--------|-------------|
| hinglish_dataset.parquet | Parquet | Main dataset — fast-loading for Python/pandas |
| hinglish_dataset.csv | CSV | Same dataset — human-readable, Excel-compatible |
| dataset_summary.csv | CSV | Per-source, per-split sentence counts and avg lengths |
| script_distribution.png | PNG | Bar chart: Roman vs Devanagari vs Mixed per source |

---

## 4. Data Schema

Every row in hinglish_dataset.parquet / .csv has these columns:

| Column | Type | Description |
|--------|------|-------------|
| sentence_id | int | Unique row ID (0, 1, 2, …) |
| source | str | Origin: "hinglid" or "lince" |
| split | str | "train", "validation", or "test" |
| text | str | Full cleaned sentence as a string |
| tokens | list[str] | Sentence split into individual words |
| lid_tags | list[str] | Per-word language tag: "HI" or "EN" (where available) |
| script_type | str | "roman_only", "devanagari_only", or "mixed_script" |

---

## 5. Key Design Decisions (Summary)

Full details are in decision.md. Here is a quick reference:

| ID | Decision | Rationale |
|----|----------|-----------|
| D-01 | Use HingLID (~32K sentences) not full HingCorpus (53M) | Tagged, pre-split, no blockers |
| D-02 | Use LinCE via HF datasets | Plug-and-play; second independent benchmark |
| D-03 | Skip GLUECoS entirely | No actual text — only Tweet IDs requiring paid API |
| D-04 | Remove URLs | No linguistic value; confuses models |
| D-05 | Remove @mentions | Real PII; would corrupt fake-PII evaluation metrics |
| D-06 | Remove duplicates | Prevents data leakage across splits |
| D-07 | Filter <3 or >100 token sentences | Removes junk/spam entries |
| D-08 | Script detection via Unicode block ranges | Fast, interpretable, library-free |
| D-09 | Keep original pre-existing splits | Preserves benchmark comparability |
| D-10 | Stratified split for unlabeled data | Ensures proportional script/source distribution |
| D-11 | Save as Parquet + CSV | Speed (parquet) + human readability (csv) |
| D-12 | Unified DataFrame from the start | Write-once pipeline; source always traceable |

---

## 6. Known Limitations & Open Issues

### L1 | Script detection cannot distinguish Hinglish-Roman from English-Roman
- Both "I am going home" and "Main ghar ja raha hoon" are 100% Latin characters.
- Resolving this requires word-level LID tags (already in the data) or a language-detection tool.
- **Action item for EDA week:** Use existing lid_tags to build a better script-language joint classifier.

### L2 | GLUECoS data is unavailable
- If Twitter API access is obtained, the pipeline is ready (twitter_authentication.txt check already in notebook).
- This should be cited as a dataset limitation in the paper.

### L3 | Full HingCorpus not used
- 53M sentences available but untagged and unwieldy.
- Could be used for unsupervised pre-training later. Manual download instructions are in the notebook.

### L4 | No PII has been injected yet
- Week 1 produces only the clean, real-text dataset.
- Fake PII injection (names, phone numbers, email addresses, etc.) is a Week 2/3 task.

---

## 7. What the Next Agent / Team Member Should Do

### If you are doing Week 2 EDA:
- Load hinglish_dataset.parquet into pandas.
- Analyse: sentence length distribution, token count distribution, HI vs EN word ratio, script type breakdown.
- Investigate the L1 limitation above — can lid_tags help us better classify sentences?
- Produce visualizations suitable for the MSE 1 report.

### If you are doing PII Injection:
- Decide what types of PII to inject: names, emails, phone numbers, Aadhaar numbers, UPI IDs, addresses.
- Design injection so it respects the language context (Hindi names in Hindi sentences, etc.).
- Keep a map of original vs injected sentence pairs for ground-truth evaluation.
- Do NOT use sentences that still contain real Twitter handles (they were stripped in Week 1, so you are safe).

### If you are doing Presidio Evaluation:
- Run Microsoft Presidio's standard English recognizers on the dataset.
- Record precision, recall, F1 per PII type and per script type.
- Compare roman_only vs mixed_script vs devanagari_only performance — this is the core finding.

---

## 8. Dependencies & Environment

| Library | Version | Purpose |
|---------|---------|---------|
| datasets | latest | Loading LinCE from Hugging Face |
| pandas | latest | DataFrame management |
| pyarrow | latest | Parquet read/write |
| matplotlib | latest | Visualization |
| scikit-learn | latest | Stratified train/test splitting |

**Environment:** Google Colab (free tier). The notebook begins with a pip install cell — run it first.

---

## 9. File Map

`
D:\NLP\
 HingGuard_Week1_Data_Preprocessing.ipynb   # Main notebook
 HingGuard_Week1_Explained.md               # Plain-language walkthrough of the notebook
 decision.md                                # All design decisions with full rationale
 agents.md                                  # This file — project briefing for agents/collaborators
 hinglish_dataset.parquet                   # Main output dataset (fast-loading)
 hinglish_dataset.csv                       # Main output dataset (human-readable)
 dataset_summary.csv                        # Per-source/split statistics table
 script_distribution.png                    # Bar chart of script types
`

---

## 10. One-Paragraph Project Summary

HingGuard collects real Hindi-English code-mixed sentences from social media (sourced from L3Cube HingLID and LinCE benchmarks), cleans them (removing URLs, real usernames, duplicates, and junk-length entries), tags them by writing script (Roman / Devanagari / Mixed), and splits them into train/validation/test sets. This cleaned, labeled dataset is then used to evaluate how well existing PII detection tools — primarily Microsoft Presidio — can identify and anonymize personal information in Hinglish text. The project's contribution is both the curated dataset and the systematic evaluation of tool performance across script types, which has not been done for Hindi-English code-mixed data before.

---

*Last updated: Week 1 — Data Collection & Preprocessing*
