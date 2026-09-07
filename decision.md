# HingGuard � Decision Log

> This file records every significant judgment call made during the project, with reasoning.
> When reviewers or examiners ask "why did you do X?", the answer is already written here.

---

## Week 1 � Data Collection & Preprocessing

---

### D-01 | Use L3Cube-HingLID as the primary data source

**Decision:** Use the smaller HingLID dataset (~32,000 sentences) from L3Cube as the main reliable source.

**Why:**
- Publicly available on GitHub � no sign-up, no approval required.
- Every word is already manually tagged as Hindi (HI) or English (EN) in CoNLL format.
- Comes pre-split into train, validation, and test files � no splitting needed.
- Real Twitter-sourced code-mixed text; high ecological validity.

**Rejected alternative:** The full L3Cube HingCorpus (53 million sentences on Google Drive).

**Why rejected:**
- Delivered as a single giant file � not git clone-able or pip install-able.
- No HI/EN word-level tags available in the full corpus, making it less useful for supervised tasks.
- Risk of filling up free Colab storage.
- Left behind manual download instructions in the notebook for future use if needed.

---

### D-02 | Use LinCE via Hugging Face datasets library

**Decision:** Load LinCE's lid_hineng (Language ID) and pos_hineng (Part-of-Speech) configs via load_dataset().

**Why:**
- Hugging Face handles download and unpacking automatically  no manual file management.
- Provides a second, independently curated benchmark that cross-validates our pipeline.
- Comes with trust_remote_code=True flag  a standard, safe practice for older-style HF datasets.

**Fallback implemented:** A try/except block catches failure gracefully, prints a clear warning, and provides manual download instructions — without crashing the rest of the notebook.

---

### D-02b | LinCE unavailable via both HF and direct download (documented, not a blocker)

**Discovered during execution (Step 3):**
- Hugging Face `datasets` v5.0 dropped support for `trust_remote_code` / script-based loaders. LinCE uses an old-style `lince.py` loader — HF now refuses to run it.
- `ritual.uh.edu` (the original data host) returned a connection timeout — server is unreachable.
- All known GitHub mirrors of LinCE CoNLL files returned 404.

**Decision:** Proceed without LinCE data for now. The try/except fallback worked exactly as designed — the pipeline does not crash and prints clear manual instructions.

**Impact on project:** L3Cube-HingLID (44,455 sentences, already fully loaded) is sufficient for preprocessing, cleaning, script detection, and splitting. LinCE can be added later if access is restored.

**Re-entry point:** If `lid_hineng.zip` / `pos_hineng.zip` are manually placed in `D:/NLP/lince_data/`, the step3 loader will auto-detect and parse them.

---

### D-03 | Skip GLUECoS entirely (document as a limitation)

**Decision:** Do not attempt to load GLUECoS Hindi-English tasks.

**Why:**
- After cloning the GLUECoS repo and inspecting actual files (not just the README), we discovered the Hindi-English tasks contain only Tweet ID numbers, not actual text.
- Retrieving text from IDs requires Twitter API access (paid/approved Twitter Developer account) � not available to the team.
- There is no legitimate workaround that doesn't involve fabricating data.

**Why document rather than ignore:**
- Citing a dataset limitation honestly is accepted and expected practice in NLP research papers.
- The notebook includes code that checks for twitter_authentication.txt � if API access is obtained later, the pipeline can resume.

---

### D-04 | Strip URLs from tweet text

**Decision:** Remove all URLs (e.g., https://t.co/xyz) during cleaning.

**Why:**
- URLs carry no linguistic or semantic value for language identification or PII detection.
- They can confuse tokenizers and downstream models.

---

### D-05 | Strip @mentions from tweet text

**Decision:** Remove all @username mentions during cleaning.

**Why (two reasons):**
1. Linguistic value: Mentions add no useful language signal for our tasks.
2. Data integrity (critical): These are real Twitter handles of real people. If real PII remains in the clean dataset and the team later deliberately injects fake PII for testing � the anonymization tools might accidentally catch real handles, corrupting accuracy measurements.

**Implication:** Removing real mentions ensures the only PII present in evaluation data is PII we knowingly injected ourselves.

---

### D-06 | Remove duplicate sentences

**Decision:** Drop all duplicate text entries after cleaning.

**Why:**
- Twitter data contains heavy retweet and copy-paste repetition.
- Duplicates artificially inflate dataset size.
- Duplicates across train and test cause data leakage, making results look better than they really are.

---

### D-07 | Filter out too-short and too-long sentences

**Decision:** Drop any sentence with fewer than 3 tokens or more than 100 tokens.

**Why:**
- Under 3 words: usually a single emoji, punctuation fragment, or non-linguistic junk.
- Over 100 words: typically spam, copy-pasted walls of text, or boilerplate � not representative of natural code-mixed speech.

---

### D-08 | Detect script using Unicode block ranges (character counting)

**Decision:** Classify each sentence as roman_only, devanagari_only, or mixed_script by counting characters in the Devanagari Unicode block (U+0900 to U+097F) vs. Latin range (A-Z, a-z).

**Why:**
- Simple, interpretable, and fast � no external library needed.
- Directly meaningful for downstream tool selection (Presidio behavior may differ across scripts).

**Known limitation (flagged in notebook):**
- Character-level script detection cannot distinguish "pure English in Roman script" from "Hinglish romanized" � both are 100% Latin characters.
- Word-level LID tags or a language-detection tool would be needed to separate them.
- Passed on to the EDA team for Week 2.

---

### D-09 | Respect original train/validation/test splits

**Decision:** For sentences sourced from HingLID and LinCE (which already come pre-split), preserve their original split assignments exactly.

**Why:**
- Re-splitting publisher-provided splits breaks comparability with prior published results.
- Future papers comparing to HingLID or LinCE benchmarks need to use the same splits.

---

### D-10 | Use stratified splitting for unlabeled data

**Decision:** For sentences without an existing split, use scikit-learn's train_test_split with stratification on (script_type, source) at 80/10/10.

**Why:**
- Stratification ensures each script type and source dataset is proportionally represented in all three splits.
- Without it, random chance could concentrate all Devanagari sentences in test and none in train.

**Reproducibility:** random_state=42 is used so the exact same split is produced every time � important for teammates checking each other's work and for paper reproducibility.

---

### D-11 | Save outputs in both Parquet and CSV formats

**Decision:** Save the final dataset table as both .parquet and .csv.

**Why:**
- .parquet � compact and fast to load back into pandas; ideal for downstream Python code.
- .csv � universally readable in Excel, text editors, and other tools; useful for manual inspection.

---

### D-12 | Combine all sources into one unified DataFrame before processing

**Decision:** Stack HingLID, LinCE (and future sources) into a single pandas DataFrame with standardized columns before applying any cleaning or analysis.

**Why:**
- Cleaning, script detection, and splitting logic is written once and applied uniformly.
- Every sentence carries a source column so it can always be traced back to its origin.
- Easier to produce cross-source statistics and charts.

**Schema:**
| Column       | Description                                   |
|--------------|-----------------------------------------------|
| sentence_id  | Unique running integer ID                     |
| source       | Origin dataset (hinglid, lince, etc.)         |
| split        | train / validation / test                     |
| text         | Full sentence as a single string              |
| tokens       | List of individual words                      |
| lid_tags     | List of HI/EN tags per word (where available) |

---

*Last updated: Week 1 � Data Collection & Preprocessing*
