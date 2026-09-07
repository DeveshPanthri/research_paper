# HingGuard — Research Notes

> Quick-reference notes, comparisons, and explanations built up during the project.
> Add new sections here as the project progresses.

---

## Dataset Comparison: L3Cube-HingLID vs LinCE

| | **L3Cube-HingLID** | **LinCE (lid_hineng + pos_hineng)** |
|---|---|---|
| **Made by** | L3Cube, Pune (Indian research lab) | RITUAL lab, Univ. of Houston (US) |
| **Size** | 44,455 sentences | 8,910 sentences |
| **Script** | Roman Hinglish (Twitter) | Roman Hinglish (Twitter) |
| **Tags available** | HI / EN per word | lang1 / lang2 / other / mixed per word + POS tags |
| **Tag meaning** | Hindi vs English | lang1=Hindi, lang2=English (+ finer: mixed, fw, ne) |
| **POS tags** | No | Yes (pos_hineng: NOUN, VERB, ADJ etc.) |
| **Pre-split** | Yes (train/val/test by authors) | Yes |
| **Source** | GitHub (free, no login) | Kaggle / ritual.uh.edu (needs account) |
| **Format** | CoNLL .txt (word TAB tag) | CSV (lists stored as strings) |
| **Our use** | Primary backbone dataset | Secondary benchmark + adds POS info |

### In plain words

- **HingLID** = bigger, simpler — just "is this word Hindi or English?" — perfect for training a language ID model.
- **LinCE** = smaller, richer — has more tag categories (mixed, ne, fw) AND part-of-speech tags — useful for more detailed analysis and as an independent benchmark to validate results against.

### Why we use both

If our PII detector performs well on both datasets independently, it is a much stronger claim than performing well on just one. Using two different sources from two different research groups also guards against dataset-specific biases.

---

*Last updated: Week 1 — Data Collection & Preprocessing*


---

## What We Did — Plain Language Summary (Week 1)

---

### The Big Goal
We are building a system that can find and hide personal information (like names, phone numbers, emails)
in Hindi-English mixed sentences (called Hinglish). Before we can build that system, we need
good quality Hinglish sentences to work with. Week 1 is all about collecting and cleaning those sentences.

---

### Step 1 — Set Up the Python Environment

**What we did:**
Installed 5 Python libraries: datasets, pandas, pyarrow, matplotlib, scikit-learn.
Then confirmed every single import works without errors.

**Why:**
These are the tools we need for the rest of the project.
- datasets = downloads NLP datasets from the internet
- pandas = stores our sentences in a spreadsheet-like table
- pyarrow = saves that table in a fast file format
- matplotlib = draws charts
- scikit-learn = splits data into train/test groups

**Result:** All 5 libraries installed. All imports working. Python 3.14.2.

---

### Step 2 — Get L3Cube-HingLID (Our Main Dataset)

**What we did:**
Downloaded a GitHub repository from L3Cube (an Indian research lab in Pune).
Inside it, we found the L3Cube-HingLID folder with 3 files:
train.txt, validation.txt, test.txt.
These files have one word per line with its language tag (HI for Hindi, EN for English),
and a blank line between sentences. This format is called CoNLL format.
We wrote a function (parse_conll_lid) that reads this format and converts it into
a list of sentences, each with its words and HI/EN tags.

**Why this dataset:**
- Free and public on GitHub, no login needed
- Already tagged word by word (HI or EN)
- Already split into train/validation/test, so we don't have to do it
- Real Twitter data, so it reflects how people actually write Hinglish

**Why not the bigger version (53 million sentences):**
- Too large, would crash free storage
- That version has no HI/EN word tags, so it's less useful for us right now

**Result:**
- Train: 31,756 sentences
- Validation: 6,279 sentences
- Test: 6,420 sentences
- Total: 44,455 sentences with clean HI/EN tags

---

### Step 3 — Get LinCE (Our Second Dataset)

**What we did:**
Tried to load LinCE from Hugging Face. Failed because Hugging Face updated their
library and stopped supporting the old-style loader LinCE uses.
Tried to download directly from the original website (ritual.uh.edu). Failed because
the server is down/unreachable.
Finally found the dataset on Kaggle. Downloaded 6 CSV files manually and loaded them
with a custom CSV parser.

**Why this dataset:**
- It is a different, independently made benchmark — using two sources makes our results stronger
- It has richer tags: not just HI/EN but also "mixed", "other", "named entity"
- It also has Part-of-Speech tags (NOUN, VERB, ADJ etc.) which are useful for deeper analysis
- Using two datasets from two different research groups guards against bias

**The try/except wrapper we wrote:**
The code tries each loading method one by one. If it fails, it catches the error,
prints a clear message explaining what went wrong and how to fix it, and moves on
without crashing the rest of the pipeline. This is important because a notebook
that crashes halfway through is very frustrating.

**Result:**
- lid_hineng (Language ID): 7,421 sentences (train + val + test)
- pos_hineng (Part-of-Speech): 1,489 sentences (train + val + test)
- Total from LinCE: 8,910 sentences

---

### Step 4 — Check GLUECoS (The Blocked Dataset)

**What we did:**
Downloaded the GLUECoS repository from Microsoft's GitHub.
Looked inside the 3 Hindi-English task folders:
Sentiment_EN_HI, NER_EN_HI, POS_EN_HI_UD.
Found that NONE of them contain actual sentences.
All three only contain files with numbers (tweet IDs or row index numbers).
These numbers point to tweets, but to get the actual tweet text you need
to call Twitter's API (called "hydration").
We checked for a twitter_authentication.txt file with API keys. It does not exist.

**Why we skipped it:**
- Twitter/X API requires a paid developer account (~$100/month minimum)
- The free Academic Research access was shut down in Feb 2023
- Without API keys, there is literally no way to get the text
- We already have 53,000+ sentences from other sources

**Why we documented it instead of just ignoring it:**
- In a research paper, you have to honestly report which datasets you tried to use
- Reviewers may ask "why didn't you use GLUECoS?" and we now have a written, citable answer
- We left a re-entry point in the code: if someone gets API keys later, they just create
  the twitter_authentication.txt file and re-run Step 4

**Result:** GLUECoS correctly identified as ID-only. Skipped without any errors or crashes.

---

### Decisions Made and Why

| Decision | What we chose | Why |
|---|---|---|
| Primary dataset | L3Cube-HingLID | Free, labeled, pre-split, no blockers |
| Secondary dataset | LinCE (Kaggle CSVs) | Independent benchmark, richer tags |
| GLUECoS | Skip entirely | No text available, Twitter API required |
| Full HingCorpus (53M) | Skip | Too large, no word-level tags |
| ritual.uh.edu direct download | Skip | Server is down |
| HuggingFace LinCE loader | Skip | HF v5.0 dropped support for it |
| Fallback strategy | CSV from Kaggle | Only working source for LinCE |

---

### Where We Stand Now

| Source | Sentences | Status |
|---|---|---|
| L3Cube HingLID | 44,455 | Loaded and parsed |
| LinCE | 8,910 | Loaded from Kaggle CSVs |
| GLUECoS | 0 | Skipped (documented) |
| TOTAL | 53,365 | Ready for cleaning |

---

### What Comes Next (Steps 5 onwards)

5. Combine all sentences into one single table (one row per sentence)
6. Clean the text (remove URLs, @mentions, duplicates, very short/long sentences)
7. Detect what script each sentence uses (Roman / Devanagari / Mixed)
8. Split into train/validation/test (keeping original splits where they exist)
9. Save as .parquet and .csv files
10. Draw charts showing the distribution

---

*Last updated: Week 1 Steps 1-4 complete*


---

### Step 5 — Combine Everything Into One Table

**What we did:**
Took all the sentences from Step 2 (HingLID) and Step 3 (LinCE) and merged them
into a single pandas DataFrame — like one big spreadsheet where every row is one sentence.
GLUECoS contributes 0 rows (as documented in Step 4).

Each row has exactly 6 columns:
- sentence_id  : a unique number for every sentence (0, 1, 2, ...)
- source       : where it came from (hinglid / lince_lid / lince_pos)
- split        : train / validation / test
- text         : the full sentence as one string
- tokens       : the sentence split into individual words (a list)
- lid_tags     : the HI/EN language tag for each word (a list)

**Why combine into one table:**
Instead of writing separate code for HingLID, separate code for LinCE etc.,
we write the cleaning and analysis code ONCE and it applies to all 53,365 sentences
at the same time. Every sentence also carries a "source" label so we can always
trace it back to where it came from.

**Result:**
- Shape: 53,365 rows x 6 columns
- hinglid: 44,455 sentences
- lince_lid: 7,421 sentences
- lince_pos: 1,489 sentences
- GLUECoS: 0 (correctly absent)

---

### Step 6 — Clean the Text

**What we did:**
Applied a 4-step cleaning pipeline to every sentence:

1. Remove URLs (https://..., t.co/... etc.)
   - Why: URLs are not language. They confuse models and add noise.

2. Remove @mentions (@username)
   - Why (two reasons):
     a) They are not useful language data
     b) They are REAL Twitter handles of REAL people. If we leave them in and
        then inject FAKE names for testing, the tools might accidentally catch
        these real handles and we'd think our system worked when it didnt.
        So we remove real personal data early to keep the evaluation clean.

3. Collapse whitespace (multiple spaces, tabs -> single space)
   - Why: Makes all sentences consistent for tokenizing.

4. Drop duplicate sentences
   - Why: Twitter data has lots of retweets. Same sentence in train AND test
          would give the model an unfair advantage (data leakage).

5. Drop sentences shorter than 3 words or longer than 100 words
   - Why: Under 3 words = usually an emoji or junk. Over 100 words = spam.

**Result:**
- Before: 53,365 sentences
- Dropped (duplicates): 660 (1.2%)
- Dropped (too short/long): 219 (0.4%)
- After: 52,486 sentences
- Total dropped: 879 (1.6%) -- healthy, not a bug

**How we know 1.6% is correct:**
- If 0% dropped: regex not working
- If 90%+ dropped: regex bug wiping real content
- 1.6% is in the "just right" zone

---

### Step 7 — Detect What Script Each Sentence Uses

**What we did:**
For every sentence, we counted how many characters fall in the Devanagari range
(Unicode U+0900 to U+097F -- these are the Hindi letters like क, ख, ग...)
vs how many are Latin letters (A-Z, a-z).

Based on those counts, we labelled each sentence:
- roman_only      : 0 Devanagari characters (all Latin/Roman)
- devanagari_only : 0 Latin characters (all Hindi script)
- mixed_script    : both Devanagari and Latin present
- unknown         : no alphabetic characters at all (rare edge case)

**Result:**
- roman_only: 52,482 (99.99%)
- mixed_script: 2 (two sentences with a Devanagari hashtag)
- unknown: 2 (sentences like "less-than 3 less-than 3" -- only symbols)
- devanagari_only: 0

**Why 99.9% roman_only is NOT a bug:**
Both HingLID and LinCE were collected specifically as Roman-script Hinglish --
people typing Hindi words using English letters ("kaise ho aap").
They are NOT Devanagari datasets. So nearly 100% roman_only is correct.

**Known limitation (D-08):**
roman_only cannot tell the difference between:
  - "I am going home"  (pure English, Roman letters)
  - "Main ghar ja raha hoon"  (Hinglish, also Roman letters)
Both look identical at the character level. To separate them, we use the
word-level lid_tags column which is already in our DataFrame.
This is flagged for the EDA team to address in Week 2.

---

### Decisions Added (Steps 5-7)

| Decision | What | Why |
|---|---|---|
| One unified DataFrame | Single table for all sources | Write cleaning code once, applies to all |
| source column | Track origin of every sentence | Always traceable, enables per-source stats |
| Remove URLs in cleaning | Regex strip | No linguistic value |
| Remove @mentions in cleaning | Regex strip | Real PII, corrupts fake-PII evaluation |
| Drop duplicates | Exact text match | Prevents data leakage |
| Filter 3-100 tokens | Length thresholds | Removes junk and spam |
| Unicode range for script | U+0900 to U+097F | Fast, no library needed, interpretable |
| Documented roman_only limitation | Note in script detection | Honest reporting for EDA team |

---

### Running Total After Steps 5-7

| Metric | Value |
|---|---|
| Raw sentences collected | 53,365 |
| After cleaning | 52,486 |
| script_type added | Yes (roman_only / mixed_script / devanagari_only) |
| Ready for splitting | Yes |

---

*Last updated: Steps 5-7 complete*

