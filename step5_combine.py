"""
HingGuard -- Step 5: Combine All Sources into One DataFrame
============================================================
Loads data from Steps 2, 3, 4 and merges into one pandas DataFrame.

Schema:
  sentence_id  : int   -- unique row number
  source       : str   -- "hinglid" | "lince_lid" | "lince_pos" | "gluecoss"
  split        : str   -- "train" | "validation" | "test"
  text         : str   -- full sentence as one string
  tokens       : list  -- list of word strings
  lid_tags     : list  -- list of HI/EN/lang1/lang2/other tags per word
"""

import os, re, ast
import pandas as pd

DATA_DIR    = r"D:\NLP"
HINGLID_DIR = r"D:\NLP\code-mixed-nlp\L3Cube-HingLID"

# ============================================================
# LOADER A: L3Cube HingLID  (from Step 2)
# ============================================================

def parse_conll_lid(filepath):
    sentences      = []
    current_tokens = []
    current_tags   = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")
            if line.strip() == "":
                if current_tokens:
                    sentences.append({
                        "text":     " ".join(current_tokens),
                        "tokens":   list(current_tokens),
                        "lid_tags": list(current_tags),
                    })
                current_tokens = []
                current_tags   = []
            else:
                parts = line.split("\t")
                if len(parts) >= 2:
                    word = parts[0].strip()
                    tag  = parts[1].strip()
                    if word:
                        current_tokens.append(word)
                        current_tags.append(tag)
    if current_tokens:
        sentences.append({
            "text":     " ".join(current_tokens),
            "tokens":   list(current_tokens),
            "lid_tags": list(current_tags),
        })
    return sentences


def load_hinglid():
    rows = []
    for split_name, filename in [("train","train.txt"),("validation","validation.txt"),("test","test.txt")]:
        fp = os.path.join(HINGLID_DIR, filename)
        sents = parse_conll_lid(fp)
        for s in sents:
            rows.append({
                "source":   "hinglid",
                "split":    split_name,
                "text":     s["text"],
                "tokens":   s["tokens"],
                "lid_tags": s["lid_tags"],
            })
        print(f"  HingLID  {split_name:<12}: {len(sents):>7,} sentences")
    return rows


# ============================================================
# LOADER B: LinCE CSVs  (from Step 3)
# ============================================================

def parse_array_repr(val):
    """Parse numpy-style array repr strings like "['tok1' 'tok2']" into a list."""
    if not isinstance(val, str):
        return []
    tokens = re.findall(r"'((?:[^'\\]|\\.)*)'", val)
    return tokens


def load_lince_csv(filepath, config_label, split_name):
    if not os.path.exists(filepath):
        print(f"  LinCE    [MISSING] {os.path.basename(filepath)}")
        return []
    df = pd.read_csv(filepath)
    rows = []
    for _, row in df.iterrows():
        tokens   = parse_array_repr(str(row.get("words", "")))
        lid_tags = parse_array_repr(str(row.get("lid",   "")))
        rows.append({
            "source":   config_label,
            "split":    split_name,
            "text":     " ".join(tokens),
            "tokens":   tokens,
            "lid_tags": lid_tags,
        })
    return rows


def load_lince():
    rows = []
    configs = {
        "lince_lid": [
            ("train",      "lid_hineng_train.csv"),
            ("validation", "lid_hineng_validation.csv"),
            ("test",       "lid_hineng_test.csv"),
        ],
        "lince_pos": [
            ("train",      "pos_hineng_train.csv"),
            ("validation", "pos_hineng_validation.csv"),
            ("test",       "pos_hineng_test.csv"),
        ],
    }
    for config_label, splits in configs.items():
        for split_name, filename in splits:
            fp     = os.path.join(DATA_DIR, filename)
            sents  = load_lince_csv(fp, config_label, split_name)
            rows  += sents
            if sents:
                print(f"  LinCE    {config_label} {split_name:<12}: {len(sents):>5,} sentences")
    return rows


# ============================================================
# LOADER C: GLUECoS  (from Step 4 -- always 0)
# ============================================================

def load_gluecoss():
    print("  GLUECoS  SKIPPED (ID-only, no Twitter API) -> 0 sentences")
    return []


# ============================================================
# MAIN: Load, merge, assign IDs
# ============================================================

print()
print("HingGuard - Step 5: Building Unified DataFrame")
print("-" * 65)
print()
print("Loading sources ...")
print()

all_rows  = []
all_rows += load_hinglid()
print()
all_rows += load_lince()
print()
all_rows += load_gluecoss()
print()

# Build DataFrame
df = pd.DataFrame(all_rows, columns=["source","split","text","tokens","lid_tags"])
df.insert(0, "sentence_id", range(len(df)))   # add unique ID as first column

# ============================================================
# CHECKPOINT
# ============================================================

print("=" * 65)
print("CHECKPOINT -- Unified DataFrame")
print("=" * 65)
print()
print(f"  df.shape          : {df.shape}  (rows x cols)")
print()
print("  df.dtypes :")
for col, dtype in df.dtypes.items():
    print(f"    {col:<15}: {dtype}")
print()
print("  df['source'].value_counts() :")
vc = df["source"].value_counts()
for src, count in vc.items():
    note = ""
    if src == "gluecoss":
        note = "  <- expected 0"
    elif count == 0:
        note = "  <- WARNING: unexpectedly empty!"
    print(f"    {src:<15}: {count:>7,}{note}")
print()
print("  df['split'].value_counts() :")
sc = df["split"].value_counts()
for sp, count in sc.items():
    print(f"    {sp:<15}: {count:>7,}")
print()

# Sanity checks
issues = []
for src in ["hinglid", "lince_lid", "lince_pos"]:
    n = len(df[df["source"] == src])
    if n == 0:
        issues.append(f"Source '{src}' is unexpectedly empty!")

if df["sentence_id"].nunique() != len(df):
    issues.append("sentence_id values are NOT unique!")

if df["text"].isnull().any():
    issues.append(f"{df['text'].isnull().sum()} rows have null text!")

# Show one example row
print("  Example row (row 0) :")
row = df.iloc[0]
print(f"    sentence_id : {row['sentence_id']}")
print(f"    source      : {row['source']}")
print(f"    split       : {row['split']}")
print(f"    text        : {row['text'][:65]}")
print(f"    tokens      : {row['tokens'][:7]}")
print(f"    lid_tags    : {row['lid_tags'][:7]}")
print()

# Final verdict
if issues:
    for issue in issues:
        print(f"  [ISSUE] {issue}")
    print()
    print("  STATUS: CHECKPOINT FAILED -- review issues above.")
else:
    print("  STATUS: CHECKPOINT PASSED -- DataFrame looks sane.")
    print("          All expected sources present. GLUECoS correctly 0.")

print("=" * 65)
