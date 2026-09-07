"""
HingGuard -- Step 3 (v3): LinCE CSV Loader
============================================
Reads Kaggle LinCE CSVs. Columns 'words' and 'lid' are stored as
numpy/pandas array repr strings like:
  "['tok1' 'tok2' 'tok3'\n 'tok4']"
We parse them with a regex that extracts single-quoted tokens.
"""

import os, re
import pandas as pd

DATA_DIR = r"D:\NLP"

# ── Robust parser for numpy array repr strings ───────────────────────────────

def parse_array_repr(val):
    """
    Parse strings like "['word1' 'word2'\n 'word3']" into ['word1','word2','word3'].
    This is the format pandas uses when printing numpy object arrays.
    """
    if not isinstance(val, str):
        return []
    # Extract all single-quoted tokens (handles escaped quotes too)
    tokens = re.findall(r"'((?:[^'\\]|\\.)*)'", val)
    return tokens


def load_csv_split(filepath, config_name, split_name):
    if not os.path.exists(filepath):
        print(f"      [MISSING] {os.path.basename(filepath)}")
        return []

    df = pd.read_csv(filepath)
    sentences = []
    for _, row in df.iterrows():
        tokens   = parse_array_repr(str(row.get("words", "")))
        lid_tags = parse_array_repr(str(row.get("lid",   "")))
        text     = " ".join(tokens)
        sentences.append({
            "text":          text,
            "tokens":        tokens,
            "lid_tags":      lid_tags,
            "split":         split_name,
            "source_config": config_name,
        })
    return sentences


# ── File map ─────────────────────────────────────────────────────────────────

CONFIGS = {
    "lid_hineng": [
        ("train",      "lid_hineng_train.csv"),
        ("validation", "lid_hineng_validation.csv"),
        ("test",       "lid_hineng_test.csv"),
    ],
    "pos_hineng": [
        ("train",      "pos_hineng_train.csv"),
        ("validation", "pos_hineng_validation.csv"),
        ("test",       "pos_hineng_test.csv"),
    ],
}

# ── Main ─────────────────────────────────────────────────────────────────────

print()
print("HingGuard - Step 3: Loading LinCE (Kaggle CSV files)")
print("-" * 65)

all_data = {}
for config_name, splits in CONFIGS.items():
    print(f"\n  Loading '{config_name}' ...")
    config_sents = []
    for split_name, filename in splits:
        filepath = os.path.join(DATA_DIR, filename)
        sents = load_csv_split(filepath, config_name, split_name)
        config_sents.extend(sents)
        if sents:
            print(f"      {split_name:<12}: {len(sents):>5,} sentences  [{filename}]")
    all_data[config_name] = config_sents

# ── Checkpoint ────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("CHECKPOINT -- LinCE Sentence Counts")
print("=" * 65)

grand_total = 0
for cfg, sents in all_data.items():
    n = len(sents)
    grand_total += n
    by_split = {}
    for s in sents:
        by_split[s["split"]] = by_split.get(s["split"], 0) + 1
    split_str = "  |  ".join(f"{sp}: {cnt:,}" for sp, cnt in sorted(by_split.items()))
    print(f"  {cfg:<18}: {n:>5,} total    [ {split_str} ]")

print(f"  {'GRAND TOTAL':<18}: {grand_total:>5,} sentences")
print()

# Show clean example
for cfg, sents in all_data.items():
    if sents:
        ex = sents[0]
        print(f"Example ({cfg}, split={ex['split']}):")
        print(f"  text     : {ex['text'][:70]}")
        print(f"  tokens   : {ex['tokens'][:8]}")
        print(f"  lid_tags : {ex['lid_tags'][:8]}")
        print()
        break

# Status
loaded  = [c for c, s in all_data.items() if len(s) > 0]
missing = [c for c in CONFIGS if c not in loaded]
if len(loaded) == len(CONFIGS):
    print("  STATUS: CHECKPOINT PASSED -- Both LinCE configs loaded successfully.")
elif len(loaded) > 0:
    print(f"  STATUS: PARTIAL -- Loaded: {loaded}")
    print(f"          Missing entire config: {missing}")
    # Check which specific files were missing
    missing_files = []
    for cfg, splits in CONFIGS.items():
        for split_name, filename in splits:
            if not os.path.exists(os.path.join(DATA_DIR, filename)):
                missing_files.append(filename)
    if missing_files:
        print(f"          Missing files: {missing_files}")
        print("          Download these from the same Kaggle page and paste into D:/NLP/")
else:
    print("  STATUS: FAILED -- No data loaded.")

print("=" * 65)
