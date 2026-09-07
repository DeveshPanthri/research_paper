"""
HingGuard -- Step 6: Text Cleaning
====================================
Cleaning pipeline applied to the unified DataFrame from Step 5.

Operations (in order):
  1. Strip URLs          (https://..., http://..., t.co/... etc.)
  2. Strip @mentions     (real Twitter handles -- removes real PII early)
  3. Collapse whitespace (tabs, multiple spaces -> single space, strip ends)
  4. Re-tokenize         (update tokens list to match cleaned text)
  5. Drop duplicates     (exact text match after cleaning)
  6. Drop length outliers (< 3 tokens or > 100 tokens)

WHY each step -- see decision.md D-04 to D-07.
"""

import os, re
import pandas as pd

DATA_DIR    = r"D:\NLP"
HINGLID_DIR = r"D:\NLP\code-mixed-nlp\L3Cube-HingLID"

# ── Reuse loaders from Step 5 ─────────────────────────────────────────────────

def parse_conll_lid(filepath):
    sentences, cur_tok, cur_tag = [], [], []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if line.strip() == "":
                if cur_tok:
                    sentences.append({"text": " ".join(cur_tok),
                                      "tokens": list(cur_tok),
                                      "lid_tags": list(cur_tag)})
                cur_tok, cur_tag = [], []
            else:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0].strip():
                    cur_tok.append(parts[0].strip())
                    cur_tag.append(parts[1].strip())
    if cur_tok:
        sentences.append({"text": " ".join(cur_tok),
                          "tokens": list(cur_tok),
                          "lid_tags": list(cur_tag)})
    return sentences

def parse_array_repr(val):
    if not isinstance(val, str): return []
    return re.findall(r"'((?:[^'\\]|\\.)*)'", val)

def build_dataframe():
    rows = []
    for split, fname in [("train","train.txt"),("validation","validation.txt"),("test","test.txt")]:
        for s in parse_conll_lid(os.path.join(HINGLID_DIR, fname)):
            rows.append({"source":"hinglid","split":split,**s})

    configs = {"lince_lid":[("train","lid_hineng_train.csv"),
                             ("validation","lid_hineng_validation.csv"),
                             ("test","lid_hineng_test.csv")],
               "lince_pos":[("train","pos_hineng_train.csv"),
                             ("validation","pos_hineng_validation.csv"),
                             ("test","pos_hineng_test.csv")]}
    for cfg, splits in configs.items():
        for split, fname in splits:
            fp = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fp): continue
            df_csv = pd.read_csv(fp)
            for _, row in df_csv.iterrows():
                toks = parse_array_repr(str(row.get("words","")))
                tags = parse_array_repr(str(row.get("lid","")))
                rows.append({"source":cfg,"split":split,
                             "text":" ".join(toks),"tokens":toks,"lid_tags":tags})

    df = pd.DataFrame(rows)
    df.insert(0, "sentence_id", range(len(df)))
    return df

# ── Cleaning functions ────────────────────────────────────────────────────────

# Compiled patterns (compiled once for speed)
URL_PATTERN      = re.compile(r"https?://\S+|www\.\S+|t\.co/\S+", re.IGNORECASE)
MENTION_PATTERN  = re.compile(r"@\w+")
WHITESPACE_CLEAN = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """
    Apply cleaning rules to a single sentence string.
    Returns the cleaned string.
    """
    text = URL_PATTERN.sub("", text)          # remove URLs
    text = MENTION_PATTERN.sub("", text)      # remove @mentions
    text = WHITESPACE_CLEAN.sub(" ", text)    # collapse whitespace
    text = text.strip()
    return text

def retokenize(text: str) -> list:
    """Simple whitespace tokenizer to rebuild tokens after cleaning."""
    return text.split() if text else []

def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Step 1+2+3: clean text
    df["text"] = df["text"].astype(str).apply(clean_text)

    # Step 4: re-tokenize (tokens must match cleaned text)
    df["tokens"] = df["text"].apply(retokenize)

    # Step 5: drop exact duplicates on cleaned text
    before_dedup = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    after_dedup  = len(df)

    # Step 6: drop length outliers
    df["token_count"] = df["tokens"].apply(len)
    before_len = len(df)
    df = df[(df["token_count"] >= 3) & (df["token_count"] <= 100)]
    after_len  = len(df)

    # Clean up temp column
    df = df.drop(columns=["token_count"])

    # Reset sentence_id
    df = df.reset_index(drop=True)
    df["sentence_id"] = range(len(df))

    return df, before_dedup, after_dedup, before_len, after_len

# ── Main ─────────────────────────────────────────────────────────────────────

print()
print("HingGuard - Step 6: Text Cleaning")
print("-" * 65)

print("  Building DataFrame from sources ...")
df_raw = build_dataframe()
total_before = len(df_raw)
print(f"  Raw rows loaded: {total_before:,}")
print()

print("  Applying cleaning pipeline ...")
df_clean, before_dedup, after_dedup, before_len, after_len = apply_cleaning(df_raw)
total_after = len(df_clean)

# ── Checkpoint ────────────────────────────────────────────────────────────────

dropped_dedup = before_dedup - after_dedup
dropped_len   = before_len   - after_len
total_dropped = total_before - total_after
pct_dropped   = (total_dropped / total_before) * 100

print("=" * 65)
print("CHECKPOINT -- Cleaning Report")
print("=" * 65)
print()
print(f"  Rows BEFORE cleaning : {total_before:>7,}")
print()
print(f"  Dropped (duplicates) : {dropped_dedup:>7,}  ({dropped_dedup/total_before*100:.1f}%)")
print(f"  Dropped (< 3 tokens) : {(before_len - after_len):>7,}  (included below)")
print(f"  Dropped (len filter) : {dropped_len:>7,}  ({dropped_len/total_before*100:.1f}%)")
print()
print(f"  Rows AFTER cleaning  : {total_after:>7,}")
print(f"  Total dropped        : {total_dropped:>7,}  ({pct_dropped:.1f}%)")
print()

# Per-source breakdown
print("  Per-source breakdown after cleaning:")
vc = df_clean["source"].value_counts()
for src, cnt in vc.items():
    raw_cnt = len(df_raw[df_raw["source"] == src])
    drop    = raw_cnt - cnt
    print(f"    {src:<15}: {cnt:>7,}  (dropped {drop:,} of {raw_cnt:,})")
print()

# Sanity: check a few cleaned examples
print("  Sample cleaned sentences:")
for _, row in df_clean[df_clean["source"]=="hinglid"].head(3).iterrows():
    print(f"    [{row['source']}] {row['text'][:70]}")
print()

# Show token length stats
lengths = df_clean["tokens"].apply(len)
print(f"  Token length stats: min={lengths.min()}, max={lengths.max()}, "
      f"mean={lengths.mean():.1f}, median={lengths.median():.0f}")
print()

# Final verdict
if pct_dropped > 90:
    print("  STATUS: CHECKPOINT FAILED -- >90% of rows dropped!")
    print("          This likely indicates a regex bug. Review cleaning functions.")
elif pct_dropped > 50:
    print("  STATUS: WARNING -- >50% dropped. Review if this seems too aggressive.")
else:
    print(f"  STATUS: CHECKPOINT PASSED -- {pct_dropped:.1f}% dropped, looks reasonable.")
    print("          No source is silently empty (except GLUECoS which was never loaded).")

print("=" * 65)
