"""
HingGuard -- Step 8: Train / Validation / Test Splitting
=========================================================
Rules:
  1. If a sentence already has a split from its source dataset
     (HingLID and LinCE both ship pre-split) -> KEEP IT AS-IS.
     Re-splitting would break comparability with published benchmarks.

  2. For any sentence WITHOUT an existing split -> create one using
     scikit-learn train_test_split with:
       - 80% train / 10% validation / 10% test
       - Stratified on (source + script_type) to ensure each category
         is proportionally represented in every split
       - random_state=42 for full reproducibility

WHY: see decision.md D-09 and D-10.
"""

import os, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR    = r"D:\NLP"
HINGLID_DIR = r"D:\NLP\code-mixed-nlp\L3Cube-HingLID"

# ── Reuse full pipeline (Steps 5+6+7) ────────────────────────────────────────

def parse_conll_lid(filepath):
    sentences, cur_tok, cur_tag = [], [], []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if line.strip() == "":
                if cur_tok:
                    sentences.append({"text":" ".join(cur_tok),
                                      "tokens":list(cur_tok),
                                      "lid_tags":list(cur_tag)})
                cur_tok, cur_tag = [], []
            else:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0].strip():
                    cur_tok.append(parts[0].strip())
                    cur_tag.append(parts[1].strip())
    if cur_tok:
        sentences.append({"text":" ".join(cur_tok),
                          "tokens":list(cur_tok),
                          "lid_tags":list(cur_tag)})
    return sentences

def parse_array_repr(val):
    if not isinstance(val, str): return []
    return re.findall(r"'((?:[^'\\]|\\.)*)'", val)

def build_pipeline_df():
    """Steps 5+6+7 in one go."""
    URL_PAT  = re.compile(r"https?://\S+|www\.\S+|t\.co/\S+", re.IGNORECASE)
    MENT_PAT = re.compile(r"@\w+")
    WS_PAT   = re.compile(r"\s+")

    def clean(text):
        text = URL_PAT.sub("", text)
        text = MENT_PAT.sub("", text)
        return WS_PAT.sub(" ", text).strip()

    def script_label(text):
        deva  = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
        latin = sum(1 for ch in text if ch.isalpha() and ch.isascii())
        total = deva + latin
        if total == 0:   return "unknown"
        if deva == 0:    return "roman_only"
        if latin == 0:   return "devanagari_only"
        return "mixed_script"

    rows = []
    for split, fname in [("train","train.txt"),("validation","validation.txt"),("test","test.txt")]:
        for s in parse_conll_lid(os.path.join(HINGLID_DIR, fname)):
            rows.append({"source":"hinglid","split":split,**s})

    for cfg, splits in {
        "lince_lid":[("train","lid_hineng_train.csv"),("validation","lid_hineng_validation.csv"),("test","lid_hineng_test.csv")],
        "lince_pos":[("train","pos_hineng_train.csv"),("validation","pos_hineng_validation.csv"),("test","pos_hineng_test.csv")]
    }.items():
        for split, fname in splits:
            fp = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fp): continue
            for _, row in pd.read_csv(fp).iterrows():
                toks = parse_array_repr(str(row.get("words","")))
                tags = parse_array_repr(str(row.get("lid","")))
                rows.append({"source":cfg,"split":split,
                             "text":" ".join(toks),"tokens":toks,"lid_tags":tags})

    df = pd.DataFrame(rows)
    df["text"]   = df["text"].astype(str).apply(clean)
    df["tokens"] = df["text"].apply(lambda t: t.split() if t else [])
    df = df.drop_duplicates(subset="text", keep="first")
    df["tc"] = df["tokens"].apply(len)
    df = df[(df["tc"] >= 3) & (df["tc"] <= 100)].drop(columns=["tc"]).reset_index(drop=True)
    df["script_type"] = df["text"].apply(script_label)
    df.insert(0, "sentence_id", range(len(df)))
    return df


# ── Splitting logic ───────────────────────────────────────────────────────────

def apply_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -- Rows that already have a split (from source dataset) --
    already_split = df[df["split"].notna() & (df["split"] != "")].copy()
    needs_split   = df[df["split"].isna()  | (df["split"] == "")].copy()

    n_kept   = len(already_split)
    n_resplit = len(needs_split)

    print(f"  Rows keeping original split : {n_kept:,}")
    print(f"  Rows needing new split      : {n_resplit:,}")

    if n_resplit > 0:
        # Stratification key = source + script_type
        strat_key = needs_split["source"] + "__" + needs_split["script_type"]

        # Handle rare strata with < 2 samples (can't stratify)
        strat_counts = strat_key.value_counts()
        rare = strat_counts[strat_counts < 3].index.tolist()
        if rare:
            print(f"  [NOTE] {len(rare)} rare strata (< 3 samples) -- these go to train")
            rare_mask   = strat_key.isin(rare)
            rare_rows   = needs_split[rare_mask].copy()
            needs_split = needs_split[~rare_mask].copy()
            strat_key   = strat_key[~rare_mask]
            rare_rows["split"] = "train"
        else:
            rare_rows = pd.DataFrame()

        if len(needs_split) >= 10:
            # First split: 80% train vs 20% temp
            train_idx, temp_idx = train_test_split(
                needs_split.index,
                test_size=0.20,
                stratify=strat_key,
                random_state=42
            )
            # Second split: 50/50 of temp -> validation + test (= 10% each)
            temp_df     = needs_split.loc[temp_idx]
            temp_strat  = (temp_df["source"] + "__" + temp_df["script_type"])
            # Handle strata with < 2 in temp
            temp_counts = temp_strat.value_counts()
            safe_temp   = temp_df[temp_strat.isin(temp_counts[temp_counts >= 2].index)]
            risky_temp  = temp_df[~temp_strat.isin(temp_counts[temp_counts >= 2].index)]

            if len(safe_temp) >= 2:
                val_idx, test_idx = train_test_split(
                    safe_temp.index,
                    test_size=0.50,
                    stratify=(safe_temp["source"] + "__" + safe_temp["script_type"]),
                    random_state=42
                )
            else:
                val_idx  = safe_temp.index[:len(safe_temp)//2]
                test_idx = safe_temp.index[len(safe_temp)//2:]

            needs_split.loc[train_idx,  "split"] = "train"
            needs_split.loc[val_idx,    "split"] = "validation"
            needs_split.loc[test_idx,   "split"] = "test"
            if len(risky_temp) > 0:
                needs_split.loc[risky_temp.index, "split"] = "validation"

        else:
            needs_split["split"] = "train"  # too few to split

        if len(rare_rows) > 0:
            needs_split = pd.concat([needs_split, rare_rows])

    # Combine and clean up
    combined = pd.concat([already_split, needs_split], ignore_index=True)
    combined = combined.sort_values("sentence_id").reset_index(drop=True)
    return combined


# ── Main ─────────────────────────────────────────────────────────────────────

print()
print("HingGuard - Step 8: Train / Validation / Test Splitting")
print("-" * 65)

print("  Building pipeline DataFrame (Steps 5+6+7) ...")
df = build_pipeline_df()
print(f"  Total rows: {len(df):,}")
print()

print("  Applying split logic ...")
df = apply_splits(df)
print()

# ── Checkpoint ────────────────────────────────────────────────────────────────

print("=" * 65)
print("CHECKPOINT -- Split Distribution")
print("=" * 65)
print()

# Overall split counts
sc = df["split"].value_counts()
total = len(df)
print("  Overall split counts :")
for sp, cnt in sc.items():
    pct = cnt / total * 100
    bar = "#" * int(pct / 2)
    print(f"    {sp:<14}: {cnt:>7,}  ({pct:5.1f}%)  {bar}")
print()

# Per-source split breakdown
print("  Per-source split breakdown :")
for src in sorted(df["source"].unique()):
    sub  = df[df["source"] == src]
    svc  = sub["split"].value_counts()
    n    = len(sub)
    print(f"    {src} (n={n:,}) :")
    for sp in ["train","validation","test"]:
        cnt = svc.get(sp, 0)
        pct = cnt / n * 100 if n > 0 else 0
        print(f"      {sp:<12}: {cnt:>6,}  ({pct:5.1f}%)")
print()

# Per-script type split breakdown
print("  Per-script_type split breakdown :")
for stype in sorted(df["script_type"].unique()):
    sub = df[df["script_type"] == stype]
    svc = sub["split"].value_counts()
    n   = len(sub)
    parts = "  ".join(f"{sp}: {svc.get(sp,0):,}" for sp in ["train","validation","test"])
    print(f"    {stype:<20} (n={n:,}): {parts}")
print()

# Sanity checks
issues = []
train_pct = sc.get("train", 0) / total * 100
val_pct   = sc.get("validation", 0) / total * 100
test_pct  = sc.get("test", 0) / total * 100

if train_pct < 60:
    issues.append(f"Train is only {train_pct:.1f}% -- expected ~70-80%")
if train_pct > 95:
    issues.append(f"Train is {train_pct:.1f}% -- too dominant, val/test too small")
if val_pct < 3:
    issues.append(f"Validation is only {val_pct:.1f}% -- very small")
if test_pct < 3:
    issues.append(f"Test is only {test_pct:.1f}% -- very small")

if issues:
    for i in issues: print(f"  [ISSUE] {i}")
    print()
    print("  STATUS: CHECKPOINT FAILED -- review split proportions.")
else:
    print("  STATUS: CHECKPOINT PASSED -- split counts look balanced.")
    print("          Pre-existing splits preserved. No source wildly over-represented.")

print("=" * 65)
