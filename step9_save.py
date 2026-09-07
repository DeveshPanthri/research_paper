import os, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- saves file without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split

DATA_DIR    = r"D:\NLP"
HINGLID_DIR = r"D:\NLP\code-mixed-nlp\L3Cube-HingLID"
OUT_DIR     = r"D:\NLP\processed"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Full pipeline (Steps 5+6+7+8) ────────────────────────────────────────────

def parse_conll_lid(filepath):
    sentences, cur_tok, cur_tag = [], [], []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if line.strip() == "":
                if cur_tok:
                    sentences.append({"text":" ".join(cur_tok),"tokens":list(cur_tok),"lid_tags":list(cur_tag)})
                cur_tok, cur_tag = [], []
            else:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0].strip():
                    cur_tok.append(parts[0].strip()); cur_tag.append(parts[1].strip())
    if cur_tok:
        sentences.append({"text":" ".join(cur_tok),"tokens":list(cur_tok),"lid_tags":list(cur_tag)})
    return sentences

def parse_array_repr(val):
    if not isinstance(val, str): return []
    return re.findall(r"'((?:[^'\\]|\\.)*)'", val)

def build_full_df():
    URL_PAT  = re.compile(r"https?://\S+|www\.\S+|t\.co/\S+", re.IGNORECASE)
    MENT_PAT = re.compile(r"@\w+")
    WS_PAT   = re.compile(r"\s+")
    def clean(t): return WS_PAT.sub(" ", MENT_PAT.sub("", URL_PAT.sub("", t))).strip()
    def script(text):
        deva  = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
        latin = sum(1 for ch in text if ch.isalpha() and ch.isascii())
        total = deva + latin
        if total == 0:  return "unknown"
        if deva == 0:   return "roman_only"
        if latin == 0:  return "devanagari_only"
        return "mixed_script"

    rows = []
    for sp, fn in [("train","train.txt"),("validation","validation.txt"),("test","test.txt")]:
        for s in parse_conll_lid(os.path.join(HINGLID_DIR, fn)):
            rows.append({"source":"hinglid","split":sp,**s})
    for cfg, splits in {
        "lince_lid":[("train","lid_hineng_train.csv"),("validation","lid_hineng_validation.csv"),("test","lid_hineng_test.csv")],
        "lince_pos":[("train","pos_hineng_train.csv"),("validation","pos_hineng_validation.csv"),("test","pos_hineng_test.csv")]
    }.items():
        for sp, fn in splits:
            fp = os.path.join(DATA_DIR, fn)
            if not os.path.exists(fp): continue
            for _, row in pd.read_csv(fp).iterrows():
                toks = parse_array_repr(str(row.get("words","")))
                tags = parse_array_repr(str(row.get("lid","")))
                rows.append({"source":cfg,"split":sp,"text":" ".join(toks),"tokens":toks,"lid_tags":tags})

    df = pd.DataFrame(rows)
    df["text"]        = df["text"].astype(str).apply(clean)
    df["tokens"]      = df["text"].apply(lambda t: t.split() if t else [])
    df = df.drop_duplicates(subset="text", keep="first")
    df["tc"] = df["tokens"].apply(len)
    df = df[(df["tc"] >= 3) & (df["tc"] <= 100)].drop(columns=["tc"]).reset_index(drop=True)
    df["script_type"] = df["text"].apply(script)
    df.insert(0, "sentence_id", range(len(df)))
    return df

# ── Main ─────────────────────────────────────────────────────────────────────

print()
print("HingGuard - Step 9: Save + Summary Stats + Chart")
print("-" * 65)

print("  Building full pipeline DataFrame ...")
df = build_full_df()
print(f"  Total rows: {len(df):,}")
print()

# ── 1. Save files ─────────────────────────────────────────────────────────────

parquet_path = os.path.join(OUT_DIR, "hinglish_dataset.parquet")
csv_path     = os.path.join(OUT_DIR, "hinglish_dataset.csv")

# Parquet can't store Python lists natively -- convert to strings for parquet
df_parquet = df.copy()
df_parquet["tokens"]   = df_parquet["tokens"].apply(str)
df_parquet["lid_tags"] = df_parquet["lid_tags"].apply(str)
df_parquet.to_parquet(parquet_path, index=False)

df.to_csv(csv_path, index=False)

print(f"  Saved: {parquet_path}")
print(f"  Saved: {csv_path}")
print()

# ── 2. Summary table ──────────────────────────────────────────────────────────

df["token_count"] = df["tokens"].apply(len)

summary = (
    df.groupby(["source","split"])
      .agg(
          sentence_count = ("sentence_id", "count"),
          avg_length     = ("token_count", "mean"),
      )
      .round({"avg_length": 1})
      .reset_index()
)

# Add totals per source
totals = (
    df.groupby("source")
      .agg(sentence_count=("sentence_id","count"), avg_length=("token_count","mean"))
      .round({"avg_length":1})
      .reset_index()
)
totals["split"] = "TOTAL"

summary_full = pd.concat([summary, totals], ignore_index=True)
summary_full = summary_full.sort_values(["source","split"]).reset_index(drop=True)

summary_path = os.path.join(OUT_DIR, "dataset_summary.csv")
summary_full.to_csv(summary_path, index=False)

print("  Summary table (sentence count + avg length per source/split):")
print()
prev_src = None
for _, row in summary_full.iterrows():
    if row["source"] != prev_src:
        print(f"  [{row['source']}]")
        prev_src = row["source"]
    marker = "  *" if row["split"] == "TOTAL" else "   "
    print(f"  {marker}  {row['split']:<14}: {int(row['sentence_count']):>6,} sentences   avg {row['avg_length']} tokens/sent")
print()
print(f"  Summary saved: {summary_path}")
print()

# ── 3. Stacked bar chart ──────────────────────────────────────────────────────

# Pivot: rows=source, cols=script_type
pivot = (
    df.groupby(["source","script_type"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)
pivot = pivot.set_index("source")

# Ensure all 3 columns exist
for col in ["roman_only","mixed_script","devanagari_only","unknown"]:
    if col not in pivot.columns:
        pivot[col] = 0

col_order  = ["roman_only","mixed_script","devanagari_only","unknown"]
col_colors = ["#4A90D9","#F5A623","#7ED321","#9B9B9B"]
col_labels = ["Roman only","Mixed script","Devanagari only","Unknown"]

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#1E1E2E")
ax.set_facecolor("#1E1E2E")

bottoms = [0] * len(pivot)
for col, color, label in zip(col_order, col_colors, col_labels):
    vals = pivot[col].values
    bars = ax.bar(pivot.index, vals, bottom=bottoms, color=color,
                  label=label, width=0.5, edgecolor="#1E1E2E", linewidth=0.8)
    # Label bars > 100
    for bar, bot, val in zip(bars, bottoms, vals):
        if val > 80:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bot + val/2,
                    f"{val:,}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    bottoms = [b + v for b, v in zip(bottoms, vals)]

ax.set_title("Script Type Distribution by Source Dataset",
             color="white", fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Source Dataset", color="#AAAAAA", fontsize=10)
ax.set_ylabel("Sentence Count", color="#AAAAAA", fontsize=10)
ax.tick_params(colors="#AAAAAA")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for spine in ax.spines.values():
    spine.set_edgecolor("#444466")
ax.legend(loc="upper right", facecolor="#2A2A3E", edgecolor="#444466",
          labelcolor="white", fontsize=9)

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "script_distribution.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"  Chart saved: {chart_path}")
print()

# ── Checkpoint ────────────────────────────────────────────────────────────────

print("=" * 65)
print("CHECKPOINT -- Output Files Verification")
print("=" * 65)
print()

files_to_check = {
    "hinglish_dataset.parquet": parquet_path,
    "hinglish_dataset.csv":     csv_path,
    "dataset_summary.csv":      summary_path,
    "script_distribution.png":  chart_path,
}

all_ok = True
for name, path in files_to_check.items():
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    status = "OK" if exists and size > 0 else "MISSING/EMPTY"
    if status != "OK": all_ok = False
    print(f"  [{status}]  {name:<30}  {size:>10,} bytes")

print()
print(f"  DataFrame shape  : {df.shape}")
print(f"  Columns          : {list(df.columns)}")
print()

if all_ok:
    print("  STATUS: CHECKPOINT PASSED -- all 4 output files exist and have content.")
else:
    print("  STATUS: CHECKPOINT FAILED -- some files missing or empty.")
print("=" * 65)
