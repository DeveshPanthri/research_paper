import os, sys

HINGLID_DIR = r"D:\NLP\code-mixed-nlp\L3Cube-HingLID"
SPLITS = {
    "train":      "train.txt",
    "validation": "validation.txt",
    "test":       "test.txt",
}

def parse_conll_lid(filepath):
    """
    Parse CoNLL-format LID file.
    Format: word<TAB>TAG per line, blank line = sentence boundary.
    Returns list of dicts: {text, tokens, lid_tags}
    """
    sentences = []
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
    # flush trailing sentence
    if current_tokens:
        sentences.append({
            "text":     " ".join(current_tokens),
            "tokens":   list(current_tokens),
            "lid_tags": list(current_tags),
        })
    return sentences

def load_hinglid():
    data = {}
    for split, filename in SPLITS.items():
        filepath = os.path.join(HINGLID_DIR, filename)
        parsed = parse_conll_lid(filepath)
        data[split] = parsed
        print(f"  Loaded {split:>12s}: {len(parsed):>7,} sentences")
    return data

print()
print("HingGuard - Step 2: Loading L3Cube-HingLID")
print("-" * 60)
data = load_hinglid()

print()
print("=" * 60)
print("CHECKPOINT -- L3Cube-HingLID Sentence Counts")
print("=" * 60)

total = 0
for split in ("train", "validation", "test"):
    n = len(data.get(split, []))
    total += n
    print(f"  {split:<14}: {n:>7,} sentences")
print(f"  {'TOTAL':<14}: {total:>7,} sentences")
print()

ex = data["train"][0]
print("Example sentence (train[0]):")
print(f"  text     : {ex['text'][:80]}")
print(f"  tokens   : {ex['tokens'][:10]}")
print(f"  lid_tags : {ex['lid_tags'][:10]}")
print(f"  length   : {len(ex['tokens'])} tokens")
print()

all_tags = set()
for split in data:
    for sent in data[split]:
        all_tags.update(sent["lid_tags"])
print(f"  Tag vocabulary: {sorted(all_tags)}")
print()

train_n = len(data.get("train", []))
if train_n >= 25000:
    print("  STATUS: CHECKPOINT PASSED -- train split count looks correct.")
else:
    print(f"  STATUS: WARNING -- train count ({train_n:,}) lower than expected (~32k).")
print("=" * 60)
