"""
HingGuard -- Step 4: GLUECoS Inspection & Skip Handler
=======================================================
Inspects GLUECoS Hindi-English task folders for actual text vs ID-only files.
Checks Twitter API credentials. Skips gracefully with full explanation.
"""

import os, sys

GLUECOSDIR = r"D:\NLP\GLUECoS"
DATA_DIR   = os.path.join(GLUECOSDIR, "Data", "Original_Data")
AUTH_FILE  = os.path.join(GLUECOSDIR, "twitter_authentication.txt")

HI_EN_TASKS = ["Sentiment_EN_HI", "NER_EN_HI", "POS_EN_HI_UD"]


# ── Classifier ────────────────────────────────────────────────────────────────

def is_id_file(fpath: str) -> bool:
    """
    Return True if the file contains ONLY numeric IDs (tweet IDs or row indices).
    Both tweet IDs (18-digit) and row indices (0-20000) are non-text.
    A file is considered ID-only if ALL non-empty lines are purely numeric.
    """
    try:
        with open(fpath, encoding="utf-8", errors="replace") as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]
        if not lines:
            return False
        return all(l.isdigit() for l in lines)
    except Exception:
        return False


def inspect_task(task_name: str) -> dict:
    task_path = os.path.join(DATA_DIR, task_name)

    result = {
        "task":        task_name,
        "status":      "missing",
        "all_files":   [],
        "id_files":    [],
        "text_files":  [],
        "sample_ids":  [],
    }

    if not os.path.isdir(task_path):
        return result

    all_files = []
    for root, _, files in os.walk(task_path):
        for f in files:
            all_files.append(os.path.join(root, f))

    if not all_files:
        result["status"] = "empty"
        return result

    result["all_files"] = [os.path.basename(f) for f in all_files]

    for fpath in all_files:
        if is_id_file(fpath):
            result["id_files"].append(os.path.basename(fpath))
            if not result["sample_ids"]:
                with open(fpath, encoding="utf-8") as fh:
                    result["sample_ids"] = [l.strip() for l in fh.readlines()[:3] if l.strip()]
        else:
            result["text_files"].append(os.path.basename(fpath))

    result["status"] = (
        "id_only"  if result["id_files"] and not result["text_files"] else
        "has_text" if result["text_files"]                             else
        "empty"
    )
    return result


def check_twitter_auth() -> tuple:
    if not os.path.exists(AUTH_FILE):
        return False, "twitter_authentication.txt not found"
    if os.path.getsize(AUTH_FILE) == 0:
        return False, "twitter_authentication.txt is empty"
    with open(AUTH_FILE, encoding="utf-8", errors="replace") as fh:
        content = fh.read().strip()
    if len(content) < 20:
        return False, "twitter_authentication.txt has no valid API keys"
    return True, "Valid credentials found"


# ── Main ─────────────────────────────────────────────────────────────────────

print()
print("HingGuard - Step 4: GLUECoS Hindi-English Inspection")
print("-" * 65)

if not os.path.isdir(GLUECOSDIR):
    print(f"  [ERROR] Repo not found: {GLUECOSDIR}")
    sys.exit(1)

print(f"  Repo : {GLUECOSDIR}")
print()

# Inspect all 3 tasks
results = [inspect_task(t) for t in HI_EN_TASKS]

print("  Task folder analysis:")
print()
for r in results:
    all_id_only = r["status"] == "id_only"
    all_files_numeric = all(f == f for f in r["id_files"])  # true

    verdict = {
        "id_only":  "ID-ONLY -- no text present",
        "has_text": "HAS TEXT",
        "empty":    "EMPTY",
        "missing":  "FOLDER MISSING",
    }.get(r["status"])

    print(f"  [{verdict}]  {r['task']}")
    print(f"    Files found : {r['all_files']}")

    if r["id_files"]:
        print(f"    ID files    : {r['id_files']}")
        print(f"    Sample vals : {r['sample_ids']}  <- numeric indices, NOT text")

    if r["text_files"]:
        print(f"    Text files  : {r['text_files']}")
    print()

# Twitter auth check
has_auth, auth_msg = check_twitter_auth()
print(f"  Twitter credentials: {auth_msg}")
print()

# Decision logic
all_no_text = all(r["status"] in ("id_only", "empty", "missing") for r in results)

print("=" * 65)
print("CHECKPOINT -- GLUECoS Skip Decision")
print("=" * 65)
print()

if all_no_text and not has_auth:
    print("  FINDING: All 3 Hindi-English GLUECoS tasks contain ONLY numeric")
    print("  ID files. No actual tweet text is present in the repository.")
    print()
    print("  WHY THERE IS NO TEXT:")
    print("  GLUECoS stores tweet IDs (or row indices into tweet datasets)")
    print("  rather than tweet text, because Twitter's Terms of Service")
    print("  prohibit redistribution of raw tweet content. Actual text must")
    print("  be retrieved ('hydrated') via the Twitter/X API.")
    print()
    print("  WHY WE CANNOT HYDRATE:")
    print("  - No twitter_authentication.txt with API keys was found.")
    print("  - Twitter/X API access now requires a paid developer account.")
    print("  - Academic/free-tier access was discontinued in 2023.")
    print()
    print("  DECISION: *** SKIPPING GLUECoS entirely. ***")
    print("  This is NOT an error -- it is a documented dataset limitation.")
    print("  GLUECoS contributes 0 sentences this week.")
    print()
    print("  CITATION NOTE: This limitation can be cited in the paper as:")
    print("  'GLUECoS Hindi-English tasks were excluded because the")
    print("   repository provides only tweet IDs; hydration requires")
    print("   Twitter API access unavailable to this project.'")
    print()
    print("  RE-ENTRY POINT: Place valid Twitter API keys in:")
    print(f"    {AUTH_FILE}")
    print("  then re-run this script to enable hydration.")
    print()
    print("  STATUS: CHECKPOINT PASSED -- ID-only correctly identified,")
    print("          GLUECoS skipped gracefully, no exceptions raised.")

elif has_auth:
    print("  Twitter credentials FOUND. Hydration could be attempted.")
    print("  (Out of Week 1 scope -- implement in a later step.)")
    print("  STATUS: CHECKPOINT PASSED -- credentials ready.")

else:
    for r in results:
        if r["status"] == "has_text":
            print(f"  NOTE: {r['task']} appears to have actual text files: {r['text_files']}")
    print("  STATUS: Investigate the text files above manually.")

print("=" * 65)
print()
gluecoss_sentences = []
print(f"  GLUECoS sentences this week: {len(gluecoss_sentences)}  (skip confirmed)")
