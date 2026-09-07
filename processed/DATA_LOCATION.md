# Week 1 Processed Data

Full dataset files are hosted on Google Drive (too large for git):
https://drive.google.com/drive/folders/1eznKzRRahF38P5Lh-kUo-dGRnTq0lcIW?usp=drive_link

## Files in that folder
- hinglish_dataset.parquet (12.4 MB) -- main dataset, use this one (smaller, faster to load)
- hinglish_dataset.csv (29.2 MB) -- same data, CSV format (use only if you need to open it outside Python)
- dataset_summary.csv (367 bytes) -- sentence counts and avg length per source/split (also committed directly to this repo)
- script_distribution.png (51 KB) -- chart of script type by source (also committed directly to this repo)

## How to use
1. Open the Drive folder above and download hinglish_dataset.parquet
2. Place it in a local `processed/` folder in your copy of this repo
3. Load it with:
   import pandas as pd
   df = pd.read_parquet("processed/hinglish_dataset.parquet")

## Generated
Date: 2026-09-07
Source notebook: notebooks/HingGuard_Week1_Data_Preprocessing.ipynb
