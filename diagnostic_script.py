# python3 - << 'EOF'
import pandas as pd
import pathlib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_dir")

bad = []
for p in pathlib.Path(CLEANED_DIR).rglob("*.csv"):
    try:
        df = pd.read_csv(p)
        if "text_clean" not in df.columns:
            print("\n❌ Missing text_clean:", p)
            print("Columns:", list(df.columns))
            bad.append(str(p))
    except Exception as e:
        print("\n❌ Failed to read:", p, e)
        bad.append(str(p))

print("\n---------- SUMMARY ----------")
print("Bad files:", len(bad))
for b in bad:
    print(" -", b)
EOF
