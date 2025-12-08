import os
import re
import csv
import glob
import pathlib
from slugify import slugify
from tqdm import tqdm
import pandas as pd

# Try to import pysrt; provide fallback guidance if missing
try:
    import pysrt
except Exception as e:
    raise RuntimeError("Please install pysrt: pip install pysrt") from e

# Optional: chardet to detect unknown encodings
try:
    import chardet
except:
    chardet = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset")          # where your downloaded .srt files live (organized by decade or flat)
OUT_DIR = os.path.join(BASE_DIR, "cleaned_dir")       # output per-decade and master CSV
MASTER_CSV = os.path.join(OUT_DIR, "all_dialogues.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# Regexes for cleaning
RE_BRACKETS = re.compile(r"(\[.*?\]|\(.*?\)|\{.*?\})", flags=re.UNICODE)    # stage directions
RE_HTML = re.compile(r"<[^>]+>")                                            # html tags
RE_MUSIC = re.compile(r"[♪♫]+")                                             # musical symbols
RE_MULTISPACE = re.compile(r"\s{2,}")
RE_SPEAKER_PREFIX = re.compile(r"^[A-Z][A-Z0-9\s\.\-]{0,40}:\s+", flags=re.UNICODE)  # "RAJ:" or "RAJ K:" etc
RE_ONLY_PUNCT = re.compile(r"^[\W_]+$")

def detect_encoding(path):
    if not chardet:
        return "utf-8"
    with open(path, "rb") as f:
        raw = f.read(20000)
    res = chardet.detect(raw)
    enc = res.get("encoding") or "utf-8"
    return enc

def clean_sub_text(text):
    if not text:
        return ""
    # Replace newlines within the subtitle with space
    t = text.replace("\r", " ").replace("\n", " ").strip()
    # Remove bracketed stage directions, html tags, music symbols
    t = RE_BRACKETS.sub(" ", t)
    t = RE_HTML.sub(" ", t)
    t = RE_MUSIC.sub(" ", t)
    # Remove common speaker prefixes like "RAJ:" at start (careful — removes only UPPERCASE prefix + colon)
    t = RE_SPEAKER_PREFIX.sub("", t)
    # Collapse multiple spaces
    t = RE_MULTISPACE.sub(" ", t)
    # Trim
    t = t.strip()
    return t

def is_dialogue_like(text):
    if not text:
        return False
    # After cleaning, discard if only punctuation or empty
    if RE_ONLY_PUNCT.match(text):
        return False
    # Discard some obvious non-dialogue short tokens (e.g., single punctuation)
    # Keep short words like "Money!" so threshold is low
    # Must have at least one alphanumeric char
    if not re.search(r"[A-Za-z0-9\u0900-\u097F]", text):
        return False
    return True

def process_srt_file(path):
    # Derive metadata from path
    # If files stored as raw_subtitles/<decade>/movie.srt, pick decade
    # find decade folder by checking any part that matches pattern like "1980s"
    p = pathlib.Path(path)
    decade = None
    for part in p.parts:
        if re.match(r"^\d{4}s$", part):   # matches 1950s, 1960s … 2020s
            decade = part
            break
    if decade is None:
        decade = "unknown"
    filename = p.stem
    film_id = slugify(filename)

    # Try to load with pysrt; attempt encoding detection if errors occur
    subs = None
    try:
        subs = pysrt.open(path, encoding='utf-8')
    except Exception:
        enc = detect_encoding(path)
        try:
            subs = pysrt.open(path, encoding=enc)
        except Exception:
            # last-resort: read raw and split by blank line (very simple fallback)
            with open(path, "r", errors="ignore") as f:
                raw = f.read()
            # naive split
            blocks = re.split(r"\n\s*\n", raw)
            subs = []
            for b in blocks:
                # try to extract time and text
                lines = b.strip().splitlines()
                if len(lines) >= 2:
                    txt = " ".join(lines[1:])
                    # create a tiny object with .start .end .text for compatibility
                    class S:
                        def __init__(self, s, e, t):
                            self.start = s
                            self.end = e
                            self.text = t
                    subs.append(S("00:00:00,000", "00:00:00,000", txt))

    rows = []
    for item in subs:
        # pysrt returns items with .start/.end as SubRipTime; else strings from fallback
        start = str(item.start) if hasattr(item, "start") else ""
        end = str(item.end) if hasattr(item, "end") else ""
        original = item.text if hasattr(item, "text") else str(item)
        cleaned = clean_sub_text(original)
        if not is_dialogue_like(cleaned):
            # skip pure stage directions like [LAUGH], [MUSIC] etc
            continue
        rows.append({
            "film_id": film_id,
            "decade": decade,
            "file_path": str(path),
            "start": start,
            "end": end,
            "text_clean": cleaned,
            "text_raw": original.strip()
        })
    return film_id, decade, rows

def write_per_movie_csv(decade, film_id, rows):
    out_folder = os.path.join(OUT_DIR, decade)
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, f"{film_id}.csv")
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_file, index=False, quoting=csv.QUOTE_MINIMAL)
    return out_file, len(df)

def main():
    srt_paths = [p for p in glob.glob(os.path.join(RAW_DIR, "**", "*.srt"), recursive=True)]
    print(f"Found {len(srt_paths)} .srt files under '{RAW_DIR}'")
    master_rows = []
    summary = []

    for path in tqdm(srt_paths):
        film_id, decade, rows = process_srt_file(path)
        out_file, count = write_per_movie_csv(decade or "unknown", film_id, rows)
        summary.append({"path": path, "film_id": film_id, "decade": decade, "lines_extracted": count, "out_file": out_file})
        master_rows.extend(rows)

    # Write master CSV
    if master_rows:
        df_all = pd.DataFrame(master_rows)
        df_all.to_csv(MASTER_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"\nWrote master CSV: {MASTER_CSV} ({len(df_all)} dialogue rows)")
    else:
        print("\nNo dialogues extracted.")

    # Summary CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUT_DIR, "extraction_summary.csv"), index=False)
    print("Summary written to cleaned_dialogues/extraction_summary.csv")
    print("Per-movie CSVs are in cleaned_dialogues/<decade>/")

if __name__ == "__main__":
    main()
