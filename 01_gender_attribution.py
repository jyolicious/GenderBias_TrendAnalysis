#!/usr/bin/env python3
"""
01_gender_attribution_with_gazetteer.py

Robust gender attribution that integrates a per-decade 'prominent_chars' gazetteer.

Place this script in your project root. Create folder:
  prominent_chars/
    1950s.txt
    1960s.txt
    ...
Each file: lines "Movie,Character" (CSV-like). Movie names may include year in parentheses.

Outputs:
  gendered_dir/<decade>/<film>.csv
  gendered_dir/gendered_all_dialogues.csv
  gendered_dir/extraction_summary_gendered.csv
  gendered_dir/low_confidence_utterances.csv (if any)
"""

import os
import re
import csv
import math
import pathlib
from difflib import get_close_matches
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# Optional stanza dependency parsing (improves evidence detection). Script works without stanza.
try:
    import stanza
    stanza_available = True
except Exception:
    stanza_available = False

# -------- USER CONFIG ----------
CLEANED_DIR = "/home/pratik/projects/GenderBias_TrendAnalysis/cleaned_dir"
GENDERED_DIR = "/home/pratik/projects/GenderBias_TrendAnalysis/gendered_dir"
GAZ_DIR = "/home/pratik/projects/GenderBias_TrendAnalysis/prominent_chars"  # place your per-decade txts here
MALE_NAMES_FILE = os.path.join(GAZ_DIR, "male_names.txt")   # optional (global names)
FEMALE_NAMES_FILE = os.path.join(GAZ_DIR, "female_names.txt")
MASTER_IN = os.path.join(CLEANED_DIR, "all_dialogues.csv")
MASTER_OUT = os.path.join(GENDERED_DIR, "gendered_all_dialogues.csv")
os.makedirs(GENDERED_DIR, exist_ok=True)

# grouping and propagation params
MAX_GAP_SAME_UTTER = 1.5
MAX_GAP_SCENE = 8.0
PROP_WINDOW = 5

# weights for evidence (tune)
W_PRONOUN = 2.0
W_TITLE = 1.6
W_NAME = 3.2
W_DEP_CUE = 1.2
W_DESCRIPTOR = 0.9
W_CHAR_MENTION_WEAK = 0.8    # mention of a character known-gender => weak boost
W_CHAR_SELF_STRONG = 3.5     # "I am <Character>" => strong boost

# regexes
MALE_PRON = re.compile(r"\b(he|him|his|himself|father|dad|papa|beta|bhai)\b", flags=re.I)
FEMALE_PRON = re.compile(r"\b(she|her|hers|herself|mother|mom|mummy|beti|behen|maa)\b", flags=re.I)

# kinship tokens (expandable)
MALE_TOKS = {"father","husband","son","brother","uncle","mr","sir","king","actor","bhai","papa"}
FEMALE_TOKS = {"mother","wife","daughter","sister","aunty","mrs","miss","maam","queen","actress","behen","maa"}

# small descriptors
MALE_DESC = {"handsome","strong","brave","hero"}
FEMALE_DESC = {"beautiful","pretty","lovely","attractive","sexy","emotional"}

# optional dependency lemmas
MALE_DEP_LEMMAS = {"husband","father","son","brother"}
FEMALE_DEP_LEMMAS = {"wife","mother","daughter","sister"}

# ---------- stanza init (GPU optimized) ----------
nlp_en = None
if stanza_available:
    try:
        stanza.download('en', verbose=False)
        nlp_en = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,lemma,depparse',
            use_gpu=True,
            device=0,
            verbose=False,
            batch_size=32   # MASSIVE SPEED BOOST
        )
        print("Stanza GPU pipeline initialized.")
    except Exception as e:
        print("Failed to initialize stanza GPU:", e)
        nlp_en = None

def time_to_seconds(tstr):
    try:
        if isinstance(tstr, (float, int)):
            return float(tstr)
        tstr = str(tstr)
        if ":" not in tstr:
            return float(re.sub(r"[^\d\.]", "", tstr) or 0.0)
        tstr = tstr.replace(",", ".")
        parts = tstr.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600.0 + float(m) * 60.0 + float(s)
        if len(parts) == 2:
            m, s = parts
            return float(m) * 60.0 + float(s)
    except Exception:
        return 0.0

# --------- Gazetteer loading ----------
def normalize_title(t):
    """Lowercase, remove punctuation except spaces, collapse multiple spaces."""
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"\(.*?\)", "", t)   # drop parentheses content (year) for normalized form
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_gazetteer(gaz_folder):
    """
    Scan gaz_folder for files named like 1950s.txt and parse lines "Movie,Character".
    Returns:
      - mapping: decade -> movie_norm_title -> set(character_names)
      - movie_title_index: lowercase original movie names (for fuzzy matching)
    """
    decade_map = defaultdict(lambda: defaultdict(set))
    movie_index = defaultdict(set)  # decade -> set of original movie titles (raw)
    for p in pathlib.Path(gaz_folder).glob("*.txt"):
        fname = p.stem  # e.g., "1950s"
        decade = fname
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # accept CSV-like line: movie,character (some files may have extra commas - we join)
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 2:
                    continue
                movie = parts[0]
                # character might contain commas if more than 2 columns; join remainder
                char = ",".join(parts[1:]).strip()
                if not movie or not char:
                    continue
                norm = normalize_title(movie)
                decade_map[decade][norm].add(char.strip())
                movie_index[decade].add(movie.strip())
    return decade_map, movie_index

# load names (optional global lists)
def load_name_set(path):
    s = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    s.add(w)
    return s

MALE_NAMES = load_name_set(MALE_NAMES_FILE)
FEMALE_NAMES = load_name_set(FEMALE_NAMES_FILE)

# fuzzy helpers
def best_gazetteer_match(movie_basename, year, decade_map, movie_index):
    """
    Try match by:
      1) use year -> decade candidate (e.g., 1990 -> 1990s) and fuzzy match within that decade
      2) if no match, fuzzy match across all decades
    Returns: (decade_key, matched_norm_title, matched_original_titles_set) or (None,None,None)
    """
    def try_match_in_decade(dec_key):
        candidates = list(decade_map[dec_key].keys())
        if not candidates:
            return None
        # normalized basename
        norm_base = normalize_title(movie_basename)
        # direct exact match
        if norm_base in decade_map[dec_key]:
            return norm_base
        # fuzzy by close matches
        close = get_close_matches(norm_base, candidates, n=1, cutoff=0.6)
        if close:
            return close[0]
        # try token overlap (score by shared tokens)
        btokens = set(norm_base.split())
        best=None; best_score=0
        for c in candidates:
            score = len(btokens & set(c.split()))
            if score > best_score:
                best_score = score; best = c
        if best_score>=1:
            return best
        return None

    # compute decade from year
    if year:
        try:
            y = int(year)
            dec_key = f"{(y//10)*10}s"
            if dec_key in decade_map:
                m = try_match_in_decade(dec_key)
                if m:
                    return dec_key, m
        except:
            pass

    # try all decades: prefer decade with highest fuzzy similarity
    for dec in decade_map.keys():
        m = try_match_in_decade(dec)
        if m:
            return dec, m
    return None, None

# ---------- stanza init (GPU optimized) ----------

# ---------- Batch Dependency Parsing (SUPER FAST) ----------
def extract_dep_lemmas_batch(text_list):
    """
    Faster batch dependency parsing using stanza GPU.
    Input: list of utterance texts.
    Output: list of lemma sets for each utterance.
    """
    if not nlp_en:
        return [set() for _ in text_list]

    # Stanza separates documents using blank lines
    joined = "\n\n".join(text_list)

    try:
        doc = nlp_en(joined)
        lemma_sets = []
        current = set()

        for sent in doc.sentences:
            # blank line → boundary between inputs
            if sent.text.strip() == "":
                lemma_sets.append(current)
                current = set()
                continue

            for w in sent.words:
                current.add(w.lemma.lower())

        # append last set
        lemma_sets.append(current)
        return lemma_sets

    except Exception as e:
        print("Batch lemma extraction failed:", e)
        return [set() for _ in text_list]



# evidence function updated to use char list
def evidence_scores_with_chars(text, char_list_for_movie, dep_lemmas=None):
    male = 0.0; female = 0.0; evidence = {}
    t = (text or "").strip()
    low = t.lower()
    if dep_lemmas is None:
        dep_lemmas = set()

    # pronoun signals
    if MALE_PRON.search(t):
        male += W_PRONOUN; evidence["pronoun"]="male"
    if FEMALE_PRON.search(t):
        female += W_PRONOUN; evidence["pronoun"]="female"

    # kinship/title words
    for tok in MALE_TOKS:
        if re.search(r"\b"+re.escape(tok)+r"\b", low):
            male += W_TITLE; evidence["title"]="male"
    for tok in FEMALE_TOKS:
        if re.search(r"\b"+re.escape(tok)+r"\b", low):
            female += W_TITLE; evidence["title"]="female"

    # descriptor words
    for d in MALE_DESC:
        if re.search(r"\b"+re.escape(d)+r"\b", low):
            male += W_DESCRIPTOR; evidence.setdefault("descriptor","male")
    for d in FEMALE_DESC:
        if re.search(r"\b"+re.escape(d)+r"\b", low):
            female += W_DESCRIPTOR; evidence.setdefault("descriptor","female")

    # name patterns (self-intros)
    name_patterns = [
        r"\bmy name is ([A-Za-z' ]{2,60})\b",
        r"\bi am ([A-Za-z' ]{2,40})\b",
        r"\bi'm ([A-Za-z' ]{2,40})\b",
        r"\bthis is ([A-Za-z' ]{2,60})\b",
        r"\bcalled ([A-Za-z' ]{2,60})\b"
    ]
    for pat in name_patterns:
        m = re.search(pat, t, flags=re.I)
        if m:
            name = m.group(1).strip().lower()
            # check against char list first (exact or fuzzy)
            if char_list_for_movie:
                for c in char_list_for_movie:
                    if name == c.lower():
                        # strong self-reference
                        # infer gender of this character via name lists if possible
                        if c.lower() in MALE_NAMES:
                            male += W_CHAR_SELF_STRONG; evidence["self_ref"]="male"
                        elif c.lower() in FEMALE_NAMES:
                            female += W_CHAR_SELF_STRONG; evidence["self_ref"]="female"
                        else:
                            # if character contains title like 'captain', bias by title
                            if any(tok in c.lower() for tok in ["captain","mr","sir","king","actor"]):
                                male += W_CHAR_SELF_STRONG; evidence["self_ref"]="male_by_title"
                            elif any(tok in c.lower() for tok in ["mrs","miss","ms","madam","actress","queen"]):
                                female += W_CHAR_SELF_STRONG; evidence["self_ref"]="female_by_title"
                            else:
                                # neutral but strong because it's self-intro of known char
                                male += W_CHAR_SELF_STRONG * 0.5
                                female += W_CHAR_SELF_STRONG * 0.5
                        break
            # fallback to global name lists
            if name in MALE_NAMES:
                male += W_NAME; evidence["name"]="male"
            elif name in FEMALE_NAMES:
                female += W_NAME; evidence["name"]="female"
            else:
                # fuzzy char match
                if char_list_for_movie:
                    lower_chars = [c.lower() for c in char_list_for_movie]
                    matches = get_close_matches(name, lower_chars, n=1, cutoff=0.85)
                    if matches:
                        c = matches[0]
                        # treat as above (title heuristics)
                        if any(tok in c for tok in ["captain","mr","sir","king","actor"]):
                            male += W_CHAR_SELF_STRONG; evidence["name"]="male_by_title"
                        elif any(tok in c for tok in ["mrs","miss","ms","madam","actress","queen"]):
                            female += W_CHAR_SELF_STRONG; evidence["name"]="female_by_title"
                        else:
                            male += W_CHAR_SELF_STRONG*0.5; female += W_CHAR_SELF_STRONG*0.5
            break

    # ----- Improved character mention detection (full + partial token match) -----
    if char_list_for_movie:
        low_tokens = low.split()

        for c in char_list_for_movie:
            cname = c.lower()
            cname_tokens = cname.split()

            # 1. Full name match
            if cname in low:
                if cname in MALE_NAMES:
                    male += W_CHAR_MENTION_WEAK
                elif cname in FEMALE_NAMES:
                    female += W_CHAR_MENTION_WEAK
                else:
                    # title-based inference
                    if any(tok in cname for tok in ["captain","mr","sir","king","actor"]):
                        male += W_CHAR_MENTION_WEAK
                    elif any(tok in cname for tok in ["mrs","miss","ms","madam","actress","queen"]):
                        female += W_CHAR_MENTION_WEAK
                    else:
                        male += W_CHAR_MENTION_WEAK * 0.2
                        female += W_CHAR_MENTION_WEAK * 0.2
                evidence.setdefault("char_mention", c)
                continue
            
            # 2. Partial token overlap ("russell" inside "captain russell")
            overlap = len(set(cname_tokens) & set(low_tokens))
            if overlap > 0:
                if cname in MALE_NAMES:
                    male += W_CHAR_MENTION_WEAK * 0.6
                elif cname in FEMALE_NAMES:
                    female += W_CHAR_MENTION_WEAK * 0.6
                else:
                    male += W_CHAR_MENTION_WEAK * 0.1
                    female += W_CHAR_MENTION_WEAK * 0.1

                evidence.setdefault("char_mention_partial", c)
    # -------------------------------------------------------------

    # dependency lemmas passed externally via batch processing
    # NOTE: we don't call stanza here anymore (SUPER SPEED BOOST)
    if dep_lemmas:
        if any(l in MALE_DEP_LEMMAS for l in dep_lemmas):
            male += W_DEP_CUE; evidence["dep"] = "male"
        if any(l in FEMALE_DEP_LEMMAS for l in dep_lemmas):
            female += W_DEP_CUE; evidence["dep"] = "female"

    return male, female, evidence

# Utterance grouping (same as previous robust script)
def group_into_utterances(df):
    starts = [time_to_seconds(x) for x in df.get("start", pd.Series(["0"]*len(df)))]
    ends = [time_to_seconds(x) for x in df.get("end", pd.Series(["0"]*len(df)))]
    texts = df["text_clean"].astype(str).fillna("").tolist()
    n = len(texts)
    if n == 0:
        return []
    cur_start = starts[0]; cur_end = ends[0]; cur_text = texts[0]; cur_indices=[0]
    rows=[]
    for i in range(1,n):
        gap = starts[i]-cur_end
        if gap <= MAX_GAP_SAME_UTTER:
            cur_text = cur_text + " " + texts[i]
            cur_end = max(cur_end, ends[i])
            cur_indices.append(i)
        else:
            rows.append({"start":cur_start,"end":cur_end,"text":cur_text.strip(),"indices":cur_indices})
            cur_start = starts[i]; cur_end = ends[i]; cur_text = texts[i]; cur_indices=[i]
    rows.append({"start":cur_start,"end":cur_end,"text":cur_text.strip(),"indices":cur_indices})
    return rows

def assign_gender_to_utterances(utterances, char_list):
    labels=[]
    for u in utterances:
        t = u["text"]
        male, female, ev = evidence_scores_with_chars(
            t, 
            char_list,
            dep_lemmas=u.get("dep_lemmas", set())
        )
        total = male + female
        if total == 0:
            chosen = "unknown"; conf = 0.0
        else:
            if male > female: chosen = "male"
            elif female > male: chosen = "female"
            else: chosen = "unknown"
            conf = abs(male - female) / (total + 1e-9)
        labels.append({"gender":chosen,"confidence":float(conf),"male_score":float(male),"female_score":float(female),"evidence":ev})
    return labels

def propagate_context(utter_labels):
    n = len(utter_labels)
    genders=[u["gender"] for u in utter_labels]; confs=[u["confidence"] for u in utter_labels]
    for i in range(n):
        if genders[i]=="unknown" or confs[i] < 0.35:
            left=None; right=None
            for j in range(i-1, max(-1,i-PROP_WINDOW-1), -1):
                if genders[j]!="unknown" and confs[j]>=0.45:
                    left=genders[j]; break
            for j in range(i+1, min(n, i+PROP_WINDOW+1)):
                if genders[j]!="unknown" and confs[j]>=0.45:
                    right=genders[j]; break
            if left and right and left==right:
                genders[i]=left; confs[i]=max(confs[i],0.4)
            elif left and not right:
                genders[i]=left; confs[i]=max(confs[i],0.35)
            elif right and not left:
                genders[i]=right; confs[i]=max(confs[i],0.35)
    for i in range(n):
        utter_labels[i]["gender"]=genders[i]; utter_labels[i]["confidence"]=confs[i]
    return utter_labels

# main per-movie processing with gazetteer lookup
def process_movie_csv(path_obj, decade_map, movie_index):
    p = path_obj
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print("skip:", p, e); return None, None
    if df.empty:
        return None, None
    #----------------------DEBUGGING text_clean column issue ----------------------
    # Check if required column exists
    if "text_clean" not in df.columns:
        print(f"Warning: 'text_clean' column not found in {p}")
        print(f"Available columns: {list(df.columns)}")
        return None, None
    #-------------------------------------------------------------------------------

    # ----- Improved year + title extraction -----
    fname = p.stem

    # extract all years appearing anywhere in filename
    years_found = re.findall(r"(19|20)\d{2}", fname)
    year = years_found[0] if years_found else None

    # also check parent folder for decade (e.g., cleaned_dir/1990s/…)
    folder_decade = None
    for part in p.parts:
        if re.match(r"^\d{4}s$", part):
            folder_decade = part

    # normalize title candidate (remove noise words & quality tags)
    title_candidate = fname.replace("_", " ").replace("-", " ")

    # remove year from title candidate if detected
    if year:
        title_candidate = re.sub(year, "", title_candidate)

    # common encoding / rip / junk words
    junk = [
    "hindi", "web", "webrip", "webdl", "netflix", "prime", "amazon",
    "dvdrip", "bluray", "bdrip", "720p", "1080p", "2160p", "x264",
    "x265", "aac", "ac3", "dts", "esubs", "subs", "dubbed", "hdrip",
    "dvdscr", "untouched", "charmeleon", "silver", "rg", "juleyano",
    "juleyanos", "rip", "print"
    ]
    pattern = r"\b(" + "|".join(junk) + r")\b"
    title_candidate = re.sub(pattern, " ",  title_candidate, flags=re.I)
    title_candidate = re.sub(r"\s+", " ",   title_candidate).strip()
    # --------------------------------------------
    

    # attempt gazetteer match
    dec_match, norm_title = best_gazetteer_match(title_candidate, year, decade_map, movie_index)

    char_list = []
    if dec_match and norm_title:
        # extract char set and normalize names (strip)
        char_list = list(decade_map[dec_match].get(norm_title, []))
    else:
        # fallback: try fuzzy across all decades by matching original movie names
        # we attempt to match title_candidate to raw titles in movie_index
        all_titles = []
        for dec in movie_index:
            for t in movie_index[dec]:
                all_titles.append((dec, t))
        # build list of normalized titles
        norms = [normalize_title(t[1]) for t in all_titles]
        close = get_close_matches(normalize_title(title_candidate), norms, n=1, cutoff=0.6)
        if close:
            idx = norms.index(close[0])
            dec, orig_title = all_titles[idx]
            norm_title = normalize_title(orig_title)
            char_list = list(decade_map[dec].get(norm_title, []))

    

    # group into utterances
    utterances = group_into_utterances(df)

    # ---- Batch dependency lemmas for all utterances ----
    if nlp_en:
        all_texts = [u["text"] for u in utterances]
        all_lemmas = extract_dep_lemmas_batch(all_texts)
    else:
        all_lemmas = [set() for _ in utterances]

    # attach lemma sets to utterances
    for u, lem in zip(utterances, all_lemmas):
        u["dep_lemmas"] = lem

    utter_labels = assign_gender_to_utterances(utterances, char_list)
    utter_labels = propagate_context(utter_labels)

    # map back to rows
    labels_per_row=[]; confs_per_row=[]; male_scores=[]; female_scores=[]; evidence_list=[]
    for u, lab in zip(utterances, utter_labels):
        for idx in u["indices"]:
            labels_per_row.append(lab["gender"])
            confs_per_row.append(lab["confidence"])
            male_scores.append(lab["male_score"])
            female_scores.append(lab["female_score"])
            evidence_list.append(str(lab["evidence"]))

    # if lengths mismatch, fill unknowns
    if len(labels_per_row) != len(df):
        L = len(df)
        labels_per_row = (labels_per_row + ["unknown"]*L)[:L]
        confs_per_row = (confs_per_row + [0.0]*L)[:L]
        male_scores = (male_scores + [0.0]*L)[:L]
        female_scores = (female_scores + [0.0]*L)[:L]
        evidence_list = (evidence_list + ["{}"]*L)[:L]

    df["speaker_gender"] = labels_per_row
    df["gender_confidence"] = confs_per_row
    df["male_score"] = male_scores
    df["female_score"] = female_scores
    df["gender_evidence"] = evidence_list

    # return df and the char_list used (for logging)
    return df, {"chars_used": char_list, "matched_title": norm_title, "matched_decade": dec_match, "title_candidate": title_candidate, "year": year}

def main():
    # load gazetteer
    decade_map, movie_index = load_gazetteer(GAZ_DIR)
    print("Loaded gazetteer decades:", list(decade_map.keys()))

    # Exclude summary/log files from processing (they don't have dialogue columns)
    movie_paths = [p for p in pathlib.Path(CLEANED_DIR).rglob("*.csv") 
                   if p.name not in ["extraction_summary.csv"]]
    print(f"Found {len(movie_paths)} per-movie CSVs under {CLEANED_DIR}")

    master_rows = []
    summary = []
    low_confidence_utts = []

    for p in tqdm(movie_paths):
        df_out, meta = process_movie_csv(p, decade_map, movie_index)
        if df_out is None:
            continue
        # write per-movie
        # derive decade from path parts or from meta matched_decade
        decade_folder = "unknown"
        for part in p.parts:
            if re.match(r"^\d{4}s$", part):
                decade_folder = part; break
        # prefer matched decade if available
        if meta and meta.get("matched_decade"):
            decade_folder = meta["matched_decade"]
        out_folder = pathlib.Path(GENDERED_DIR) / decade_folder
        out_folder.mkdir(parents=True, exist_ok=True)
        out_file = out_folder / (p.stem + ".csv")
        df_out.to_csv(out_file, index=False, quoting=csv.QUOTE_MINIMAL)

        # collect low confidence examples
        low_mask = df_out["gender_confidence"] < 0.35
        if low_mask.any():
            examples = df_out.loc[low_mask, ["text_clean","gender_confidence"]].drop_duplicates().head(200)
            for _, r in examples.iterrows():
                low_confidence_utts.append({"file": str(p), "text": r["text_clean"], "confidence": float(r["gender_confidence"])})

        summary.append({"path": str(p), "film_id": p.stem, "decade": decade_folder, "lines": len(df_out), "out_file": str(out_file), "meta": meta})
        master_rows.append(df_out)

    if master_rows:
        dfall = pd.concat(master_rows, ignore_index=True)
        dfall.to_csv(MASTER_OUT, index=False, quoting=csv.QUOTE_MINIMAL)
        print("Wrote master:", MASTER_OUT, len(dfall))

    pd.DataFrame(summary).to_csv(os.path.join(GENDERED_DIR, "extraction_summary_gendered.csv"), index=False)
    if low_confidence_utts:
        pd.DataFrame(low_confidence_utts).to_csv(os.path.join(GENDERED_DIR, "low_confidence_utterances.csv"), index=False, quoting=csv.QUOTE_MINIMAL)
        print("Wrote low-confidence sample:", os.path.join(GENDERED_DIR, "low_confidence_utterances.csv"))
    print("Done.")

if __name__ == "__main__":
    main()
