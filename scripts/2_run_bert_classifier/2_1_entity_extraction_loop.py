import os
import csv
import pandas as pd
import sys, os
from pathlib import Path

# TODO this doesn't currently move over all other job columns, like date posted etc.
# TODO check with Apostolos, is it true that bert only checks first 100 words?

# NOTE: I adapted linker.py; I added a counter

# ----------------------------------
# Setup
# ----------------------------------

HERE = Path(__file__).resolve()
# scripts/2_run_bert_classifier/<this_file>.py  -> repo root is two levels up
ROOT = HERE.parents[2]              # .../tobiya-livelihoods-classifier
SRC  = ROOT / "src"                 # if you use a src/ layout, this covers it too

for p in (ROOT, SRC):
    if p.exists():
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

# If your code uses repo-relative file paths, this helps them resolve consistently:
os.chdir(ROOT)

# Now project specific import
from inference.linker import EntityLinker

# Initialize the entity linker
## Note: I prefer to first get uuids and map them to preferred label during cleaning, as the if you ask EntityLinker to get preffered_labe directly is inconsistent and sometimes still gets uuids
pipeline = EntityLinker(k=50, output_format='uuid')   # occupations
pipeline_skills = EntityLinker(k=50, output_format='uuid')  # skills

# Input / output
basedir = "C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/"
filepath = basedir + "data/pre_study/harambee_jobs_clean_without_duplicates.csv"
filepath_write = basedir + "data/pre_study/BERT_extracted_occupations_skills_uuid.csv"

# ----------------------------------
# Read & prep input
# ----------------------------------
df = pd.read_csv(filepath)
print("Number of rows in the dataset before extraction:")
print(df.shape[0])

# Ensure required text fields are non-null to avoid linker errors
for c in ["full_details", "opportunity_description", "opportunity_requirements"]:
    if c in df.columns:
        df[c] = df[c].fillna('')
    else:
        # If any expected column is missing, create empty column
        df[c] = ''

# ----------------------------------
# Determine which opportunity_group_id have already been processed
# ----------------------------------
processed_ids = set()
file_exists = os.path.exists(filepath_write)

if file_exists:
    try:
        # Read only the ID column from the existing output
        prev = pd.read_csv(filepath_write, usecols=["opportunity_group_id"])
        processed_ids = set(prev["opportunity_group_id"].astype(str))
        print(f"Found {len(processed_ids):,} previously processed opportunities in the output file.")
    except Exception as e:
        # If the existing file is malformed or missing the column, fall back to treating as empty
        print(f"Warning: Could not read existing processed IDs from output ({e}). Proceeding as if none were processed.")

# ----------------------------------
# Open output file (append if exists, write header only when creating)
# ----------------------------------
write_header = not file_exists
mode = "a" if file_exists else "w"

fields = [
    "opportunity_group_id", "opportunity_ref_id", "opportunity_title",
    "opportunity_description", "opportunity_requirements", "full_details",
    "extracted_occupation_from_title", "extracted_occupation_from_full_details", "extracted_optional_skills", "extracted_essential_skills"
]

n_seen_this_run = set()
n_skipped_existing = 0
n_skipped_missing_id = 0
n_written = 0

with open(filepath_write, mode=mode, newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(fields)

    # ----------------------------------
    # Process only new opportunities
    # ----------------------------------
    for row in df.itertuples(index=False):
        # Pull fields safely (namedtuple access)
        gid = getattr(row, "opportunity_group_id", None)

        # Skip if ID missing
        if pd.isna(gid):
            n_skipped_missing_id += 1
            continue

        gid_str = str(gid)

        # Skip if already processed before or already handled in this run
        if gid_str in processed_ids or gid_str in n_seen_this_run:
            n_skipped_existing += 1
            continue

        # Only now do the heavy work
        title = getattr(row, "opportunity_title", "")
        full = getattr(row, "full_details", "")
        occ = pipeline(title)
        occ_full = pipeline(full)

        desc = getattr(row, "opportunity_description", "")
        reqs = getattr(row, "opportunity_requirements", "")

        skills_from_desc = pipeline_skills(desc)
        skills_from_reqs = pipeline_skills(reqs)

        ref_id = getattr(row, "opportunity_ref_id", "")

        writer.writerow([
            gid_str, ref_id, title, desc, reqs, full,
            occ, occ_full, skills_from_desc, skills_from_reqs
        ]) # CAREFUL when changing this, given that column names are defined further above

        n_seen_this_run.add(gid_str)
        n_written += 1

print(
    "Done.\n"
    f"  Newly written rows: {n_written:,}\n"
    f"  Skipped (already in output or duplicate this run): {n_skipped_existing:,}\n"
    f"  Skipped (missing opportunity_group_id): {n_skipped_missing_id:,}\n"
    f"Results saved to: {filepath_write}"
)
