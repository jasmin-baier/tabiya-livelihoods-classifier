
# TODO Only keep final chosen occupation
# TODO Also merge back on thos jobs that had 0 occupations/skills from bert (if at least one of them was non-zero it should be in LLM files)
# TODO Make sure to merge other job variables back in
# TODO map back to uuids --> but the ones from tabiya taxonomy (in future make robust to changes in ids somehow)
# TODO bring into correct structure as per Miro

import pandas as pd
import json
from pathlib import Path

# --- File Path Definition ---

# Define the base directory where the files are located
base_dir = Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study")

# Create the full, platform-independent paths to your files
occupations_file = base_dir / 'job_responses_occupations_combined.json'
skills_file = base_dir / 'job_responses_skills_combined.json'
harambee_file = base_dir / 'harambee_jobs_clean_subset.csv'

# --- Loading and Initial Cleaning ---

try:
    df_occupations = pd.read_json(occupations_file)
    df_skills = pd.read_json(skills_file)
    harambee_df = pd.read_csv(harambee_file)
except FileNotFoundError as e:
    print(f"Error: Could not find a file. Please check your path and filenames.\n{e}")
    exit()

# Define columns to drop
cols_to_drop = ['error', 'error_details']

# 1. Clean the occupations DataFrame
if 'raw_response' in df_occupations.columns:
    df_occupations.rename(columns={'raw_response': 'error_response_occupation'}, inplace=True)
df_occupations.drop(columns=cols_to_drop, inplace=True, errors='ignore')
df_occupations.rename(columns={'final_choice': 'final_occupation_choice'}, inplace=True)

# 2. Clean the skills DataFrame
if 'raw_response' in df_skills.columns:
    df_skills.rename(columns={'raw_response': 'error_response_skills'}, inplace=True)
df_skills.drop(columns=cols_to_drop, inplace=True, errors='ignore')


# --- Normalization and Merging ---

def normalize_column(series):
    """Converts single-element lists to strings to ensure merge keys are consistent."""
    return series.apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

merge_keys = ['opportunity_group_id', 'opportunity_ref_id', 'opportunity_title']

for col in merge_keys:
    if col in df_occupations.columns:
        df_occupations[col] = normalize_column(df_occupations[col])
    if col in df_skills.columns:
        df_skills[col] = normalize_column(df_skills[col])

# IMPORTANT: Ensure merge key data types match before merging
for col in merge_keys:
    if col in harambee_df.columns:
        harambee_df[col] = harambee_df[col].astype(str)
    if col in df_occupations.columns:
        df_occupations[col] = df_occupations[col].astype(str)
    if col in df_skills.columns:
        df_skills[col] = df_skills[col].astype(str)

# Perform the first merge using 'outer' to keep all records from both files
merged_df = pd.merge(df_occupations, df_skills, on=merge_keys, how='outer')

# Perform the final merge to integrate the harambee data
final_df = pd.merge(merged_df, harambee_df, on=merge_keys, how='outer')


# --- Output ---

# Display the first 20 rows and all columns of the final DataFrame
pd.set_option('display.max_columns', None)
print("Merge complete. Displaying the first 20 rows of the final data:")
print("---------------------------------------------------------------")
print(final_df.head(20))

# Optionally save the final result to a new file
final_df.to_json(base_dir / 'final_opportunity_database.json', orient='records', indent=2)