# NOTE: It is important that this script renames extracted_occupations and extracted_skills1 to "potential_occupations" and "potential_skills" for LLM file to work

import pandas as pd
import json
import ast
from typing import List, Dict, Any
from pathlib import Path
import re

# TODO: Now still using extracted_skills1, check if requirements are just a subset of skills 1; if yes OR more comprehensive, use extracted_skills2 & skill_requirements separately
# TODO I have currently commented out all extra fields, in two places (e.g. date_posted etc) --> change once I decide how to pull these fields through the whole pipeline
# TODO: opportunity DB should also have: when posted/when ends + and indicator if at date = today it is still relevant; or if it has been deleted (then also not relevant) --> for study consider keeping all relevant all the time to have larger number of jobs / have function that ensures there are at least 1000 jobs to compare to
# TODO: opportunity_ref_ids are not unique, I always need both opportunity_group_id and opportunity_ref_id
# TODO output doesn't yet actually have all the columns I want

# TODO Note that BERT taxonomy files don't have uuid history, so need to check if uuids will match correctly with Compass identified uuids

# TODO think about qualifications and opportunity_requirements
# Current judgement: Probably stay away from it, and only talk about skills matches, clearly state that it doesn't take qualifications into account
# Considerations if I did want to include qualifications
# if it mentions matric, manually add it as requirement, bert likely won't understand
# I have to map requirements to qualifications, but our taxonomy only has skills and occupations? Couldn't find secondary school certificate for example
# How can I tell the system that if someone has upper secondary qualification, they also have all of the qualifications below
# possibly have to add South African qulifications manually, similar to matric bit in; opportunity_requirements can get quite complicated though (include employment status, criminal record, driver's license, own car, matric only a few months go etc.)
# South African NQF doesn't perfectly map onto EQF, since EQF only has 8 levels: https://www.saqa.org.za/wp-content/uploads/2023/02/National-Qualifications-Framework.pdf

# ── LOAD MAPPINGS ────────────────────────────────────────────────────────────
taxonomy_dir = Path("C:/Users/jasmi/Documents/GitHub/tabiya-livelihoods-classifier/inference/files")
OCC_MAP_PATH = taxonomy_dir / "occupations_augmented.csv"
SKILL_MAP_PATH = taxonomy_dir / "skills.csv"
#QUAL_MAP_PATH = taxonomy_dir / "qualifications.csv" # decided not to use qualifications for now + anyway this doesn't include South African ones

def load_id_label_mapping(csv_path: Path) -> dict[str, str]:
    """
    Build a dict that maps every ID *and* every UUID in UUIDHISTORY
    to its PREFERREDLABEL.
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # --- Dynamically find the label / id column ---
    possible_label_cols = ["PREFERREDLABEL", "skills", "preffered_label"] # nNote: I know that "preffered_label" is incorrect spelling, but this is what it is in BERT taxonomy
    label_col_name = None
    for col in possible_label_cols:
        if col in df.columns:
            label_col_name = col
            print(f"Found label column '{label_col_name}' in {csv_path.name}")
            break  # Stop once a valid column is found

    possible_id_cols = ["ID", "uuid"]
    id_col_name = None
    for col in possible_id_cols:
        if col in df.columns:
            id_col_name = col
            print(f"Found id column '{id_col_name}' in {csv_path.name}")
            break  # Stop once a valid column is found

    # Raise an error if no suitable label column is found
    if label_col_name is None:
        raise ValueError(
            f"Could not find a label column in {csv_path.name}. "
            f"Expected one of: {possible_label_cols}"
        )
    
    if id_col_name is None:
        raise ValueError(
            f"Could not find a label column in {csv_path.name}. "
            f"Expected one of: {possible_id_cols}"
        )

    # No map to correct columns
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        label = row[label_col_name].strip()

        # primary ID
        primary_id = row[id_col_name].strip()
        if primary_id:
            mapping[primary_id] = label

        # any UUIDs in UUIDHISTORY (can be 'uuid1;uuid2 uuid3,…') -- BERT taxonomy files don't have uuid history
        #for uuid in re.split(r"[;,|\s]+", row["UUIDHISTORY"]):
        #    uuid = uuid.strip()
        #    if uuid:
        #        mapping[uuid] = label

    return mapping

def load_valid_uuids(csv_path: Path, column_name: str = "uuid") -> set:
    """Loads a specific column from a CSV into a set for fast lookups."""
    try:
        df = pd.read_csv(csv_path, dtype=str, usecols=[column_name])
        return set(df[column_name].dropna())
    except Exception as e:
        print(f"Error loading valid UUIDs from {csv_path}: {e}")
        return set()

occ_id2label   = load_id_label_mapping(OCC_MAP_PATH)
skill_id2label = load_id_label_mapping(SKILL_MAP_PATH)

# Load the valid UUIDs into sets for efficient validation of parsed uuids
print("\nLoading valid UUIDs for filtering...")
valid_occupation_uuids = load_valid_uuids(OCC_MAP_PATH, column_name="uuid")
valid_skill_uuids      = load_valid_uuids(SKILL_MAP_PATH, column_name="uuid")
print(f"Found {len(valid_occupation_uuids)} valid occupation UUIDs.")
print(f"Found {len(valid_skill_uuids)} valid skill UUIDs.")

# ── PARSE FUNCTIONS ──────────────────────────────────────────────────────────
def _normalise_retrieved(value: Any) -> List[str]:
    """Return a flat list of strings, whatever shape the input has."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
            elif isinstance(v, list):
                out.extend(_normalise_retrieved(v))     # flatten nested lists
        return out
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []

def _safe_load_literal(cell: Any):
    """Try JSON, then literal_eval, else return the raw cell."""
    if not isinstance(cell, str):
        return cell
    text = cell.strip()
    # empty string ➜ nothing to parse
    if not text:
        return []
    # 1️⃣ try JSON first (double-quoted, strict)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2️⃣ fall back to Python literal (single quotes, etc.)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text   # give back the raw string so we can regex it later
    
def _extract_retrieved_list(parsed_item: Any) -> List[str]:
    """Robustly pluck the 'retrieved' payload from one list item."""
    if not isinstance(parsed_item, dict):
        return []
    raw = parsed_item.get("retrieved")
    return _normalise_retrieved(raw)

def parse_extracted_items(cell: Any, data_type: str) -> List[str]:
    """
    Parses a cell that may contain a JSON/literal list of extracted items.
    Handles complex nested structures from BERT output.
    """
    parsed = _safe_load_literal(cell)

    if isinstance(parsed, str):
        # The fallback can use the data_type for better error messages
        return extract_fallback_tokens(parsed, data_type)

    seen: set[str] = set()
    out:  list[str] = []

    if isinstance(parsed, list):
        for item in parsed:
            for extracted_item in _extract_retrieved_list(item):
                if extracted_item not in seen:
                    seen.add(extracted_item)
                    out.append(extracted_item)
    return out

def extract_fallback_tokens(retrieved_string: str, data_type: str) -> List[str]:
    """
    Fallback method to extract retrieved names when parsing fails
    """
    # Look for common patterns in the retrieved field
    import re
    
    # Find all 'retrieved': 'retrieved_name' patterns
    retrieved_pattern = r"'retrieved':\s*'([^']+)'"
    matches = re.findall(retrieved_pattern, retrieved_string)
    
    # Clean and deduplicate
    retrieved = []
    for match in matches:
        cleaned = match.strip()
        if cleaned and len(cleaned) > 1 and cleaned not in retrieved:
            retrieved.append(cleaned)
    
    return retrieved if retrieved else [f"Unable to parse {data_type}"]

# ── PRIMARY FUNCTION putting it all together ──────────────────────────────────────────────────────────
def transform_csv_to_job_data(csv_file_path: str, output_json_path: str) -> List[Dict[str, Any]]:
    """
    Transform the CSV data into the format expected by the job processing code
    """
    print(f"Reading CSV file: {csv_file_path}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
    
    # First, rename the columns to match the expected format in later LLM reranking
    df.rename(columns={'extracted_occupation': 'potential_occupations'}, inplace=True)
    df.rename(columns={'extracted_skills2': 'potential_skills'}, inplace=True)
    df.rename(columns={'extracted_requirements': 'potential_skill_requirements'}, inplace=True)

    # Display column names for verification
    print(f"CSV columns: {list(df.columns)}")
    
    # Check if potential_skills column exists
    has_occupations_column = 'potential_occupations' in df.columns
    has_skills_column = 'potential_skills' in df.columns
    has_skill_requirements_column = 'potential_skill_requirements' in df.columns
    
    job_data_list = []
    failed_parsing = []
    
    for index, row in df.iterrows():
        try:
            # Extract and clean the data
            # TODO consider making this more efficient at some point
            opportunity_group_id = str(row['opportunity_group_id'])
            opportunity_ref_id = str(row['opportunity_ref_id'])
            opportunity_title = str(row['opportunity_title']).strip()
            opportunity_description = str(row['opportunity_description']).strip()
            opportunity_requirements = str(row['opportunity_requirements']).strip()
            full_details = str(row['full_details']).strip()
            #company_name = str(row['company_name']).strip()
            #contract_type = str(row['contract_type']).strip()
            #date_posted = str(row['date_posted']).strip()
            #date_closing = str(row['date_closing']).strip()
            #certification_type = str(row['certification_type']).strip()
            #city = str(row['city']).strip()
            #province = str(row['province']).strip()
            #latitude = str(row['latitude']).strip()
            #longitude = str(row['longitude']).strip()
            #salary_type = str(row['salary_type']).strip()
            #salary = str(row['salary']).strip()
            #opportunity_duration = str(row['opportunity_duration']).strip()
            #is_online = str(row['is_online']).strip()
            #opportunity_url = str(row['opportunity_url']).strip()
            
            # --- PARSE EACH COLUMN ---
            # Parse occupations
            occupations = []
            if has_occupations_column and pd.notna(row['potential_occupations']):
                occupations = parse_extracted_items(row['potential_occupations'], "occupations")
            # Remove duplicates
            occupations = list(dict.fromkeys(occupations))

            # Parse skills
            skills = []
            if has_skills_column and pd.notna(row['potential_skills']):
                skills = parse_extracted_items(row['potential_skills'], "skills")
            # Remove duplicates
            skills = list(dict.fromkeys(skills))

            # Parse skill requirements
            skill_requirements = []
            if has_skill_requirements_column and pd.notna(row['potential_skill_requirements']):
                skill_requirements = parse_extracted_items(row['potential_skill_requirements'], "skill requirements")
            # Remove duplicates
            skill_requirements = list(dict.fromkeys(skill_requirements))

            # --- FILTER - Keep only the UUIDs that exist in the respective taxonomy files ---
            occupations = [occ for occ in occupations if occ in valid_occupation_uuids]
            skills = [s for s in skills if s in valid_skill_uuids]
            skill_requirements = [sr for sr in skill_requirements if sr in valid_skill_uuids]

            # --- map IDs → preferred labels ---
            occupation_labels = list(dict.fromkeys(
            occ_id2label.get(occ, f"UNKNOWN_{occ}") for occ in occupations
            ))

            skill_labels = list(dict.fromkeys(
            skill_id2label.get(sk, f"UNKNOWN_{sk}") for sk in skills
            ))

            skill_requirements_labels = list(dict.fromkeys(
            skill_id2label.get(skrq, f"UNKNOWN_{skrq}") for skrq in skill_requirements
            ))            
            
            # Create the job data entry
            job_entry = {
                "opportunity_group_id" : opportunity_group_id,
                "opportunity_ref_id" : opportunity_ref_id,
                "opportunity_title": opportunity_title,
                "opportunity_description": opportunity_description,
                "opportunity_requirements": opportunity_requirements,
                "full_details": full_details,
                "potential_occupations_uuids": occupations,
                "potential_occupations": occupation_labels,
                "potential_skills_uuids": skills,
                "potential_skills": skill_labels,
                "potential_skill_requirements_uuids": skill_requirements,
                "potential_skill_requirements": skill_requirements_labels,
                #"company_name": company_name,
                #"contract_type": contract_type,
                #"date_posted": date_posted,
                #"date_closing": date_closing,
                #"certification_type" : certification_type,
                #"city" : city,
                #"province" : province,
                #"latitude" : latitude,
                #"longitude" : longitude,
                #"salary_type" : salary_type,
                #"salary" : salary,
                #"opportunity_duration" : opportunity_duration,
                #"is_online" : is_online,
                #"opportunity_url" : opportunity_url
            }
            
            job_data_list.append(job_entry)
            print(f"✓ Processed job {opportunity_group_id} - {opportunity_ref_id}: {opportunity_title}")
            print(f"  Found {len(occupations)} occupations: {occupations}")
            if has_skills_column:
                print(f"  Found {len(skills)} skills: {skills}")
            
        except Exception as e:
            error_info = {
                'row_index': index,
                'opportunity_group_id': row.get('opportunity_group_id', 'unknown'),
                'opportunity_ref_id': row.get('opportunity_ref_id', 'unknown'),
                'error': str(e)
            }
            failed_parsing.append(error_info)
            print(f"✗ Failed to process row {index}: {e}")
            continue
    
    # Save the transformed data to JSON
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(job_data_list, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Successfully saved {len(job_data_list)} job entries to {output_json_path}")
    except Exception as e:
        print(f"✗ Error saving JSON file: {e}")
        return job_data_list
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRANSFORMATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows in CSV: {len(df)}")
    print(f"Successfully processed: {len(job_data_list)}")
    print(f"Failed to process: {len(failed_parsing)}")
    print(f"Skills column included: {has_skills_column}")
    
    if failed_parsing:
        print(f"\nFailed rows:")
        for failure in failed_parsing:
            print(f"  - Row {failure['row_index']} (GroupID: {failure['opportunity_group_id']}, opportunity_ref_id: {failure['opportunity_ref_id']}): {failure['error']}")
    
    print(f"{'='*60}")
    
    return job_data_list

def preview_transformation(csv_file_path: str, num_rows: int = 3):
    """
    Preview how the first few rows will be transformed without saving
    """
    print(f"PREVIEW: Transforming first {num_rows} rows from {csv_file_path}")
    print("="*60)
    
    try:
        df = pd.read_csv(csv_file_path)
        
        # Check if skills column exists
        has_skills = 'potential_skills' in df.columns
        print(f"Has potential_skills column: {has_skills}")
        
        for i in range(min(num_rows, len(df))):
            row = df.iloc[i]
            print(f"\nROW {i+1}:")
            print(f"Job ID: {row['opportunity_ref_id']}")
            print(f"Job Title: {row['opportunity_title']}")
            print(f"Job Description: {row['opportunity_description'][:100]}...")
            print(f"Job Requirements: {row['opportunity_requirements'][:100]}...")
            
            # Parse occupations
            occupations = parse_potential_occupations(row['potential_occupations'])
            print(f"Parsed Occupations ({len(occupations)}): {occupations}")
            
            # ADDED: Parse skills if column exists
            if has_skills and pd.notna(row['potential_skills']):
                skills = parse_potential_skills(row['potential_skills'])
                print(f"Parsed Skills ({len(skills)}): {skills}")
            elif has_skills:
                print(f"Parsed Skills (0): No skills data available")
            
            print("-" * 40)
            
    except Exception as e:
        print(f"Error in preview: {e}")

# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    # Set the base directory
    base_dir = Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study")

    # File paths
    csv_file_path = base_dir / "BERT_extracted_occupations_skills_uuid.csv"  # Your CSV file path  
    output_json_path = base_dir / "bert_cleaned.json"  # Output JSON file
  
    # First, preview the transformation
    print("PREVIEWING TRANSFORMATION...")
    preview_transformation(csv_file_path, num_rows=3)
    
    # Then do the full transformation
    print("\n\nFULL TRANSFORMATION...")
    job_data = transform_csv_to_job_data(csv_file_path, output_json_path)
    
    if job_data:
        print(f"\n✓ Transformation complete! Use '{output_json_path}' with your job processing code.")
    else:
        print("\n✗ Transformation failed!")