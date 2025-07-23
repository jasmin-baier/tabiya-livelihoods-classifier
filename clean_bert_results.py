# NOTE: It is important that this script renames extracted_occupations and extracted_skills1 to "potential_occupations" and "potential_skills" for LLM file to work

import pandas as pd
import json
import ast
from typing import List, Dict, Any
from pathlib import Path
import re

# ── LOAD MAPPINGS ────────────────────────────────────────────────────────────
taxonomy_dir = Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/Tabiya ESCO adapted")
OCC_MAP_PATH = taxonomy_dir / "occupations.csv"
SKILL_MAP_PATH = taxonomy_dir / "skills.csv"

def load_id_label_mapping(csv_path: Path) -> dict[str, str]:
    """
    Build a dict that maps every ID *and* every UUID in UUIDHISTORY
    to its PREFERREDLABEL.
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        label = row["PREFERREDLABEL"].strip()

        # primary ID
        primary_id = row["ID"].strip()
        if primary_id:
            mapping[primary_id] = label

        # any UUIDs in UUIDHISTORY (can be 'uuid1;uuid2 uuid3,…')
        for uuid in re.split(r"[;,|\s]+", row["UUIDHISTORY"]):
            uuid = uuid.strip()
            if uuid:
                mapping[uuid] = label

    return mapping

occ_id2label   = load_id_label_mapping(OCC_MAP_PATH)
skill_id2label = load_id_label_mapping(SKILL_MAP_PATH)

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

def parse_potential_occupations(cell) -> List[str]:
    parsed = _safe_load_literal(cell)

    # Try regex fallback first if parsing failed:
    if isinstance(parsed, str):
        return extract_fallback_tokens(parsed, "occupations")

    seen: set[str] = set()
    out:  list[str] = []

    if isinstance(parsed, list):
        for item in parsed:
            for occ in _extract_retrieved_list(item):
                if occ not in seen:       # O(1) lookup
                    seen.add(occ)
                    out.append(occ)       # keep original order
    return out

def parse_potential_skills(cell) -> List[str]:
    parsed = _safe_load_literal(cell)

    if isinstance(parsed, str):
        return extract_fallback_tokens(parsed, "skills")

    seen: set[str] = set()
    out:  list[str] = []

    if isinstance(parsed, list):
        for item in parsed:
            for skill in _extract_retrieved_list(item):
                if skill not in seen:
                    seen.add(skill)
                    out.append(skill)
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
    df.rename(columns={'extracted_skills1': 'potential_skills'}, inplace=True)

    # Display column names for verification
    print(f"CSV columns: {list(df.columns)}")
    
    # Check if potential_skills column exists
    has_skills_column = 'potential_skills' in df.columns
    print(f"Has potential_skills column: {has_skills_column}")
    
    job_data_list = []
    failed_parsing = []
    
    for index, row in df.iterrows():
        try:
            # Extract and clean the data
            job_id = str(row['ReferenceNumber'])
            job_title = str(row['job_title']).strip()
            job_description = str(row['job_description']).strip()
            job_requirements = str(row['job_requirements']).strip()
            
            # Parse the complex potential_occupations field
            occupations = parse_potential_occupations(row['potential_occupations'])
            # Remove duplicates
            occupations = list(dict.fromkeys(occupations))  # preserves order
            
            # ADDED: Parse potential_skills if the column exists
            skills = []
            if has_skills_column and pd.notna(row['potential_skills']):
                skills = parse_potential_skills(row['potential_skills'])
            elif has_skills_column:
                print(f"  ⚠ No skills data for job {job_id}")
            # Remove duplicates
            skills = list(dict.fromkeys(skills))

            # ── map IDs → preferred labels ──────────────────►
            occupation_labels = list(dict.fromkeys(
            occ_id2label.get(occ, f"UNKNOWN_{occ}") for occ in occupations
            ))

            skill_labels = list(dict.fromkeys(
            skill_id2label.get(sk, f"UNKNOWN_{sk}") for sk in skills
            ))
            
            # Create the job data entry
            job_entry = {
                "job_id": job_id,
                "job_title": job_title,
                "job_description": job_description,
                "job_requirements": job_requirements,
                "potential_occupations_uuids": occupations,
                "potential_occupations": occupation_labels 
            }
            
            # ADDED: Include skills in the output if available
            if has_skills_column:
                job_entry["potential_skills_uuids"] = skills
                job_entry["potential_skills"] = skill_labels
            
            job_data_list.append(job_entry)
            print(f"✓ Processed job {job_id}: {job_title}")
            print(f"  Found {len(occupations)} occupations: {occupations}")
            if has_skills_column:
                print(f"  Found {len(skills)} skills: {skills}")
            
        except Exception as e:
            error_info = {
                'row_index': index,
                'job_id': row.get('ReferenceNumber', 'unknown'),
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
            print(f"  - Row {failure['row_index']} (ID: {failure['job_id']}): {failure['error']}")
    
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
            print(f"Job ID: {row['ReferenceNumber']}")
            print(f"Job Title: {row['job_title']}")
            print(f"Job Description: {row['job_description'][:100]}...")
            print(f"Job Requirements: {row['job_requirements'][:100]}...")
            
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
    csv_file_path = base_dir / "2025-07-21_BERT_extracted_occupations_skills_uuid.csv"  # Your CSV file path
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