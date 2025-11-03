# NOTE: It is important that this script renames extracted_occupations and extracted_skills1 to "potential_occupations" and "potential_skills" for LLM file to work

import pandas as pd
import io
import json
import ast
from typing import List, Dict, Any, Tuple
from pathlib import Path
import re

# TODO: still consider adding all skills from occupation --> has to happen BEFORE the LLM step. This can be an alternative opp_db, similar how the skills_group version is an alternative opp_db
# TODO: Now still using extracted_skills1, check if requirements are just a subset of skills 1; if yes OR more comprehensive, use extracted_skills2 & skill_requirements separately
# TODO I have currently commented out all extra fields, in two places (e.g. date_posted etc) --> change once I decide how to pull these fields through the whole pipeline
# TODO: opportunity DB should also have: when posted/when ends + and indicator if at date = today it is still relevant; or if it has been deleted (then also not relevant) --> for study consider keeping all relevant all the time to have larger number of jobs / have function that ensures there are at least 1000 jobs to compare to
# TODO: opportunity_ref_ids are not unique, I always need both opportunity_group_id and opportunity_ref_id
# TODO output doesn't yet actually have all the columns I want

# PRIORITY
# TODO: NOTE I need to make sure to actually just the original id; ideally I want to keep both though; but matching should use the origin; or have only origin_id but then have metadata information which taxonomy was used [for future] --> metadata should be: model/taxonomy id, uuid of the model
# NOTE push for compass to actually use south african taxonomy for main study

# NOTE  that BERT taxonomy used oldest esco uuids (oldest in Compass uuid history)

# TODO think about qualifications and opportunity_requirements
# Current judgement: Probably stay away from it, and only talk about skills matches, clearly state that it doesn't take qualifications into account
# Considerations if I did want to include qualifications
# if it mentions matric, manually add it as requirement, bert likely won't understand
# I have to map requirements to qualifications, but our taxonomy only has skills and occupations? Couldn't find secondary school certificate for example
# How can I tell the system that if someone has upper secondary qualification, they also have all of the qualifications below
# possibly have to add South African qulifications manually, similar to matric bit in; opportunity_requirements can get quite complicated though (include employment status, criminal record, driver's license, own car, matric only a few months go etc.)
# South African NQF doesn't perfectly map onto EQF, since EQF only has 8 levels: https://www.saqa.org.za/wp-content/uploads/2023/02/National-Qualifications-Framework.pdf

# NOTE: It is important that this script renames extracted_occupations and extracted_skills1 to "potential_occupations" and "potential_skills" for LLM file to work

# ============================================================================
# CONFIGURATION & DATA LOADING
# ============================================================================

# --- Define File Paths ---
# Tabiya South Africa Taxonomy (for labels, descriptions, etc.)
tabiya_taxonomy_dir = Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/Tabiya South Africa v1.0.0")
SKILLS_FULL_ENTITY_PATH = tabiya_taxonomy_dir / "skills.csv"
OCCUPATIONS_FULL_ENTITY_PATH = tabiya_taxonomy_dir / "occupations.csv"
SKILLHIERARCHY_PATH = tabiya_taxonomy_dir / "skill_hierarchy.csv"
SKILLGROUP_PATH = tabiya_taxonomy_dir / "skill_groups.csv"
OCCHIERARCHY_PATH = tabiya_taxonomy_dir / "occupation_hierarchy.csv"
OCCGROUP_PATH = tabiya_taxonomy_dir / "occupation_groups.csv"


# --- Input/Output for the main transformation ---
# This assumes the script is run from a location where it can see the data directory
base_dir = Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study")
BERT_RESULTS_CSV_PATH = base_dir / "BERT_extracted_occupations_skills_uuid.csv"
OUTPUT_JSON_PATH = base_dir / "bert_cleaned_withgroups.json"


# A regular expression to robustly identify any UUID in a string.
UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', re.I)

def build_identifier_map(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Builds a dictionary that maps every found ID and UUID to its entire record (row).
    This allows for flexible lookups of any data field (e.g., DESCRIPTION) for a given ID.
    """
    if not csv_path.exists():
        print(f"Warning: File not found at {csv_path}. Skipping.")
        return {}
    
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    print(f"Processing data from '{csv_path.name}'...")

    possible_id_cols = ["ID", "uuid"]
    id_col_name = next((col for col in possible_id_cols if col in df.columns), None)
    if id_col_name:
        print(f"Found primary id column '{id_col_name}'")

    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        record = row.to_dict()
        all_ids_in_row = set()

        for cell_value in row.values:
            if isinstance(cell_value, str):
                found_uuids = UUID_PATTERN.findall(cell_value)
                for uuid in found_uuids:
                    all_ids_in_row.add(uuid.lower())
        
        if id_col_name:
            primary_id = row[id_col_name].strip()
            if primary_id:
                all_ids_in_row.add(primary_id)

        for uid in all_ids_in_row:
            if uid:
                mapping[uid] = record
    return mapping

def build_hierarchy_map(csv_path: Path) -> Dict[str, str]:
    """
    Builds a direct mapping from a child ID to its parent ID from a hierarchy CSV.
    """
    if not csv_path.exists():
        print(f"Warning: Hierarchy file not found at {csv_path}. Skipping.")
        return {}
    
    df = pd.read_csv(csv_path, dtype=str)
    print(f"Processing hierarchy from '{csv_path.name}'...")
    hierarchy_map = {}
    if 'CHILDID' in df.columns and 'PARENTID' in df.columns:
        df.dropna(subset=['CHILDID', 'PARENTID'], inplace=True)
        hierarchy_map = pd.Series(df.PARENTID.values, index=df.CHILDID).to_dict()
    else:
        print(f"Warning: 'CHILDID' or 'PARENTID' columns not found in {csv_path.name}")
    
    return hierarchy_map

# ============================================================================
# PARSING HELPER FUNCTIONS
# ============================================================================

def _safe_load_literal(cell: Any):
    """Try JSON, then literal_eval, else return the raw cell."""
    if not isinstance(cell, str):
        return cell
    text = cell.strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text

def _extract_retrieved_list(parsed_item: Any) -> List[str]:
    """Robustly pluck the 'retrieved' payload from one list item."""
    if not isinstance(parsed_item, dict):
        return []
    raw = parsed_item.get("retrieved")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [str(v).strip() for v in raw if isinstance(v, str) and v.strip()]
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    return []

def parse_extracted_items(cell: Any) -> List[str]:
    """
    Parses a cell that may contain a JSON/literal list of extracted items.
    """
    parsed = _safe_load_literal(cell)
    if isinstance(parsed, str):
        retrieved_pattern = r"'retrieved':\s*'([^']+)'"
        matches = re.findall(retrieved_pattern, parsed)
        return list(dict.fromkeys([m.strip() for m in matches if m.strip()]))

    out: list[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            out.extend(_extract_retrieved_list(item))
    return list(dict.fromkeys(out))

# ============================================================================
# PRIMARY TRANSFORMATION FUNCTION
# ============================================================================

def _get_group_info(
    item_uuids: List[str],
    item_map: Dict[str, Dict[str, Any]],
    hierarchy_map: Dict[str, str], 
    group_map: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """
    Finds unique group labels and descriptions for item UUIDs, handling
    historical IDs and ensuring label-description pairs are synced.
    """
    group_details = set()
    for hist_uuid in item_uuids:
        item_record = item_map.get(hist_uuid)
        if item_record:
            current_id = item_record.get('ID')
            if current_id:
                parent_id = hierarchy_map.get(current_id)
                if parent_id:
                    group_info = group_map.get(parent_id)
                    if group_info:
                        label = group_info.get('PREFERREDLABEL')
                        description = group_info.get('DESCRIPTION', '') # Default to empty
                        if label:
                            group_details.add((label, description)) # Add as a tuple
    
    if not group_details:
        return [], []

    # Sort by label (the first element in the tuple) to ensure consistent order
    sorted_details = sorted(list(group_details))
    
    # Unzip into separate, synchronized lists
    labels, descriptions = zip(*sorted_details)
    
    return list(labels), list(descriptions)


def transform_csv_to_job_data(
    csv_file_path: str, 
    output_json_path: str,
    occ_map: Dict[str, Dict[str, Any]],
    skill_map: Dict[str, Dict[str, Any]],
    occ_hierarchy_map: Dict[str, str],
    skill_hierarchy_map: Dict[str, str],
    occ_group_map: Dict[str, Dict[str, Any]],
    skill_group_map: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Transforms the raw BERT output CSV into a clean JSON file, enriching it
    with labels, descriptions, and group mappings from the provided taxonomy maps.
    """
    print(f"\nReading CSV file: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
    
    # Rename columns for clarity and consistency
    df.rename(columns={
        'extracted_occupation': 'potential_occupations_raw',
        'extracted_skills2': 'potential_skills_raw',
        'extracted_requirements': 'potential_skill_requirements_raw'
    }, inplace=True)

    # Sets of valid UUIDs for efficient filtering
    valid_occupation_uuids = set(occ_map.keys())
    valid_skill_uuids = set(skill_map.keys())
    
    job_data_list = []
    for index, row in df.iterrows():
        try:
            # --- 1. Parse and Filter UUIDs from raw columns ---
            occupations = [
                occ for occ in parse_extracted_items(row.get('potential_occupations_raw'))
                if occ in valid_occupation_uuids
            ]
            skills = [
                s for s in parse_extracted_items(row.get('potential_skills_raw'))
                if s in valid_skill_uuids
            ]
            skill_requirements = [
                sr for sr in parse_extracted_items(row.get('potential_skill_requirements_raw'))
                if sr in valid_skill_uuids
            ]

            # --- 2. Map UUIDs to Labels and Descriptions ---
            occupation_labels = [occ_map.get(occ, {}).get('PREFERREDLABEL', f"UNKNOWN_{occ}") for occ in occupations]
            occupation_descriptions = [occ_map.get(occ, {}).get('DESCRIPTION', '') for occ in occupations]

            skill_labels = [skill_map.get(sk, {}).get('PREFERREDLABEL', f"UNKNOWN_{sk}") for sk in skills]
            skill_descriptions = [skill_map.get(sk, {}).get('DESCRIPTION', '') for sk in skills]

            skill_requirements_labels = [skill_map.get(sr, {}).get('PREFERREDLABEL', f"UNKNOWN_{sr}") for sr in skill_requirements]
            skill_requirements_descriptions = [skill_map.get(sr, {}).get('DESCRIPTION', '') for sr in skill_requirements]
            
            # --- 3. Map items to their group labels and descriptions ---
            occ_groups, occ_group_descs = _get_group_info(occupations, occ_map, occ_hierarchy_map, occ_group_map)
            skill_groups, skill_group_descs = _get_group_info(skills, skill_map, skill_hierarchy_map, skill_group_map)
            skill_req_groups, skill_req_group_descs = _get_group_info(skill_requirements, skill_map, skill_hierarchy_map, skill_group_map)

            # --- 4. Assemble the final job entry object ---
            job_entry = {
                "opportunity_group_id" : str(row['opportunity_group_id']),
                "opportunity_ref_id" : str(row['opportunity_ref_id']),
                "opportunity_title": str(row['opportunity_title']).strip(),
                "opportunity_description": str(row['opportunity_description']).strip(),
                "opportunity_requirements": str(row['opportunity_requirements']).strip(),
                "full_details": str(row['full_details']).strip(),
                
                "potential_occupations_uuids": occupations,
                "potential_occupations": occupation_labels,
                "potential_occupations_descriptions": occupation_descriptions,
                "potential_occupation_groups": occ_groups,
                "potential_occupation_group_descriptions": occ_group_descs,

                "potential_skills_uuids": skills,
                "potential_skills": skill_labels,
                "potential_skills_descriptions": skill_descriptions,
                "potential_skill_groups": skill_groups,
                "potential_skill_group_descriptions": skill_group_descs,

                "potential_skill_requirements_uuids": skill_requirements,
                "potential_skill_requirements": skill_requirements_labels,
                "potential_skill_requirements_descriptions": skill_requirements_descriptions,
                "potential_skill_requirements_groups": skill_req_groups,
                "potential_skill_requirements_group_descriptions": skill_req_group_descs,
            }
            
            job_data_list.append(job_entry)

        except Exception as e:
            print(f"✗ Failed to process row {index} (ID: {row.get('opportunity_ref_id', 'N/A')}): {e}")
            continue
    
    # --- 5. Save the transformed data to JSON ---
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(job_data_list, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Successfully saved {len(job_data_list)} job entries to {output_json_path}")
    except Exception as e:
        print(f"✗ Error saving JSON file: {e}")
    
    return job_data_list

# ============================================================================
# EXECUTION SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("--- Starting Data Transformation Pipeline ---")
    
    # 1. Build the identifier maps from the taxonomy files
    print("\n[Step 1/3] Building identifier maps from taxonomy files...")
    skill_map = build_identifier_map(SKILLS_FULL_ENTITY_PATH)
    occ_map = build_identifier_map(OCCUPATIONS_FULL_ENTITY_PATH)
    skill_group_map = build_identifier_map(SKILLGROUP_PATH)
    occ_group_map = build_identifier_map(OCCGROUP_PATH)
    
    print("\n[Step 2/3] Building hierarchy maps...")
    skill_hierarchy_map = build_hierarchy_map(SKILLHIERARCHY_PATH)
    occ_hierarchy_map = build_hierarchy_map(OCCHIERARCHY_PATH)

    print(f"\nLoaded {len(skill_map)} skill identifiers and {len(occ_map)} occupation identifiers.")
    print(f"Loaded {len(skill_group_map)} skill group identifiers and {len(occ_group_map)} occupation group identifiers.")
    print(f"Loaded {len(skill_hierarchy_map)} skill hierarchy relations and {len(occ_hierarchy_map)} occupation hierarchy relations.")

    # 3. Run the main transformation
    print("\n[Step 3/3] Transforming raw BERT results...")
    if all([skill_map, occ_map, skill_group_map, occ_group_map, skill_hierarchy_map, occ_hierarchy_map]):
        job_data = transform_csv_to_job_data(
            csv_file_path=BERT_RESULTS_CSV_PATH,
            output_json_path=OUTPUT_JSON_PATH,
            occ_map=occ_map,
            skill_map=skill_map,
            occ_hierarchy_map=occ_hierarchy_map,
            skill_hierarchy_map=skill_hierarchy_map,
            occ_group_map=occ_group_map,
            skill_group_map=skill_group_map
        )
    else:
        print("✗ Halting transformation because one or more maps failed to load.")
        job_data = []

    # 4. Final summary
    print("\n[Step 4/4] Pipeline Finished.")
    if job_data:
        print(f"✓ Transformation complete! Use '{OUTPUT_JSON_PATH}' for your subsequent analysis.")
    else:
        print("\n✗ Transformation failed or produced no data!")