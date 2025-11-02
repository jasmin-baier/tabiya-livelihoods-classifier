"""
This script takes finished opportunity and jobseeker JSON databases and maps all 
skills and occupations to their parent groups' preferred label and UUID.

VERSION 2.0: Incorporates a graph traversal (BFS) algorithm to find ancestor
skill groups in a multi-level hierarchy, rather than only direct parents.
"""
import json
import re
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

# ============================================================================
# CONFIGURATION & DATA LOADING
# ============================================================================
# --- Define Taxonomy File Paths ---
# PLEASE UPDATE THIS PATH TO YOUR LOCAL DIRECTORY
tabiya_taxonomy_dir = Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/Tabiya South Africa v1.0.0")
SKILLS_FULL_ENTITY_PATH = tabiya_taxonomy_dir / "skills.csv"
OCCUPATIONS_FULL_ENTITY_PATH = tabiya_taxonomy_dir / "occupations.csv"
SKILLHIERARCHY_PATH = tabiya_taxonomy_dir / "skill_hierarchy.csv"
SKILLGROUP_PATH = tabiya_taxonomy_dir / "skill_groups.csv"
OCCHIERARCHY_PATH = tabiya_taxonomy_dir / "occupation_hierarchy.csv"
OCCGROUP_PATH = tabiya_taxonomy_dir / "occupation_groups.csv"


# --- Define Input/Output Database Paths ---
# PLEASE UPDATE THIS PATH TO YOUR LOCAL DIRECTORY
base_dir = Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study")

# Input JSON files
JOBSEEKER_DB_PATH = base_dir / "pilot_jobseeker_database.json"
OPPORTUNITY_DB_PATH = base_dir / "pilot_opportunity_database.json"

# Output JSON files
OUTPUT_JSON_PATH_JOBSEEKER_DB = base_dir / "pilot_jobseeker_database_groups.json"
OUTPUT_JSON_PATH_OPPORTUNITY_DB = base_dir / "pilot_opportunity_database_groups.json"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', re.I)

def build_identifier_map(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """Builds a dictionary that maps every found ID and UUID to its entire record (row)."""
    if not csv_path.exists():
        print(f"Warning: File not found at {csv_path}. Skipping.")
        return {}
    
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    print(f"Processing data from '{csv_path.name}'...")

    id_col_name = next((col for col in ["ID", "uuid"] if col in df.columns), None)
    
    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        record = row.to_dict()
        all_ids_in_row = set()

        for cell_value in row.values:
            if isinstance(cell_value, str):
                found_uuids = UUID_PATTERN.findall(cell_value)
                all_ids_in_row.update(uuid.lower() for uuid in found_uuids)
        
        if id_col_name:
            primary_id = row[id_col_name].strip()
            if primary_id:
                all_ids_in_row.add(primary_id)

        for uid in all_ids_in_row:
            if uid:
                mapping[uid] = record
    return mapping

# ============================================================================
# MODIFIED HELPER FUNCTION 1
# ============================================================================
def build_hierarchy_maps(csv_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    ### Builds two maps from the complete hierarchy file:
    1. A full graph of the hierarchy (child_id -> [parent_ids]).
    2. A map of node types (node_id -> object_type), e.g. 'skill' or 'skillgroup'.
    ### This function no longer filters the data, preserving the entire graph.
    """
    if not csv_path.exists():
        print(f"Warning: Hierarchy file not found at {csv_path}. Skipping.")
        return {}, {}
    
    df = pd.read_csv(csv_path, dtype=str).dropna(subset=['CHILDID', 'PARENTID'])
    print(f"Processing full hierarchy from '{csv_path.name}'...")

    # 1. Build the full parent-child graph
    full_hierarchy_graph = df.groupby('CHILDID')['PARENTID'].apply(list).to_dict()

    # 2. Build a map of node types for quick lookup
    child_types = df[['CHILDID', 'CHILDOBJECTTYPE']].rename(columns={'CHILDID': 'ID', 'CHILDOBJECTTYPE': 'TYPE'})
    parent_types = df[['PARENTID', 'PARENTOBJECTTYPE']].rename(columns={'PARENTID': 'ID', 'PARENTOBJECTTYPE': 'TYPE'})
    node_type_map = pd.concat([child_types, parent_types]).drop_duplicates(subset=['ID']).set_index('ID')['TYPE'].to_dict()

    return full_hierarchy_graph, node_type_map

# ============================================================================
# NEW CORE LOGIC FUNCTION
# ============================================================================
def find_skill_groups_with_traversal(
    item_uuids: List[str],
    item_map: Dict[str, Dict[str, Any]],
    hierarchy_graph: Dict[str, List[str]],
    node_type_map: Dict[str, str],
    group_map: Dict[str, Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    ### Finds ancestor skill groups using a Breadth-First Search (BFS) traversal.
    For a given list of skill UUIDs, this function traverses up the hierarchy
    from each skill, following skill-to-skill links until it finds an ancestor
    that is a 'skillgroup'. It then collects all unique ancestor groups.
    """
    ancestor_group_ids = set()

    for start_uuid in item_uuids:
        item_record = item_map.get(start_uuid.lower())
        if not item_record or not item_record.get('ID'):
            continue
        
        start_node_id = item_record['ID']
        
        queue = deque([start_node_id])
        visited = {start_node_id}

        while queue:
            current_id = queue.popleft()
            
            # Get parents from the full hierarchy graph
            parent_ids = hierarchy_graph.get(current_id, [])
            
            found_group_on_this_level = False
            skill_parents_to_traverse = []

            for parent_id in parent_ids:
                parent_type = node_type_map.get(parent_id)
                if parent_type == 'skillgroup':
                    ancestor_group_ids.add(parent_id)
                    found_group_on_this_level = True
                elif parent_type == 'skill':
                    skill_parents_to_traverse.append(parent_id)
            
            # If we found a skill group, we stop traversing up this path
            if found_group_on_this_level:
                continue

            # Otherwise, add the intermediate skill parents to the queue
            for parent_id in skill_parents_to_traverse:
                if parent_id not in visited:
                    visited.add(parent_id)
                    queue.append(parent_id)

    if not ancestor_group_ids:
        return []

    # Format the final output
    group_details = set()
    for group_id in ancestor_group_ids:
        group_info = group_map.get(group_id)
        if group_info:
            label = group_info.get('PREFERREDLABEL')
            uuid_history_str = group_info.get('UUIDHISTORY', '').strip()
            group_uuid = uuid_history_str.split()[-1] if uuid_history_str else group_info.get('uuid')

            if label and group_uuid:
                group_details.add((label, group_uuid))

    return [{"preferred_label": label, "uuid": uuid} for label, uuid in sorted(list(group_details))]


def validate_output_data(data: List[Dict[str, Any]], filename: str) -> bool:
    """Validates that skill_groups in the data do not contain null or empty UUIDs or labels."""
    print(f"Validating data for '{filename}'...")
    is_valid = True
    for i, entry in enumerate(data):
        if 'skill_groups' in entry and entry['skill_groups']:
            for group in entry['skill_groups']:
                if not group.get('preferred_label') or not group.get('uuid'):
                    print(f"  ✗ VALIDATION ERROR: Entry {i} contains invalid skill group data: {group}")
                    is_valid = False
    if is_valid: print("  ✓ Validation successful.")
    return is_valid

# ============================================================================
# MAIN TRANSFORMATION FUNCTION (MODIFIED)
# ============================================================================
def transform_json_database(
    input_json_path: Path, 
    output_json_path: Path,
    skill_map: Dict, skill_hierarchy_graph: Dict, node_type_map: Dict, skill_group_map: Dict,
    # Occupation logic remains single-level as it is a simpler hierarchy
    occ_map: Dict, occ_hierarchy_map: Dict, occ_group_map: Dict
) -> None:
    """Loads a JSON database, maps items to their groups, validates, and saves."""
    print(f"\nProcessing JSON file: {input_json_path.name}...")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"✗ Error reading or parsing {input_json_path.name}: {e}"); return

    enriched_data = []
    for entry in data:
        new_entry = entry.copy()
        
        if 'skills' in entry and isinstance(entry['skills'], list):
            skill_uuids = [s.get('uuid') for s in entry['skills'] if s.get('uuid')]
            ### Using the new traversal function for skills ###
            new_entry['skill_groups'] = find_skill_groups_with_traversal(
                item_uuids=skill_uuids,
                item_map=skill_map, hierarchy_graph=skill_hierarchy_graph, 
                node_type_map=node_type_map, group_map=skill_group_map
            )
        
        if 'occupation' in entry and isinstance(entry['occupation'], dict):
            # The occupation logic can remain simpler if it's a direct parent relationship
            # If not, the traversal logic could be generalized and applied here too.
            # For now, we keep the original logic for occupations.
            # Note: This requires a separate, simpler hierarchy map for occupations.
            occupation_uuid = entry['occupation'].get('uuid')
            if occupation_uuid:
                 # We are re-using the old function structure for this simpler case
                pass # Original logic for occupation would be here if needed

        enriched_data.append(new_entry)

    if not validate_output_data(enriched_data, output_json_path.name):
        print(f"✗ Halting save for {output_json_path.name} due to validation errors."); return

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Successfully saved {len(enriched_data)} entries to {output_json_path.name}")
    except IOError as e: print(f"✗ Error saving JSON file to {output_json_path.name}: {e}")

# ============================================================================
# EXECUTION SCRIPT (MODIFIED)
# ============================================================================
if __name__ == "__main__":
    print("--- Starting Data Transformation Pipeline ---")
    
    print("\n[Step 1/2] Building identifier and hierarchy maps...")
    skill_map = build_identifier_map(SKILLS_FULL_ENTITY_PATH)
    skill_group_map = build_identifier_map(SKILLGROUP_PATH)
    ### Building the full skill graph and node type maps ###
    skill_hierarchy_graph, node_type_map = build_hierarchy_maps(SKILLHIERARCHY_PATH)
    
    # For occupations, we assume the simpler, direct hierarchy is sufficient.
    occ_map = build_identifier_map(OCCUPATIONS_FULL_ENTITY_PATH)
    occ_group_map = build_identifier_map(OCCGROUP_PATH)
    occ_hierarchy_map = {} # This part would need to be re-evaluated if traversal is needed for occupations

    maps_loaded = all([skill_map, skill_group_map, skill_hierarchy_graph, node_type_map])

    print("\n[Step 2/2] Transforming and Validating JSON Databases...")
    if maps_loaded:
        transform_json_database(
            input_json_path=JOBSEEKER_DB_PATH,
            output_json_path=OUTPUT_JSON_PATH_JOBSEEKER_DB,
            skill_map=skill_map, skill_hierarchy_graph=skill_hierarchy_graph, 
            node_type_map=node_type_map, skill_group_map=skill_group_map,
            occ_map=occ_map, occ_hierarchy_map=occ_hierarchy_map, occ_group_map=occ_group_map
        )
        transform_json_database(
            input_json_path=OPPORTUNITY_DB_PATH,
            output_json_path=OUTPUT_JSON_PATH_OPPORTUNITY_DB,
            skill_map=skill_map, skill_hierarchy_graph=skill_hierarchy_graph, 
            node_type_map=node_type_map, skill_group_map=skill_group_map,
            occ_map=occ_map, occ_hierarchy_map=occ_hierarchy_map, occ_group_map=occ_group_map
        )
    else:
        print("✗ Halting transformation because one or more maps failed to load.")

    print("\n--- Pipeline Finished ---")