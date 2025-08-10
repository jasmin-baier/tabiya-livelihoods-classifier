
# TODO Only keep final chosen occupation
# TODO Also merge back on thos jobs that had 0 occupations/skills from bert (if at least one of them was non-zero it should be in LLM files)
# TODO Make sure to merge other job variables back in
# TODO map back to uuids --> but the ones from tabiya taxonomy (in future make robust to changes in ids somehow)
# TODO bring into correct structure as per Miro
# TODO Duplicates: omg just pass the URL through as primary ID — but no even fewer unique one's here...have to check what is going on
# TODO maybe in the full RCT keep "required" and "important" skills separate

import json
import pandas as pd
from pathlib import Path

def restructure_job_data(occupations_json_path, skills_json_path, skills_entity_path, occupations_entity_path, extra_data_path, output_path):
    """
    Restructures and merges job data from multiple sources.

    This function takes paths to two JSON files containing ranked occupations and skills,
    CSV files for entity lookups (skills and occupations), and a CSV with additional
    job data. It processes this information and produces a single, restructured
    JSON file as the output.

    Args:
        occupations_json_path (str or Path): Path to the occupations JSON file.
        skills_json_path (str or Path): Path to the skills JSON file.
        skills_entity_path (str or Path): Path to the skills entity CSV file.
        occupations_entity_path (str or Path): Path to the occupations entity CSV file.
        extra_data_path (str or Path): Path to the CSV file with additional job data.
        output_path (str or Path): Path to save the final restructured JSON file.
    """
    # 1. Load all the necessary files into pandas DataFrames.
    # Use a try-except block to handle potential file not found errors.
    try:
        with open(occupations_json_path, 'r') as f:
            occupations_data = json.load(f)
        with open(skills_json_path, 'r') as f:
            skills_data = json.load(f)

        skills_df = pd.read_csv(skills_entity_path)
        occupations_df = pd.read_csv(occupations_entity_path)
        extra_data_df = pd.read_csv(extra_data_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Create mappings from preferred labels to UUIDs for faster lookups.
    # Updated to use 'ID' and 'PREFERREDLABEL' columns.
    skill_to_uuid = pd.Series(skills_df.ID.values, index=skills_df.PREFERREDLABEL).to_dict()
    occupation_to_uuid = pd.Series(occupations_df.ID.values, index=occupations_df.PREFERREDLABEL).to_dict()

    # 2. Process the occupations data.
    # We'll extract the highest-ranked occupation for each job.
    processed_occupations = []
    for record in occupations_data:
        # Check if 'ranked_occupations' exists and is not empty
        if 'ranked_occupations' in record and record['ranked_occupations']:
            # Find the occupation with rank 1
            highest_rank_occupation = next((occ for occ in record['ranked_occupations'] if occ.get('rank') == 1), None)
            if highest_rank_occupation:
                occupation_label = highest_rank_occupation.get('occupation')
                processed_occupations.append({
                    "opportunity_group_id": record.get("opportunity_group_id"),
                    "opportunity_ref_id": record.get("opportunity_ref_id"),
                    "job_title": record.get("opportunity_title"),
                    "occupation": {
                        "preferred_label": occupation_label,
                        "uuid": occupation_to_uuid.get(occupation_label)
                    }
                })

    processed_occupations_df = pd.DataFrame(processed_occupations)

    # 3. Process the skills data.
    # We'll combine required and important skills and restructure them.
    processed_skills = []
    for record in skills_data:
        all_skills = record.get('required_skills', []) + record.get('top_important_skills', [])
        
        skills_list = []
        for skill_label in all_skills:
            skills_list.append({
                "preferred_label": skill_label,
                "uuid": skill_to_uuid.get(skill_label)
            })
        
        processed_skills.append({
            "opportunity_group_id": record.get("opportunity_group_id"),
            "opportunity_ref_id": record.get("opportunity_ref_id"),
            "job_title": record.get("opportunity_title"),
            "skills": skills_list
        })

    processed_skills_df = pd.DataFrame(processed_skills)

    # 4. Merge the processed data.
    # We'll use the three specified ID columns for a robust merge.
    id_columns = ["opportunity_group_id", "opportunity_ref_id", "job_title"]
    
    # First, merge the occupations and skills data.
    merged_df = pd.merge(processed_occupations_df, processed_skills_df, on=id_columns, how="left")
    
    # Next, merge the result with the extra data.
    # We need to ensure the column types are consistent for merging.
    for col in id_columns:
        if col in extra_data_df.columns:
            extra_data_df[col] = extra_data_df[col].astype(str)
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype(str)
            
    final_df = pd.merge(extra_data_df, merged_df, on=id_columns, how="left")

    # 5. Save the final restructured data to a JSON file.
    # The 'records' orientation will create a list of JSON objects, as requested.
    final_df.to_json(output_path, orient='records', indent=4)
    print(f"Successfully restructured data and saved to {output_path}")


if __name__ == '__main__':
    # Define the file paths.
    # Please adjust these paths to match the location of your files.
    
    # NOTE: The user provided paths from their local machine. 
    # For this script to be runnable, these paths need to be adjusted.
    # I will use placeholder names for the JSON files based on the screenshots.
    
    base_dir = Path.home() / "OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass"
    tabiya_taxonomy_dir = base_dir / "Tabiya South Africa v1.0.0"
    data_dir = base_dir / "data"

    # Assuming these are the names of your initial JSON files from the screenshots
    occupations_json_path = data_dir / "pre_study/job_responses_occupations_version-oppdescskillsno.json" 
    skills_json_path = data_dir / "pre_study/job_responses_skills_version-oppdescskillsno.json"
    
    skills_entity_path = tabiya_taxonomy_dir / "skills.csv"
    occupations_entity_path = tabiya_taxonomy_dir / "occupations.csv"
    extra_data_path = data_dir / "pre_study/harambee_jobs_clean.csv"
    
    output_path = data_dir / "pilot_opportunity_database.json"

    # Create dummy files for demonstration if they don't exist
    if not occupations_json_path.exists():
        occupations_json_path.parent.mkdir(parents=True, exist_ok=True)
        # Creating a dummy JSON based on the screenshot for the script to run
        dummy_occupations = [
            {"opportunity_group_id": "1269045", "opportunity_ref_id": "1269045", "opportunity_title": "Intern Property Practitioner Real Estate Agent", "ranked_occupations": [{"occupation": "real estate agent", "rank": 1}, {"occupation": "letting agent", "rank": 2}]},
            {"opportunity_group_id": "1535496", "opportunity_ref_id": "1535496", "opportunity_title": "Data Protection Basics", "ranked_occupations": []}
        ]
        with open(occupations_json_path, 'w') as f:
            json.dump(dummy_occupations, f)

    if not skills_json_path.exists():
        skills_json_path.parent.mkdir(parents=True, exist_ok=True)
        # Creating a dummy JSON based on the screenshot for the script to run
        dummy_skills = [
            {"opportunity_group_id": "1768375", "opportunity_ref_id": "1768375", "opportunity_title": "In - Store Brand Ambassador: Umlazi", "required_skills": ["communicate with customers", "act reliably"], "top_important_skills": []},
            {"opportunity_group_id": "1768376", "opportunity_ref_id": "1768376", "opportunity_title": "Lead Generation Agent", "required_skills": ["act reliably"], "top_important_skills": ["advertising techniques"]}
        ]
        with open(skills_json_path, 'w') as f:
            json.dump(dummy_skills, f)
            
    if not skills_entity_path.exists():
        skills_entity_path.parent.mkdir(parents=True, exist_ok=True)
        # Updated dummy file to use 'ID' and 'PREFERREDLABEL'
        dummy_skills_csv = pd.DataFrame({
            'ID': ['uuid_skill_1', 'uuid_skill_2', 'uuid_skill_3'],
            'PREFERREDLABEL': ['communicate with customers', 'act reliably', 'advertising techniques']
        })
        dummy_skills_csv.to_csv(skills_entity_path, index=False)

    if not occupations_entity_path.exists():
        occupations_entity_path.parent.mkdir(parents=True, exist_ok=True)
        # Updated dummy file to use 'ID' and 'PREFERREDLABEL'
        dummy_occupations_csv = pd.DataFrame({
            'ID': ['uuid_occ_1', 'uuid_occ_2'],
            'PREFERREDLABEL': ['real estate agent', 'letting agent']
        })
        dummy_occupations_csv.to_csv(occupations_entity_path, index=False)

    if not extra_data_path.exists():
        extra_data_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_extra_data = pd.DataFrame({
            'opportunity_group_id': ['1269045', '1768375'],
            'opportunity_ref_id': ['1269045', '1768375'],
            'job_title': ['Intern Property Practitioner Real Estate Agent', 'In - Store Brand Ambassador: Umlazi'],
            'extra_field_1': ['value1', 'value2'],
            'extra_field_2': ['valueA', 'valueB']
        })
        dummy_extra_data.to_csv(extra_data_path, index=False)


    # Run the main function.
    restructure_job_data(
        occupations_json_path=occupations_json_path,
        skills_json_path=skills_json_path,
        skills_entity_path=skills_entity_path,
        occupations_entity_path=occupations_entity_path,
        extra_data_path=extra_data_path,
        output_path=output_path
    )