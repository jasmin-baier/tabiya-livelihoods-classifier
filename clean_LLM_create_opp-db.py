
# TODO Only keep final chosen occupation
# TODO Also merge back on thos jobs that had 0 occupations/skills from bert (if at least one of them was non-zero it should be in LLM files)
# TODO Make sure to merge other job variables back in
# TODO map back to uuids --> but the ones from tabiya taxonomy (in future make robust to changes in ids somehow)
# TODO bring into correct structure as per Miro
# TODO Duplicates: omg just pass the URL through as primary ID — but no even fewer unique one's here...have to check what is going on
# TODO maybe in the full RCT keep "required" and "important" skills separate
# TODO for now created_at and updated_at are just today's date. Make script more sophisticated by recognizing which opportunities we are actually adding/updating and which already existed.
# TODO consider filtering out opportunities that are skills trainings; step 7 has a first approach
# TODO: Some of the opportunities seem hallucinated, as this file cannot find uuids for them. Will probably have to do some re-runs here as well, similar to how I did them for skills.
# TODO: This file now set all active = True, but code exists and is just commented out to actually check the status of the opportunity

# NOTE: In the main function "restructure_job_data" I can set remove_null_skill_uuids = True or False, depending on whether I am running this to find jobs for my bert-rerun or to share a final database with the tech team.

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def restructure_job_data(occupations_json_path, skills_json_path, skills_entity_path, occupations_entity_path, extra_data_path, output_path, remove_null_skill_uuids=False):
    """
    Restructures and merges job data from multiple sources, adding temporal fields.

    This function takes paths to two JSON files containing ranked occupations and skills,
    CSV files for entity lookups (skills and occupations), and a CSV with additional
    job data. It merges all data, removes duplicates, filters by certification type,
    adds 'active', 'created_at', and 'updated_at' fields, and produces a single,
    restructured JSON file as the output.

    Args:
        occupations_json_path (str or Path): Path to the occupations JSON file.
        skills_json_path (str or Path): Path to the skills JSON file.
        skills_entity_path (str or Path): Path to the skills entity CSV file.
        occupations_entity_path (str or Path): Path to the occupations entity CSV file.
        extra_data_path (str or Path): Path to the CSV file with additional job data.
        output_path (str or Path): Path to save the final restructured JSON file.
        remove_null_skill_uuids (bool): If True, removes skills from opportunities that
                                       could not be mapped to a UUID. Defaults to False.
    """
    # 1. Load all the necessary files into pandas DataFrames.
    try:
        with open(occupations_json_path, 'r', encoding='utf-8') as f:
            occupations_data = json.load(f)
        with open(skills_json_path, 'r', encoding='utf-8') as f:
            skills_data = json.load(f)

        skills_df = pd.read_csv(skills_entity_path)
        occupations_df = pd.read_csv(occupations_entity_path)
        extra_data_df = pd.read_csv(extra_data_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # 2. Create mappings from preferred labels to UUIDs for faster lookups.
    # Drop rows where key columns are missing to prevent errors.
    skills_df.dropna(subset=['UUIDHISTORY', 'PREFERREDLABEL'], inplace=True)
    occupations_df.dropna(subset=['UUIDHISTORY', 'PREFERREDLABEL'], inplace=True)

    # Split the uuidhistory string and take the last element.
    skill_to_uuid = pd.Series(
        skills_df['UUIDHISTORY'].str.split('\n').str[-1].values,
        index=skills_df.PREFERREDLABEL
    ).to_dict()

    occupation_to_uuid = pd.Series(
        occupations_df['UUIDHISTORY'].str.split('\n').str[-1].values,
        index=occupations_df.PREFERREDLABEL
    ).to_dict()

    # 3. Process the occupations data.
    processed_occupations = []
    for record in occupations_data:
        if 'ranked_occupations' in record and record['ranked_occupations']:
            highest_rank_occupation = next((occ for occ in record['ranked_occupations'] if occ.get('rank') == 1), None)
            if highest_rank_occupation:
                occupation_label = highest_rank_occupation.get('occupation')
                processed_occupations.append({
                    "opportunity_group_id": record.get("opportunity_group_id"),
                    "opportunity_ref_id": record.get("opportunity_ref_id"),
                    "opportunity_title": record.get("opportunity_title"),
                    "occupation": {
                        "preferred_label": occupation_label,
                        "uuid": occupation_to_uuid.get(occupation_label)
                    }
                })
    processed_occupations_df = pd.DataFrame(processed_occupations)

    # 4. Process the skills data.
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
            "opportunity_title": record.get("opportunity_title"),
            "skills": skills_list
        })
        
    # 4.5 [OPTIONAL] Remove skills with null UUIDs based on the function argument.
    if remove_null_skill_uuids:
        print("Optional step enabled: Removing skills with null UUIDs.")
        total_skills_removed = 0
        for record in processed_skills:
            initial_skill_count = len(record['skills'])
            # Re-assign the filtered list of skills back to the record.
            record['skills'] = [skill for skill in record['skills'] if pd.notna(skill.get('uuid'))]
            skills_removed_in_record = initial_skill_count - len(record['skills'])
            total_skills_removed += skills_removed_in_record
        print(f"Total skills removed due to null UUID: {total_skills_removed}")
    
    processed_skills_df = pd.DataFrame(processed_skills)

    # 5. Merge the processed data.
    id_columns = ["opportunity_group_id", "opportunity_ref_id", "opportunity_title"]
    merged_df = pd.merge(processed_occupations_df, processed_skills_df, on=id_columns, how="left")
    
    for col in id_columns:
        if col in extra_data_df.columns:
            extra_data_df[col] = extra_data_df[col].astype(str)
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype(str)
            
    final_df = pd.merge(extra_data_df, merged_df, on=id_columns, how="left")

    # 6. Remove duplicate opportunities based on the three ID columns.
    initial_rows_pre_dedupe = len(final_df)
    final_df.drop_duplicates(subset=id_columns, keep='first', inplace=True)
    print(f"Removed {initial_rows_pre_dedupe - len(final_df)} duplicate opportunities. Remaining rows: {len(final_df)}")

    # 7. TODO maybe: Filter out non-accredited certifications from the final merged data, to only keep actual jobs. But not ideal way to filter, and maybe we want to keep skill building opportunities but remove required skills here?
    #if "certification_type" in final_df.columns:
    #    initial_rows_pre_filter = len(final_df)
    #    final_df = final_df[final_df["certification_type"] != "Non-accredited certification"].copy()
    #    print(f"Filtered out {initial_rows_pre_filter - len(final_df)} non-accredited certifications. Final rows: {len(final_df)}")
    #else:
    #    print("Warning: 'certification_type' column not found in final_df. No filtering applied.")


    # 8. Add time-based and status fields.
    now = datetime.now()
    final_df['created_at'] = now.isoformat()
    final_df['updated_at'] = now.isoformat()

    date_posted = pd.to_datetime(final_df['date_posted'], errors='coerce')
    date_closing = pd.to_datetime(final_df['date_closing'], errors='coerce')
    #final_df['active'] = ((date_posted <= now) & (date_closing >= now)).fillna(False)
    final_df['active'] = True

    # 9. Save the final restructured data to a JSON file.
    final_df.to_json(output_path, orient='records', indent=4)
    print(f"Successfully restructured data and saved to {output_path}")


if __name__ == '__main__':
    # Define file paths
    base_dir = Path.home() / "OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass"
    tabiya_taxonomy_dir = base_dir / "Tabiya South Africa v1.0.0"
    data_dir = base_dir / "data"

    occupations_json_path = data_dir / "pre_study/job_responses_occupations_version-oppdescskillsno.json" 
    skills_json_path = data_dir / "pre_study/job_responses_skills_version-oppdescskillsno.json"
    
    skills_entity_path = tabiya_taxonomy_dir / "skills.csv"
    occupations_entity_path = tabiya_taxonomy_dir / "occupations.csv"
    extra_data_path = data_dir / "pre_study/harambee_jobs_clean.csv"
    
    output_path = data_dir / "pre_study/pilot_opportunity_database.json"

    # Run the main function
    # To remove skills with null UUIDs, set remove_null_skill_uuids=True
    # To keep all skills regardless of UUID mapping, set it to False or remove the argument
    restructure_job_data(
        occupations_json_path=occupations_json_path,
        skills_json_path=skills_json_path,
        skills_entity_path=skills_entity_path,
        occupations_entity_path=occupations_entity_path,
        extra_data_path=extra_data_path,
        output_path=output_path,
        remove_null_skill_uuids=False # <-- TODO Set this to True or False as needed
    )