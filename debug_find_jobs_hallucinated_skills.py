import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_rerun_file_from_pilot_db_robust():
    """
    Identifies jobs with null UUIDs from the pilot database, handling both
    list-based and dictionary-based JSON structures. It then creates a 
    new input file from bert_cleaned.json for reprocessing.
    """
    # --- Configuration ---
    base_dir = Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study")
    processed_db_file = base_dir / "pilot_opportunity_database.json" 
    original_input_file = base_dir / "bert_cleaned.json"
    rerun_input_file = base_dir / "bert_cleaned_rerun.json"

    # --- 1. Load the processed database and find the list of jobs ---
    logging.info(f"Loading '{processed_db_file.name}'...")
    job_list = None
    try:
        with open(processed_db_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            logging.info("JSON is a list. Processing directly.")
            job_list = data
        elif isinstance(data, dict):
            logging.info("JSON is a dictionary. Searching for the list of jobs...")
            for key, value in data.items():
                if isinstance(value, list):
                    logging.info(f"Found the job list under the key: '{key}'")
                    job_list = value
                    break
        
        if not job_list:
            logging.error("Could not find a list of jobs within the JSON file. Exiting.")
            return

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Could not load or parse '{processed_db_file.name}'. Error: {e}")
        return

    # --- 2. Identify problematic opportunity_ref_ids from the job list ---
    problematic_ref_ids = set()
    for job in job_list:
        # CORRECTED LINE: This now handles cases where 'skills' is null
        # or where a skill inside the list is null.
        skills_list = job.get('skills') or []
        if any(skill and skill.get('uuid') is None for skill in skills_list):
            ref_id = job.get('opportunity_ref_id')
            if ref_id:
                problematic_ref_ids.add(ref_id)
    
    logging.info(f"Found {len(problematic_ref_ids)} unique jobs to re-process.")
    if not problematic_ref_ids:
        logging.info("No jobs with null skill UUIDs were found. Exiting.")
        return

    # --- 3. Create the new input file by filtering bert_cleaned.json ---
    logging.info(f"Filtering '{original_input_file.name}' to create the rerun input file...")
    try:
        with open(original_input_file, 'r', encoding='utf-8') as f:
            all_original_jobs = json.load(f)

        jobs_to_rerun = [
            job for job in all_original_jobs 
            if job.get('opportunity_ref_id') in problematic_ref_ids
        ]

        with open(rerun_input_file, 'w', encoding='utf-8') as f:
            json.dump(jobs_to_rerun, f, indent=2, ensure_ascii=False)

        logging.info(f"Successfully created '{rerun_input_file.name}' with {len(jobs_to_rerun)} jobs.")

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Could not load or parse '{original_input_file.name}'. Error: {e}")

if __name__ == "__main__":
    create_rerun_file_from_pilot_db_robust()