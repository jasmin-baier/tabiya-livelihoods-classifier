import json
import ijson
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Step 1: Import the correct Vertex AI libraries
import vertexai
from google.api_core import exceptions as google_exceptions
from vertexai.generative_models import GenerationConfig, GenerativeModel

# TODO before final full job runs, handle duplicates better in beginning of pipeline

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_or_initialize_results(output_file: Path) -> Tuple[List[Dict[str, Any]], set]:
    if output_file.exists():
        logging.info(f"Found existing output file. Loading previous results from {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                processed_ids = {
                    result['opportunity_ref_id']
                    for result in existing_results
                    if 'error' not in result
                }
                logging.info(f"Loaded {len(existing_results)} previous results. Found {len(processed_ids)} already processed jobs.")
                return existing_results, processed_ids
            except json.JSONDecodeError:
                logging.warning(f"Could not decode JSON from {output_file}. Starting fresh.")
                return [], set()
    return [], set()

def save_results_to_file(all_responses: List[Dict[str, Any]], output_file: Path):
    """
    Safely writes data by using a temporary file and an atomic replace.
    Includes retry logic to handle file locks from sync services like OneDrive.
    """
    temp_file = output_file.with_suffix(output_file.suffix + '.tmp')
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(all_responses, f, indent=2, ensure_ascii=False)
            
            # The atomic operation: this will overwrite the destination file safely.
            temp_file.replace(output_file)
            
            # If we get here, the save was successful, so we can exit the loop.
            return # Exit the function successfully
            
        except Exception as e:
            # Check if it's the specific error we're looking for, or any permission error
            if attempt < max_retries - 1:
                logging.warning(f"Save failed on attempt {attempt + 1}/{max_retries}. Retrying in 1 second. Error: {e}")
                time.sleep(1) # Wait a moment for the other process (OneDrive) to release the lock
            else:
                logging.error(f"FATAL: Could not save results to {output_file} after {max_retries} attempts. Error: {e}")
                # Clean up the temporary file on final failure
                if temp_file.exists():
                    temp_file.unlink()

# ============================================================================
# 2. PROMPT CREATION (No changes needed here)
# ============================================================================
def create_occupation_prompt(full_details: str, potential_occupations: List[Dict[str, str]]) -> str:
    """
    Creates a prompt for the LLM, including both occupation labels and their descriptions.
    """
    # Format each occupation with its description for clarity in the prompt
    occupations_text = "\n".join(
        [f"- **{occ['label']}**: {occ['description']}" for occ in potential_occupations if occ.get('label')]
    )
    
    return f"""You are an expert job placement specialist.
    Based on the following opportunity details and a list of potential occupations with their official descriptions, your task is to identify the most suitable occupation.

    **Opportunity Details:**
    ---
    {full_details}
    ---

    **Potential Occupations with Descriptions:**
    {occupations_text}

    **Instructions:**
    1.  Carefully analyze the opportunity details against the provided occupation descriptions.
    2.  Rank all the potential occupations from most to least likely to be the best fit.
    3.  From your top-ranked occupations, choose the single one that best represents the core function of the opportunity.
    4.  Provide clear, concise reasoning for your final choice, referencing specific parts of the opportunity details and how they align with the official occupation description.
    5.  Output the results in a valid JSON format as specified below. Do not add any text or markdown formatting outside of the JSON block.
    
    Do not make up any occupations, unless there are none provided. If there is a list provided, only use those.

    **Example JSON Output:**
    ```json
    {{
      "ranked_occupations": [
        {{"occupation": "Software Engineer", "rank": 1}},
        {{"occupation": "Data Scientist", "rank": 2}}
      ],
      "final_choice": {{
        "occupation": "Software Engineer",
        "reasoning": "The description's primary focus on 'designing, developing, and maintaining software systems' aligns directly with the core responsibilities of a Software Engineer, as opposed to the statistical analysis focus of a Data Scientist."
      }}
    }}
    ```"""

def create_skills_prompt(full_details: str, potential_skills: List[str], potential_skill_requirements: List[str]) -> str:
    combined_skills = sorted(list(set(potential_skills + potential_skill_requirements)))
    skills_text = "\n".join([f"- {sk}" for sk in combined_skills])
    
    return f"""You are an expert technical recruiter with deep knowledge of the job market. Your task is to analyze a job opportunity and categorize skills based on the employer's needs at the time of application.

**Opportunity Details:**
---
{full_details}
---

**List of Potential Skills to Choose From:**
{skills_text}

**Instructions:**
1.  **Analyze the Role**: Carefully read the opportunity details to understand what the employer is looking for in an applicant *at the time of application*.

2.  **Identify "Required Skills"**: From the provided list, select only the skills that an employer **needs to see in a candidate to even consider hiring them**. These are the non-negotiable, "must-have" prerequisites.

3.  **Identify "Important Skills"**: From the provided list, select the skills that **would make an applicant stronger** or more competitive. These are desirable, "nice-to-have" skills. Please return the a good number of important skills, ordered from most to least important. Use your judgement to decide which and how many are relevant.

4.  **Exclude "On-the-Job" Skills**: Crucially, **do not include skills that the job ad states or implies will be taught or developed after hiring**. Your focus is strictly on the requirements at the point of application.

5.  **Handle Entry-Level Roles**: If there are no clear prerequisites (e.g., for a learnership), it is correct to return an empty list for `required_skills`.

6.  **Use Provided List Only**: Do not make up any skills. You must only choose from the list of potential skills provided.

7.  **Strict JSON Output**: Format your response as a single, valid JSON object. Do not include any explanatory text or markdown formatting outside of the JSON block.

8.  **Verbatim Matching**: The skill strings you return in the JSON output **must be an exact, verbatim copy** of a skill from the provided list. Do not alter the skill in any way (e.g., do not change 'Project Management' to 'project_management'). **Skills with underscores are invalid and must be ignored.**

**Example JSON Output:**
```json
{{
  "required_skills": [
    "Commercial Driver's License (CDL)",
    "Clean Driving Record"
  ],
  "top_important_skills": [
    "Customer Service",
    "Time Management",
    "Communication",
    "Problem-Solving",
    "Defensive Driving"
  ]
}}
```"""

# ============================================================================
# 3. CORE PROCESSING LOGIC
# ============================================================================
def parse_llm_response(response_text: str) -> Tuple[Dict[str, Any], bool]:
    clean_text = response_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    try:
        parsed_response = json.loads(clean_text.strip())
        return parsed_response, True
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse JSON. Error: {e}")
        error_details = {"error": "Invalid JSON response", "error_details": str(e), "raw_response": response_text}
        return error_details, False

def call_vertexai_with_retry(
    client: GenerativeModel,
    contents: str,
    config: GenerationConfig,
    max_retries: int = 3
) -> str:
    retriable_errors = (
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.InternalServerError,
    )
    for attempt in range(max_retries):
        try:
            # Step 2: Use the Vertex AI client's method
            response = client.generate_content(contents=contents, generation_config=config)
            return response.text
        except retriable_errors as e:
            logging.warning(f"API call failed with retriable error: {e}. Attempt {attempt + 1} of {max_retries}.")
            if attempt + 1 == max_retries:
                logging.error("Max retries reached. Failing the API call.")
                raise
            time.sleep(2 ** attempt)
    raise Exception("API call failed after all retries.")

def process_single_job(
    client: GenerativeModel,
    job_data: Dict[str, Any],
    config: GenerationConfig,
    process_type: str,
) -> Tuple[Dict[str, Any], bool]:
    opportunity_group_id = job_data['opportunity_group_id']
    opportunity_ref_id = job_data['opportunity_ref_id']
    opportunity_title = job_data['opportunity_title']
    base_response = {
        "opportunity_group_id": opportunity_group_id, "opportunity_ref_id": opportunity_ref_id, "opportunity_title": opportunity_title
    }
    
    if process_type == 'skills':
        if not job_data.get('potential_skills') and not job_data.get('potential_skill_requirements'):
            logging.warning(f"Skipping SKILLS for {opportunity_ref_id}: No potential skills provided.")
            base_response.update({"error": "Skipped", "error_details": "No potential skills in input."})
            return base_response, False
        prompt_text = create_skills_prompt(
            job_data['full_details'], job_data.get('potential_skills', []), job_data.get('potential_skill_requirements', [])
        )
    elif process_type == 'occupations':
        # Combine labels and descriptions into a list of dictionaries
        occ_labels = job_data.get('potential_occupations', [])
        occ_descriptions = job_data.get('potential_occupations_descriptions', [])
        
        if not occ_labels:
            logging.warning(f"Skipping OCCUPATION for {opportunity_ref_id}: No potential occupations provided.")
            base_response.update({"error": "Skipped", "error_details": "No potential occupations in input."})
            return base_response, False
            
        # Ensure descriptions list is the same length as labels, padding with empty strings if necessary
        occ_descriptions.extend([''] * (len(occ_labels) - len(occ_descriptions)))
        
        combined_occupations = [
            {'label': label, 'description': desc}
            for label, desc in zip(occ_labels, occ_descriptions)
        ]
        
        prompt_text = create_occupation_prompt(
            job_data['full_details'], combined_occupations
        )
    else:
        raise ValueError("Invalid process_type specified.")
        
    try:
        logging.info(f"Generating LLM response for {opportunity_ref_id}...")
        response_text = call_vertexai_with_retry(client, prompt_text, config)
    except Exception as e:
        logging.error(f"Generation failed for {opportunity_ref_id} after all retries: {e}")
        base_response.update({"error": "Generation failed", "error_details": str(e)})
        return base_response, False
        
    parsed_data, is_valid = parse_llm_response(response_text)
    if not is_valid:
        base_response.update(parsed_data)
        return base_response, False
        
    logging.info(f"Successfully processed {opportunity_ref_id}.")
    base_response.update(parsed_data)
    return base_response, True

# ============================================================================
# 4. MAIN EXECUTION ORCHESTRATOR
# ============================================================================
# This is the new, memory-efficient version of the function
def process_all_jobs(
    input_file: Path,
    output_file: Path,
    process_type: str,
    config: Dict[str, Any]
):
    """
    Main orchestration function. Processes jobs one-by-one from the input
    file to conserve memory.
    """
    logging.info(f"\n{'='*60}\n--- STARTING {process_type.upper()} PROCESSING ---\n{'='*60}")
    
    # Initialize Vertex AI and the model
    vertexai.init(project=config['project'], location=config['location'])
    client = GenerativeModel(config['model_name'])
    
    generation_config = GenerationConfig(
        temperature=config['temperature'],
        top_p=config['top_p'],
        max_output_tokens=config['max_output_tokens'],
    )
    
    all_responses, processed_ids = load_or_initialize_results(output_file)
    response_map = {res['opportunity_ref_id']: res for res in all_responses}
    
    # --- Main Loop: Reads one job at a time from the file ---
    processed_count = 0
    with open(input_file, 'rb') as f: # Must open in binary mode for ijson
        # Create an iterator that yields one job object at a time
        jobs_iterator = ijson.items(f, 'item')
        
        for i, job_data in enumerate(jobs_iterator, 1):
            processed_count = i
            job_ref_id = job_data['opportunity_ref_id']
            job_title = job_data['opportunity_title']

            logging.info(f"\n--- Processing job {i} ({job_ref_id}: {job_title}) ---")

            if job_ref_id in processed_ids:
                logging.info(f"Job {job_ref_id} has already been processed. Skipping.")
                continue

            response, is_valid = process_single_job(client, job_data, generation_config, process_type)
            
            # Add or update the response in our map and save incrementally
            response_map[job_ref_id] = response
            save_results_to_file(list(response_map.values()), output_file)

    # --- Final Summary ---
    final_results = list(response_map.values())
    successful_jobs = [r for r in final_results if 'error' not in r]
    failed_jobs = [r for r in final_results if 'error' in r]
    
    logging.info(f"\n\n{'='*60}\n{process_type.upper()} PROCESSING SUMMARY\n{'='*60}")
    logging.info(f"Total jobs processed from file: {processed_count}")
    logging.info(f"Total results in output file: {len(final_results)}")
    logging.info(f"Successful: {len(successful_jobs)}")
    logging.info(f"Failed/Skipped: {len(failed_jobs)}")
    if failed_jobs:
        logging.warning("\n--- Failed/Skipped Jobs ---")
        for job in failed_jobs:
            error_details = job.get('error_details', 'N/A')
            logging.warning(f"  - {job['opportunity_ref_id']}: {job['opportunity_title']} ({job['error']}: {str(error_details)[:200]})") # Truncate long errors
    logging.info(f"\nFinal results are saved in: {output_file}\n{'='*60}")

# ============================================================================
# EXECUTION SCRIPT
# ============================================================================

if __name__ == "__main__":
    CONFIG = {
        "base_dir": Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study"),
        "input_file_name": "bert_cleaned.json",
        #"input_file_name": "bert_cleaned_subset250.json",
        #"input_file_name": "bert_cleaned_subset6.json",
        #"input_file_name": "bert_cleaned_subset1.json",
        "model_name": "gemini-2.5-pro", # Use a valid Vertex AI model name
        "project": "ihu-access",
        "location": "global", # Or a specific region like "us-central1"
        "temperature": 0.3,
        "top_p": 0.95,
        "max_output_tokens": 8192,
        "max_reruns": 5,
    }

    def run_with_retry(process_type, input_file, output_file, config):
        # First, get the total number of jobs to process
        total_jobs = 0
        with open(input_file, 'rb') as f:
            for _ in ijson.items(f, 'item'):
                total_jobs += 1
        logging.info(f"Found {total_jobs} total jobs in {input_file.name} for {process_type} processing.")

        rerun_count = 0
        while rerun_count < config["max_reruns"]:
            process_all_jobs(
                input_file=input_file,
                output_file=output_file,
                process_type=process_type,
                config=config
            )

            # Check if all jobs were processed successfully
            final_results, processed_ids = load_or_initialize_results(output_file)
            
            if len(processed_ids) >= total_jobs:
                logging.info(f"All {total_jobs} jobs successfully processed for {process_type}. Run complete.")
                break
            else:
                rerun_count += 1
                logging.warning(f"Run {rerun_count}/{config['max_reruns']} incomplete. {len(processed_ids)}/{total_jobs} jobs processed.")
                logging.warning("There are still unprocessed jobs. Retrying in 10 seconds...")
                time.sleep(10)
        else:
             logging.error(f"Max reruns reached for {process_type}, but not all jobs were processed. Please check for persistent errors.")


    # --- Execute Occupation Reranking with Retry Logic ---
    #run_with_retry(
    #    process_type='occupations',
    #    input_file=CONFIG['base_dir'] / CONFIG['input_file_name'],
    #    output_file=CONFIG['base_dir'] / "job_responses_occupations_version-oppdescskillsno.json",
    #    config=CONFIG
    #)

    # --- Execute Skills Reranking with Retry Logic ---
    run_with_retry(
        process_type='skills',
        input_file=CONFIG['base_dir'] / CONFIG['input_file_name'],
        output_file=CONFIG['base_dir'] / "job_responses_skills_version-oppdescskillsno.json",
        config=CONFIG
    )