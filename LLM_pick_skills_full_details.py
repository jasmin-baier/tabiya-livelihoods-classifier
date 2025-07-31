import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Step 1: Import the correct Vertex AI libraries
import vertexai
from google.api_core import exceptions as google_exceptions
from vertexai.generative_models import GenerationConfig, GenerativeModel

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
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"FATAL: Could not save results to {output_file}. Error: {e}")

def load_job_data(json_file_path: Path) -> List[Dict[str, Any]]:
    logging.info(f"Loading job data from {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# ============================================================================
# 2. PROMPT CREATION (No changes needed here)
# ============================================================================
def create_occupation_prompt(full_details: str, potential_occupations: List[str]) -> str:
    occupations_text = "\n".join([f"- {occ}" for occ in potential_occupations])
    return f"""You are an expert job placement specialist.
    Based on the following opportunity details and list of potential occupations, your task is to identify the most suitable occupation.
    **Opportunity Details:**\n---\n{full_details}\n---\n**Potential Occupations:**\n{occupations_text}
    **Instructions:**\n1.  Carefully analyze the opportunity details.\n2.  Rank all the potential occupations from most to least likely to be the best fit.\n3.  From your top three ranked occupations, choose the single one that best represents the core function of the opportunity.\n4.  Provide clear, concise reasoning for your final choice, referencing specific parts of the opportunity details.\n5.  Output the results in a valid JSON format as specified below. Do not add any text or markdown formatting outside of the JSON block.
    Do not make up any skills, unless there are none provided. If there is a list provided, only use those.
    **Example JSON Output:**\n```json\n{{\n  "ranked_occupations": [\n    {{"occupation": "Software Engineer", "rank": 1}},\n    {{"occupation": "Data Scientist", "rank": 2}}\n  ],\n  "final_choice": {{\n    "occupation": "Software Engineer",\n    "reasoning": "The description's primary focus on designing, developing, and maintaining software systems aligns directly with the core responsibilities of a Software Engineer role."\n  }}\n}}\n```"""

def create_skills_prompt(full_details: str, potential_skills: List[str], potential_skill_requirements: List[str]) -> str:
    combined_skills = sorted(list(set(potential_skills + potential_skill_requirements)))
    skills_text = "\n".join([f"- {sk}" for sk in combined_skills])
    return f"""You are an expert technical recruiter with deep knowledge of the job market.
    Based on the provided opportunity details and a list of potential skills, your task is to identify the key skills required for the role.
    **Opportunity Details:**\n---\n{full_details}\n---\n**List of Potential Skills to Choose From:**\n{skills_text}
    **Instructions:**\n1.  **Analyze the Text**: Scrutinize the opportunity details to understand the employer's needs.\n2.  **Identify Required Skills**: From the provided list, select all skills that are **required** for the job. A "required" skill is a non-negotiable prerequisite, meaning an applicant would likely be disqualified without it.\n3.  **Identify Important Skills**: From the provided list, select the most **important** skills. "Important" skills are those that would make a candidate highly competitive. This list should be ordered by importance, from most to least.\n4.  **Handle Entry-Level Roles**: If the opportunity appears to be an entry-level position, learnership, or has no explicit prerequisites, it is acceptable to return an empty list for `required_skills`.\n5.  **Strict JSON Output**: Format your response as a single, valid JSON object. Do not include any explanatory text or markdown formatting outside of the JSON block.
    Do not make up any skills. Only use those provided.
    **Example JSON Output:**\n```json\n{{\n  "required_skills": [\n    "Python",\n    "SQL"\n  ],\n  "top_5_important_skills": [\n    "Machine Learning",\n    "Data Analysis",\n    "Python",\n    "Communication",\n    "SQL"\n  ]\n}}\n```"""

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
        if not job_data.get('potential_occupations'):
            logging.warning(f"Skipping OCCUPATION for {opportunity_ref_id}: No potential occupations provided.")
            base_response.update({"error": "Skipped", "error_details": "No potential occupations in input."})
            return base_response, False
        prompt_text = create_occupation_prompt(
            job_data['full_details'], job_data['potential_occupations']
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
def process_all_jobs(
    input_file: Path,
    output_file: Path,
    process_type: str,
    config: Dict[str, Any]
):
    logging.info(f"\n{'='*60}\n--- STARTING {process_type.upper()} PROCESSING ---\n{'='*60}")
    
    # Step 3: Initialize Vertex AI and the model
    vertexai.init(project=config['project'], location=config['location'])
    client = GenerativeModel(config['model_name'])
    
    generation_config = GenerationConfig(
        temperature=config['temperature'],
        top_p=config['top_p'],
        max_output_tokens=config['max_output_tokens'],
    )
    job_data_list = load_job_data(input_file)
    all_responses, processed_ids = load_or_initialize_results(output_file)
    response_map = {res['opportunity_ref_id']: res for res in all_responses}
    total_jobs = len(job_data_list)

    for i, job_data in enumerate(job_data_list, 1):
        job_ref_id = job_data['opportunity_ref_id'][0]
        job_title = job_data['opportunity_title'][0]
        logging.info(f"\n--- Processing job {i}/{total_jobs} ({job_ref_id}: {job_title}) ---")
        if job_ref_id in processed_ids:
            logging.info(f"Job {job_ref_id} has already been processed. Skipping.")
            continue
        response, is_valid = process_single_job(client, job_data, generation_config, process_type)
        response_map[job_ref_id] = response
        save_results_to_file(list(response_map.values()), output_file)

    final_results = list(response_map.values())
    successful_jobs = [r for r in final_results if 'error' not in r]
    failed_jobs = [r for r in final_results if 'error' in r]
    logging.info(f"\n\n{'='*60}\n{process_type.upper()} PROCESSING SUMMARY\n{'='*60}")
    logging.info(f"Total jobs in input: {total_jobs}")
    logging.info(f"Total results in output file: {len(final_results)}")
    logging.info(f"Successful: {len(successful_jobs)}")
    logging.info(f"Failed/Skipped: {len(failed_jobs)}")
    if failed_jobs:
        logging.warning("\n--- Failed/Skipped Jobs ---")
        for job in failed_jobs:
            logging.warning(f"  - {job['opportunity_ref_id']}: {job['opportunity_title']} ({job['error']}: {job.get('error_details', 'N/A')})")
    logging.info(f"\nFinal results are saved in: {output_file}\n{'='*60}")

if __name__ == "__main__":
    CONFIG = {
        "base_dir": Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study"),
        #"input_file_name": "bert_cleaned.json",
        "input_file_name": "bert_cleaned_subset250.json",
        #"input_file_name": "bert_cleaned_subset1.json",
        "model_name": "gemini-2.5-pro", # Use a valid Vertex AI model name
        "project": "ihu-access",
        "location": "global", # Or a specific region like "us-central1"
        "temperature": 0.5,
        "top_p": 0.95,
        "max_output_tokens": 8192,
    }

    # Execute Occupation Reranking
    process_all_jobs(
        input_file=CONFIG['base_dir'] / CONFIG['input_file_name'],
        output_file=CONFIG['base_dir'] / "job_responses_occupations_robust.json",
        process_type='occupations',
        config=CONFIG
    )

    # Execute Skills Reranking
    process_all_jobs(
        input_file=CONFIG['base_dir'] / CONFIG['input_file_name'],
        output_file=CONFIG['base_dir'] / "job_responses_skills_robust.json",
        process_type='skills',
        config=CONFIG
    )