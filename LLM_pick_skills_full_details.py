import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types

# TODO: Give option not to choose any required skills if job ad sounds like there aren't any pre-requisites in terms of skills, especially if it a learnership or certificate
# TODO Ask Gemini to help improve my prompt, note that I specifically want to get at "key necessary skills and requirements the employer would be looking for in an applicant"
# TODO Do I currently tell LLM how many skills to return? Do i have it choose how many are most important?

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================

# Configure logging for clear and structured output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_or_initialize_results(output_file: Path) -> Tuple[List[Dict[str, Any]], set]:
    """
    Loads existing results from the output file or returns an empty structure.
    This allows the script to be restarted and skip already processed jobs.
    """
    if output_file.exists():
        logging.info(f"Found existing output file. Loading previous results from {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                # Create a set of reference IDs that have already been processed successfully
                processed_ids = {
                    result['opportunity_ref_id']
                    for result in existing_results
                    if 'error' not in result or result.get('error') != "Skipped"
                }
                logging.info(f"Loaded {len(existing_results)} previous results. Found {len(processed_ids)} already processed jobs.")
                return existing_results, processed_ids
            except json.JSONDecodeError:
                logging.warning(f"Could not decode JSON from {output_file}. Starting fresh.")
                return [], set()
    return [], set()

def save_results_to_file(all_responses: List[Dict[str, Any]], output_file: Path):
    """Saves the entire list of responses to a JSON file, overwriting it."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"FATAL: Could not save results to {output_file}. Error: {e}")

def load_job_data(json_file_path: Path) -> List[Dict[str, Any]]:
    """Load opportunity data from a JSON file."""
    logging.info(f"Loading job data from {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# ============================================================================
# 2. PROMPT CREATION
# ============================================================================

def create_occupation_prompt(full_details: str, potential_occupations: List[str]) -> str:
    """Create the prompt for occupation ranking and selection."""
    occupations_text = "\n".join([f"- {occ}" for occ in potential_occupations])
    return f"""You are an expert job placement specialist.
    
Based on the following opportunity details and list of potential occupations, your task is to identify the most suitable occupation.

**Opportunity Details:**
---
{full_details}
---

**Potential Occupations:**
{occupations_text}

**Instructions:**
1.  Carefully analyze the opportunity details.
2.  Rank all the potential occupations from most to least likely to be the best fit.
3.  From your top three ranked occupations, choose the single one that best represents the core function of the opportunity.
4.  Provide clear, concise reasoning for your final choice, referencing specific parts of the opportunity details.
5.  Output the results in a valid JSON format as specified below. Do not add any text or markdown formatting outside of the JSON block.

**Example JSON Output:**
```json
{{
  "ranked_occupations": [
    {{"occupation": "Software Engineer", "rank": 1}},
    {{"occupation": "Data Scientist", "rank": 2}}
  ],
  "final_choice": {{
    "occupation": "Software Engineer",
    "reasoning": "The description's primary focus on designing, developing, and maintaining software systems aligns directly with the core responsibilities of a Software Engineer role."
  }}
}}
```"""

def create_skills_prompt(full_details: str, potential_skills: List[str], potential_skill_requirements: List[str]) -> str:
    """Create the prompt for skill extraction and ranking, with improved instructions."""
    combined_skills = sorted(list(set(potential_skills + potential_skill_requirements)))
    skills_text = "\n".join([f"- {sk}" for sk in combined_skills])
    
    return f"""You are an expert technical recruiter with deep knowledge of the job market.

Based on the provided opportunity details and a list of potential skills, your task is to identify the key skills required for the role.

**Opportunity Details:**
---
{full_details}
---

**List of Potential Skills to Choose From:**
{skills_text}

**Instructions:**
1.  **Analyze the Text**: Scrutinize the opportunity details to understand the employer's needs.
2.  **Identify Required Skills**: From the provided list, select all skills that are **required** for the job. A "required" skill is a non-negotiable prerequisite, meaning an applicant would likely be disqualified without it.
3.  **Identify Important Skills**: From the provided list, select the top 5 most **important** skills. "Important" skills are those that would make a candidate highly competitive. This list should be ordered by importance, from most to least.
4.  **Handle Entry-Level Roles**: If the opportunity appears to be an entry-level position, learnership, or has no explicit prerequisites, it is acceptable to return an empty list for `required_skills`.
5.  **Strict JSON Output**: Format your response as a single, valid JSON object. Do not include any explanatory text or markdown formatting outside of the JSON block.

**Example JSON Output:**
```json
{{
  "required_skills": [
    "Python",
    "SQL"
  ],
  "top_5_important_skills": [
    "Machine Learning",
    "Data Analysis",
    "Python",
    "Communication",
    "SQL"
  ]
}}
```"""


# ============================================================================
# 3. CORE PROCESSING LOGIC
# ============================================================================

def parse_llm_response(response_text: str) -> Tuple[Dict[str, Any], bool]:
    """Cleans and parses the model's JSON response string."""
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

def call_gemini_with_retry(
    client: genai.GenerativeModel,
    contents: List[types.Content],
    config: types.GenerateContentConfig,
    max_retries: int = 3
) -> str:
    """Calls the Gemini API with a retry mechanism for transient errors."""
    retriable_errors = (
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.InternalServerError,
    )
    for attempt in range(max_retries):
        try:
            # Using streaming for potential future use or large responses
            response_chunks = client.generate_content(contents=contents, generation_config=config, stream=False)
            return response_chunks.text
        except retriable_errors as e:
            logging.warning(f"API call failed with retriable error: {e}. Attempt {attempt + 1} of {max_retries}.")
            if attempt + 1 == max_retries:
                logging.error("Max retries reached. Failing the API call.")
                raise  # Re-raise the exception to be caught by the main loop
            time.sleep(2 ** attempt)  # Exponential backoff
    raise Exception("API call failed after all retries.") # Should not be reached, but for safety


def process_single_job(
    client: genai.GenerativeModel,
    job_data: Dict[str, Any],
    config: types.GenerateContentConfig,
    process_type: str,
) -> Tuple[Dict[str, Any], bool]:
    """
    Processes a single job for either 'skills' or 'occupations'.
    Handles prompt creation, API call, and response parsing.
    """
    opportunity_group_id = job_data['opportunity_group_id'][0]
    opportunity_ref_id = job_data['opportunity_ref_id'][0]
    opportunity_title = job_data['opportunity_title'][0]

    base_response = {
        "opportunity_group_id": opportunity_group_id,
        "opportunity_ref_id": opportunity_ref_id,
        "opportunity_title": opportunity_title,
    }
    
    # --- 1. Create Prompt based on process type ---
    if process_type == 'skills':
        if not job_data.get('potential_skills') and not job_data.get('potential_skill_requirements'):
            logging.warning(f"Skipping SKILLS for {opportunity_ref_id}: No potential skills provided.")
            base_response.update({"error": "Skipped", "error_details": "No potential skills in input."})
            return base_response, False
        prompt_text = create_skills_prompt(
            job_data['full_details'],
            job_data.get('potential_skills', []),
            job_data.get('potential_skill_requirements', [])
        )
    elif process_type == 'occupations':
        if not job_data.get('potential_occupations'):
            logging.warning(f"Skipping OCCUPATION for {opportunity_ref_id}: No potential occupations provided.")
            base_response.update({"error": "Skipped", "error_details": "No potential occupations in input."})
            return base_response, False
        prompt_text = create_occupation_prompt(
            job_data['full_details'],
            job_data['potential_occupations']
        )
    else:
        raise ValueError("Invalid process_type specified.")

    # --- 2. Call LLM with retry logic ---
    contents = [types.Part.from_text(text=prompt_text)]
    try:
        logging.info(f"Generating LLM response for {opportunity_ref_id}...")
        response_text = call_gemini_with_retry(client, contents, config)
    except Exception as e:
        logging.error(f"Generation failed for {opportunity_ref_id} after all retries: {e}")
        base_response.update({"error": "Generation failed", "error_details": str(e)})
        return base_response, False

    # --- 3. Parse Response ---
    parsed_data, is_valid = parse_llm_response(response_text)
    if not is_valid:
        base_response.update(parsed_data) # Add error details from parsing
        return base_response, False

    # --- 4. Success ---
    logging.info(f"Successfully processed {opportunity_ref_id}.")
    base_response.update(parsed_data)
    return base_response, True


# ============================================================================
# 4. MAIN EXECUTION ORCHESTRATOR
# ============================================================================

def process_all_jobs(
    input_file: Path,
    output_file: Path,
    process_type: str, # 'skills' or 'occupations'
    config: Dict[str, Any]
):
    """
    Main orchestration function to process all jobs for a given type.
    Handles loading data, skipping processed jobs, calling the processor,
    and saving results incrementally.
    """
    logging.info(f"\n{'='*60}\n--- STARTING {process_type.upper()} PROCESSING ---\n{'='*60}")
    
    # --- Setup ---
    client = genai.GenerativeModel(config['model_name'])
    system_instruction_text = (
        "You are an expert job recruiter. You are excellent at scanning job descriptions and extracting the most important skills."
        if process_type == 'skills'
        else "You are an expert job placement specialist. Your task is to rank potential occupations and select the best fit for a job opportunity."
    )
    
    generation_config = types.GenerationConfig(
        temperature=config['temperature'],
        top_p=config['top_p'],
        max_output_tokens=config['max_output_tokens'],
    )
    
    job_data_list = load_job_data(input_file)
    all_responses, processed_ids = load_or_initialize_results(output_file)
    
    # Create a map of existing results for easy updating
    response_map = {res['opportunity_ref_id']: res for res in all_responses}

    # --- Main Loop ---
    total_jobs = len(job_data_list)
    for i, job_data in enumerate(job_data_list, 1):
        job_ref_id = job_data['opportunity_ref_id'][0]
        job_title = job_data['opportunity_title'][0]

        logging.info(f"\n--- Processing job {i}/{total_jobs} ({job_ref_id}: {job_title}) ---")

        if job_ref_id in processed_ids:
            logging.info(f"Job {job_ref_id} has already been processed. Skipping.")
            continue

        response, is_valid = process_single_job(client, job_data, generation_config, process_type)
        
        # Add or update the response in our map
        response_map[job_ref_id] = response
        
        # Incrementally save the entire updated list after each API call
        save_results_to_file(list(response_map.values()), output_file)

    # --- Final Summary ---
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
    # Centralized configuration for easy changes
    CONFIG = {
        "base_dir": Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study"),
        #"input_file_name": "bert_cleaned.json", 
        #"input_file_name": "bert_cleaned_subset250.json", # for pipeline testing 
        # "input_file_name": "bert_cleaned_subset6.json", # small trial
        "input_file_name": "bert_cleaned_subset1.json", # tiny trial
        "model_name": "gemini-1.5-pro-latest", # Or "gemini-1.5-flash-latest" for speed/cost savings
        "temperature": 0.5, # Lowered for more deterministic and structured output
        "top_p": 0.95,
        "max_output_tokens": 8192,
    }

    # Initialize the GenAI client once
    genai.configure(project="ihu-access", location="global")

    # --- Execute Occupation Reranking ---
    process_all_jobs(
        input_file=CONFIG['base_dir'] / CONFIG['input_file_name'],
        output_file=CONFIG['base_dir'] / "job_responses_occupations_robust.json",
        process_type='occupations',
        config=CONFIG
    )

    # --- Execute Skills Reranking ---
    process_all_jobs(
        input_file=CONFIG['base_dir'] / CONFIG['input_file_name'],
        output_file=CONFIG['base_dir'] / "job_responses_skills_robust.json",
        process_type='skills',
        config=CONFIG
    )