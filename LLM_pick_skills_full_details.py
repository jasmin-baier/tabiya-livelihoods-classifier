from google import genai
from google.genai import types
import json
from pathlib import Path

# TODO: It currently comes up with skills and occupations IF there is no input.

# ============================================================================
# SHARED HELPER FUNCTIONS
# ============================================================================

def load_job_data(json_file_path):
    """Load opportunity data from a JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def parse_response(opportunity_group_id, opportunity_ref_id, opportunity_title, response_text):
    """Parse and validate the model's JSON response."""
    
    # Clean the response string to remove Markdown formatting
    clean_text = response_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:] # Remove ```json\n
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3] # Remove ```
    
    try:
        # Now, parse the cleaned string
        parsed_response = json.loads(clean_text.strip())
        print(f"✓ Successfully parsed JSON for opportunity {opportunity_group_id} - {opportunity_ref_id}")
        return parsed_response, True
    except json.JSONDecodeError as e:
        error_response = {
            "opportunity_group_id": opportunity_group_id,
            "opportunity_ref_id": opportunity_ref_id,
            "opportunity_title": opportunity_title,
            "error": "Invalid JSON response",
            "error_details": str(e),
            "raw_response": response_text.strip() # Log the original raw response for debugging
        }
        print(f"⚠ Failed to parse JSON for opportunity {opportunity_group_id} - {opportunity_ref_id}: {e}")
        return error_response, False


def save_all_responses_to_file(all_responses, output_file):
    """Save all responses to a single JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved all responses to: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to save the combined file: {e}")
        return False

# ============================================================================
# --- Ask LLM to pick OCCUPATION ---
# ============================================================================

def create_occupation_prompt(full_details, potential_occupations):
    """Create the prompt for occupation ranking and selection."""
    occupations_text = "\n".join([f"- {occ}" for occ in potential_occupations])
    
    prompt_text = f"""You are be provided with the following information:
Opportunity Details: {full_details}
Potential Occupations: {occupations_text}

Instructions:
1. Rank the potential occupations from most to least likely to be the best fit.
2. From the top three, choose the one that best fits the opportunity.
3. Provide detailed reasoning for your choice.
4. Output the results in JSON format, ranked list, final choice, and reasoning.

Do not make up any occupations. Only use the provided list.

Example JSON Output:
{{
  "ranked_occupations": [
    {{"occupation": "Software Engineer", "rank": 1}},
    {{"occupation": "Data Scientist", "rank": 2}}
  ],
  "final_choice": {{
    "occupation": "Software Engineer",
    "reasoning": "The description's focus on software development aligns with a Software Engineer role."
  }}
}}"""
    return prompt_text

def process_single_job_for_occupation(client, model, job_data, generate_content_config):
    """Process a single job to determine its occupation."""
    opportunity_group_id = job_data['opportunity_group_id'][0]
    opportunity_ref_id = job_data['opportunity_ref_id'][0]
    opportunity_title = job_data['opportunity_title'][0]
    
    # If the list of potential occupations is empty, skip this job.
    if not job_data.get('potential_occupations'):
        print(f"⚪ Skipping Job ID {opportunity_group_id} - {opportunity_ref_id}: No potential occupations provided.")
        skip_response = {
            "opportunity_group_id": opportunity_group_id,
            "opportunity_ref_id": opportunity_ref_id,
            "opportunity_title": opportunity_title,
            "error": "Skipped",
            "error_details": "No potential occupations were provided in the input data."
        }
        return skip_response, False
    
    print(f"\n{'='*50}\nProcessing Occupation for Job ID: {opportunity_group_id} - {opportunity_ref_id}\nJob Title: {job_data['opportunity_title'][0]}\n{'='*50}")

    prompt_text = create_occupation_prompt(
        job_data['full_details'],
        job_data['potential_occupations']
    )
    
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])]
    
    print(f"\nResponse for Job ID {opportunity_group_id} - {opportunity_ref_id}:\n" + "-"*30)
    response_text = ""
    try:
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
            print(chunk.text, end="")
            response_text += chunk.text
        
        print("\n" + "-" * 30)
        return parse_response(opportunity_group_id, opportunity_ref_id, opportunity_title, response_text)
        
    except Exception as e:
        print(f"\n✗ Error generating response for job {opportunity_group_id} - {opportunity_ref_id}: {e}")
        return {"opportunity_group_id": opportunity_group_id, 
                "opportunity_ref_id": opportunity_ref_id, 
                "opportunity_title": opportunity_title,
                "error": "Generation failed", 
                "error_details": str(e)}, False

def generate_for_all_jobs_occupations(json_file_path, output_file):
    """Main function to process occupations for all jobs."""
    client = genai.Client(vertexai=True, project="ihu-access", location="global")
    si_text = """You are an expert job placement specialist. You will receive opportunity details and a list of potential occupations. Your task is to rank the potential occupations from most to least fitting based on the opportunity details. Then, from the top three occupations, select the one that best fits the opportunity details. The output should be a JSON file containing the ranked list of occupations, the final choice, and the reasoning behind the choice."""
    model = "gemini-2.5-pro"
    generate_content_config = types.GenerateContentConfig(temperature=1, top_p=0.95, max_output_tokens=8192, system_instruction=[types.Part.from_text(text=si_text)])
    
    job_data_list = load_job_data(json_file_path)
    print(f"Loaded {len(job_data_list)} jobs. Output will be saved to: {output_file}")
    
    all_responses, successful_jobs, failed_jobs = [], [], []
    
    for i, job_data in enumerate(job_data_list, 1):
        print(f"\n\nProcessing job {i}/{len(job_data_list)} for occupation.")
        response, is_valid = process_single_job_for_occupation(client, model, job_data, generate_content_config)
        
        if is_valid:
            # Add the opportunity title and ids to the successful response
            response['opportunity_title'] = job_data['opportunity_title'][0] 
            response['opportunity_group_id'] = job_data['opportunity_group_id'][0] 
            response['opportunity_ref_id'] = job_data['opportunity_ref_id'][0] 

        all_responses.append(response)
        
        (successful_jobs if is_valid else failed_jobs).append({
            'opportunity_group_id': job_data.get('opportunity_group_id', 'unknown'),
            'opportunity_ref_id': job_data.get('opportunity_ref_id', 'unknown'),
            'opportunity_title': job_data.get('opportunity_title', ['unknown'])[0], 
            'error': response.get('error', 'Unknown error') if not is_valid else None
        })
    
    save_all_responses_to_file(all_responses, output_file)
    print_summary(len(job_data_list), successful_jobs, failed_jobs, output_file)

# ============================================================================
# --- Ask LLM to pick SKILLS ---
# ============================================================================

def create_skills_prompt(full_details, potential_skills, potential_skill_requirements):
    """Create the prompt for skill extraction and ranking."""
    combined_skills = potential_skills + potential_skill_requirements
    skills_text = "\n".join([f"- {sk}" for sk in combined_skills])
    
    prompt_text = f"""Here are the opportunity details:
{full_details}

Here is a list of potential skills:
{skills_text}

Follow these rules:
* Scan the opportunity details.
* Rank the skills by importance.
* Choose all skills that seem required for hire.
* Choose the top 5 most important skills, ordered by importance.
* Output the results in a structured JSON format.

Do not make up any skills. Only use the provided list.

Here is an example of the desired output format:
{{
  "required_skills": ["skill1", "skill2"],
  "top_5_important_skills": ["skill3", "skill4", "skill5", "skill6", "skill7"]
}}"""
    return prompt_text

def process_single_job_for_skills(client, model, job_data, generate_content_config):
    """Process a single job to extract its key skills."""
    opportunity_group_id = job_data['opportunity_group_id'][0]
    opportunity_ref_id = job_data['opportunity_ref_id'][0]
    opportunity_title = job_data['opportunity_title'][0]

    # If BOTH skills lists are empty, skip this job.
    # We use .get() as a safe way to access keys that might not exist.
    if not job_data.get('potential_skills') and not job_data.get('potential_skill_requirements'):
        print(f"⚪ Skipping Job ID {opportunity_group_id} - {opportunity_ref_id}: No potential skills provided.")
        skip_response = {
            "opportunity_group_id": opportunity_group_id,
            "opportunity_ref_id": opportunity_ref_id,
            "opportunity_title": opportunity_title,
            "error": "Skipped",
            "error_details": "No potential skills or skill requirements were provided in the input data."
        }
        return skip_response, False


    print(f"\n{'='*50}\nProcessing Skills for Job ID: {opportunity_group_id} - {opportunity_ref_id}\nJob Title: {job_data['opportunity_title'][0]}\n{'='*50}")

    prompt_text = create_skills_prompt(
        job_data['full_details'],
        job_data.get('potential_skills', []),
        job_data.get('potential_skill_requirements', []) 
    )
    
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])]
    
    print(f"\nResponse for Opportunity ID {opportunity_group_id} - {opportunity_ref_id}:\n" + "-" * 30)
    response_text = ""
    try:
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
            print(chunk.text, end="")
            response_text += chunk.text
        
        print("\n" + "-" * 30)
        return parse_response(opportunity_group_id, opportunity_ref_id, opportunity_title, response_text)
        
    except Exception as e:
        print(f"\n✗ Error generating response for opportunity {opportunity_group_id} - {opportunity_ref_id}: {e}")
        return {"opportunity_group_id": opportunity_group_id, 
                "opportunity_ref_id": opportunity_ref_id, 
                "ooportunity_title": opportunity_title, 
                "error": "Generation failed", 
                "error_details": str(e)}, False

def generate_for_all_jobs_skills(json_file_path, output_file):
    """Main function to process skills for all jobs."""
    client = genai.Client(vertexai=True, project="ihu-access", location="global")
    si_text = """You are an expert job recruiter. You are excellent at scanning job descriptions (called opportunity details) and extracting the most important skills for the job opportunity."""
    model = "gemini-2.5-pro"
    generate_content_config = types.GenerateContentConfig(temperature=1, top_p=0.95, max_output_tokens=8192, system_instruction=[types.Part.from_text(text=si_text)])
    
    job_data_list = load_job_data(json_file_path)
    print(f"Loaded {len(job_data_list)} jobs. Output will be saved to: {output_file}")
    
    all_responses, successful_jobs, failed_jobs = [], [], []

    for i, job_data in enumerate(job_data_list, 1):
        print(f"\n\nProcessing job {i}/{len(job_data_list)} for skills.")
        response, is_valid = process_single_job_for_skills(client, model, job_data, generate_content_config)
        
        if is_valid:
            # Add the opportunity title and ids to the successful response
            response['opportunity_title'] = job_data['opportunity_title'][0] 
            response['opportunity_group_id'] = job_data['opportunity_group_id'][0] 
            response['opportunity_ref_id'] = job_data['opportunity_ref_id'][0] 

        all_responses.append(response)

        (successful_jobs if is_valid else failed_jobs).append({
            'opportunity_group_id': job_data.get('opportunity_group_id', 'unknown'),
            'opportunity_ref_id': job_data.get('opportunity_ref_id', 'unknown'),
            'opportunity_title': job_data.get('opportunity_title', ['unknown'])[0],
            'error': response.get('error', 'Unknown error') if not is_valid else None
        })

    save_all_responses_to_file(all_responses, output_file)
    print_summary(len(job_data_list), successful_jobs, failed_jobs, output_file)

# ============================================================================
# UTILITY AND EXECUTION
# ============================================================================

def print_summary(total_jobs, successful_jobs, failed_jobs, output_file):
    """Prints a summary of the processing results."""
    print(f"\n\n{'='*60}\nPROCESSING SUMMARY\n{'='*60}")
    print(f"Total jobs processed: {total_jobs}")
    print(f"Successful: {len(successful_jobs)}")
    print(f"Failed: {len(failed_jobs)}")
    
    if successful_jobs:
        print("\n✓ Successfully processed jobs:")
        for job in successful_jobs:
            print(f"  - {job['opportunity_ref_id']}: {job['opportunity_title']}")
    
    if failed_jobs:
        print("\n✗ Failed jobs:")
        for job in failed_jobs:
            print(f"  - {job['opportunity_ref_id']}: {job['opportunity_title']} ({job['error']})")
    
    print(f"\nAll responses saved in: {output_file}\n{'='*60}")

if __name__ == "__main__":
    # Set the base directory
    base_dir = Path("C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study")

    # --- Execute Occupation Reranking ---
    print("\n--- STARTING OCCUPATION RERANKING ---")
    occupation_input_file = base_dir / "opportunities_llm_test.json"
    occupation_output_file = base_dir / "job_responses_occupations_combined.json"
    generate_for_all_jobs_occupations(occupation_input_file, occupation_output_file)

    # --- Execute Skills Reranking ---
    print("\n--- STARTING SKILLS RERANKING ---")
    skills_input_file = base_dir / "opportunities_llm_test.json"
    skills_output_file = base_dir / "job_responses_skills_combined.json"
    generate_for_all_jobs_skills(skills_input_file, skills_output_file)