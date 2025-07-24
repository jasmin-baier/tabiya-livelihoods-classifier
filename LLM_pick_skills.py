from google import genai
from google.genai import types
import json
import base64
import os
from datetime import datetime

# TODO: Think about whether there is an advantage in giving LLM full details instead -- has company name and salary etc but then harder to save together with description in output
# TODO: There are quite a few cases where Bert didn't extract any occupations. Probably this is correct (they seem to be mostly trainings) but double check and think about how to handle such cases
# TODO when writing into Opportunity Database, add date it was created and date it was updated (e.g. relevancy needs to be updated daily)
# TODO: probably here or at next stage want to separate vacancy opportunities from learning opportunities

# Take jobs with list of skills from cleaned Bert

# Get labels from taxonomy

####### -------------------------------------------- #######
####### -------  Ask LLM to pick occupation  ------- #######
####### -------------------------------------------- #######


def load_job_data(json_file_path):
    """Load job data from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def parse_response(ReferenceNumber, response_text):
    """Parse and validate the model response"""
    try:
        # Try to parse the response as JSON
        parsed_response = json.loads(response_text.strip())
        print(f"✓ Successfully parsed JSON response for job {ReferenceNumber}")
        return parsed_response, True
    except json.JSONDecodeError as e:
        # If it's not valid JSON, create an error entry
        error_response = {
            "ReferenceNumber": ReferenceNumber,
            "error": "Invalid JSON response",
            "error_details": str(e),
            "raw_response": response_text.strip()
        }
        print(f"⚠ Failed to parse JSON for job {ReferenceNumber}: {str(e)}")
        return error_response, False

def save_all_responses_to_file(all_responses, output_file="job_occupations_combined.json"):
    """Save all responses to a single JSON array file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved all responses to: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to save combined file: {str(e)}")
        return False
    """Load job data from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_job_prompt(ReferenceNumber, job_title, job_description, job_requirements, potential_occupations):
    """Create the prompt text for a specific job"""
    occupations_text = "\n".join([f"- {occ}" for occ in potential_occupations])
    
    prompt_text = f"""You will be provided with the following information:
Job ID: {ReferenceNumber}
Job Title: {job_title}
Job Description: {job_description}
Job Requirements: {job_requirements}
Potential Occupations:
{occupations_text}

Instructions:
1. Rank the potential occupations from most to least likely to be the best fit for the job description and requirements.
2. From the top three occupations, choose the one that best fits the job title, job description, and job requirements.
3. Provide a detailed reasoning for your final choice.
4. Output the results in JSON format, including the job ID, ranked list of occupations, the final choice, and the reasoning behind the choice.

Do not make up any occupations. Only use the provided potential occupations.

Example JSON Output:
{{
  "ReferenceNumber": "12345",
  "ranked_occupations": [
    {{
      "occupation": "Software Engineer",
      "rank": 1
    }},
    {{
      "occupation": "Data Scientist",
      "rank": 2
    }},
    {{
      "occupation": "Web Developer",
      "rank": 3
    }},
    {{
      "occupation": "System Administrator",
      "rank": 4
    }}
  ],
  "final_choice": {{
    "occupation": "Software Engineer",
    "reasoning": "Based on the job description, the primary responsibilities involve developing and maintaining software applications, which aligns perfectly with the skills and expertise of a Software Engineer."
  }}
}}"""
    
    return prompt_text

def process_single_job(client, model, job_data, generate_content_config):
    """Process a single job entry"""
    ReferenceNumber = job_data['ReferenceNumber']
    job_title = job_data['job_title']
    job_description = job_data['job_description']
    job_requirements = job_data['job_requirements']
    potential_occupations = job_data['potential_occupations']
    
    print(f"\n{'='*50}")
    print(f"Processing Job ID: {ReferenceNumber}")
    print(f"Job Title: {job_title}")
    print(f"{'='*50}")
    
    # Create the prompt for this specific job
    prompt_text = create_job_prompt(ReferenceNumber, job_title, job_description, job_requirements, potential_occupations)
    text_part = types.Part.from_text(text=prompt_text)
    
    contents = [
        types.Content(
            role="user",
            parts=[text_part]
        )
    ]
    
    print(f"\nResponse for Job ID {ReferenceNumber}:")
    print("-" * 30)
    
    # Collect the response text
    response_text = ""
    
    try:
        # Generate and collect the response
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")
            response_text += chunk.text
        
        print("\n" + "-" * 30)
        
        # Parse and return the response
        parsed_response, is_valid = parse_response(ReferenceNumber, response_text)
        return parsed_response, is_valid
        
    except Exception as e:
        print(f"\n✗ Error generating response for job {ReferenceNumber}: {str(e)}")
        error_response = {
            "ReferenceNumber": ReferenceNumber,
            "error": "Generation failed",
            "error_details": str(e),
            "job_title": job_title
        }
        return error_response, False

def generate_for_all_jobs(json_file_path, output_file="job_occupations_combined.json"):
    """Main function to process all jobs from JSON file and save to single JSON array"""
    # Initialize the client
    client = genai.Client(
        vertexai=True,
        project="ihu-access",
        location="global",
    )
    
    # System instruction
    si_text = """You are an expert job placement specialist. You will receive a job ID, job title, job description, job requirements, and a list of potential occupations. Your task is to rank the potential occupations from most to least fitting based on the job description and requirements. Then, from the top three occupations, select the one that best fits the job title, job description, and job requirements. The output should be a JSON file containing the job ID, ranked list of occupations, the final choice, and the reasoning behind the choice."""
    
    model = "gemini-2.5-pro"
    
    # Configuration
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        seed=0,
        max_output_tokens=65535,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
        system_instruction=[types.Part.from_text(text=si_text)],
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
    )
    
    # Load job data
    try:
        job_data_list = load_job_data(json_file_path)
        print(f"Loaded {len(job_data_list)} jobs from {json_file_path}")
        print(f"Output will be saved to: {output_file}")
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return
    
    # CHANGED: Initialize lists to collect all responses
    all_responses = []
    successful_jobs = []
    failed_jobs = []
    
    # Process each job
    for i, job_data in enumerate(job_data_list, 1):
        try:
            print(f"\n\nProcessing job {i}/{len(job_data_list)}")
            response, is_valid = process_single_job(client, model, job_data, generate_content_config)
            
            # CHANGED: Add response to the array regardless of validity
            all_responses.append(response)
            
            if is_valid:
                successful_jobs.append({
                    'ReferenceNumber': job_data['ReferenceNumber'],
                    'job_title': job_data['job_title']
                })
            else:
                failed_jobs.append({
                    'ReferenceNumber': job_data.get('ReferenceNumber', 'unknown'),
                    'job_title': job_data.get('job_title', 'unknown'),
                    'error': response.get('error', 'Unknown error')
                })
                
        except Exception as e:
            # CHANGED: Even for complete failures, add an error entry to the array
            error_info = {
                'ReferenceNumber': job_data.get('ReferenceNumber', 'unknown'),
                'job_title': job_data.get('job_title', 'unknown'),
                'error': 'Processing failed',
                'error_details': str(e)
            }
            all_responses.append(error_info)
            failed_jobs.append(error_info)
            print(f"Error processing job {job_data.get('ReferenceNumber', 'unknown')}: {str(e)}")
            continue
    
    # CHANGED: Save all responses to a single JSON file
    save_success = save_all_responses_to_file(all_responses, output_file)
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total jobs processed: {len(job_data_list)}")
    print(f"Successful: {len(successful_jobs)}")
    print(f"Failed: {len(failed_jobs)}")
    print(f"File save: {'✓ Success' if save_success else '✗ Failed'}")
    
    if successful_jobs:
        print(f"\n✓ Successfully processed jobs:")
        for job in successful_jobs:
            print(f"  - {job['ReferenceNumber']}: {job['job_title']}")
    
    if failed_jobs:
        print(f"\n✗ Failed jobs:")
        for job in failed_jobs:
            print(f"  - {job['ReferenceNumber']}: {job['job_title']} ({job['error']})")
    
    print(f"\nAll responses saved in: {output_file}")
    print(f"{'='*60}")
    
    return all_responses, successful_jobs, failed_jobs

# Usage example
if __name__ == "__main__":
    # Replace 'jobs.json' with the path to your JSON file
    json_file_path = "jobs.json"
    output_file = "job_responses_combined.json"  # CHANGED: Single output file instead of directory
    
    all_responses, successful, failed = generate_for_all_jobs(json_file_path, output_file)

####### ------------------------------------------------------- #######
####### ------- Ask LLM to pick key and required skills ------- #######
####### ------------------------------------------------------- #######

def load_job_data(json_file_path):
    """Load job data from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def parse_response(ReferenceNumber, response_text):
    """Parse and validate the model response"""
    try:
        # Try to parse the response as JSON
        parsed_response = json.loads(response_text.strip())
        print(f"✓ Successfully parsed JSON response for job {ReferenceNumber}")
        return parsed_response, True
    except json.JSONDecodeError as e:
        # If it's not valid JSON, create an error entry
        error_response = {
            "ReferenceNumber": ReferenceNumber,
            "error": "Invalid JSON response",
            "error_details": str(e),
            "raw_response": response_text.strip()
        }
        print(f"⚠ Failed to parse JSON for job {ReferenceNumber}: {str(e)}")
        return error_response, False

def save_all_responses_to_file(all_responses, output_file="job_skills_combined.json"):
    """Save all responses to a single JSON array file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved all responses to: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to save combined file: {str(e)}")
        return False
    """Load job data from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_job_prompt(ReferenceNumber, job_title, job_description, job_requirements, potential_skills):
    """Create the prompt text for a specific job"""
    skills_text = "\n".join([f"- {occ}" for occ in potential_skills])
    
    prompt_text = f"""Here is the job description:
{job_description}

Here is the job title:
{job_title}

Here are the job requirements:
{job_requirements}

Here is a list of potential skills:
{skills_text}

Here is the reference number:
{ReferenceNumber}

Follow these rules:
* You must scan the job title, job description, and job requirements.
* You must rank the skills by how important they are to the job.
* You must choose all skills that seem required.
* You must choose the top 5 of those that just seem very important to the job and make sure they are ordered by importance.
* You must output the results in a structured JSON format.

Do not make up any skills. Only use the provided potential skills.

Here is an example of the desired output format:
{
  "required_skills": ["skill1", "skill2", "skill3"],
  "top_5_important_skills": ["skill4", "skill5", "skill6", "skill7", "skill8"]
}"""
    
    return prompt_text

def process_single_job(client, model, job_data, generate_content_config):
    """Process a single job entry"""
    ReferenceNumber = job_data['ReferenceNumber']
    job_title = job_data['job_title']
    job_description = job_data['job_description']
    job_requirements = job_data['job_requirements']
    potential_skills = job_data['potential_skills']
    
    print(f"\n{'='*50}")
    print(f"Processing Job ID: {ReferenceNumber}")
    print(f"Job Title: {job_title}")
    print(f"{'='*50}")
    
    # Create the prompt for this specific job
    prompt_text = create_job_prompt(ReferenceNumber, job_title, job_description, job_requirements, potential_skills)
    text_part = types.Part.from_text(text=prompt_text)
    
    contents = [
        types.Content(
            role="user",
            parts=[text_part]
        )
    ]
    
    print(f"\nResponse for Job ID {ReferenceNumber}:")
    print("-" * 30)
    
    # Collect the response text
    response_text = ""
    
    try:
        # Generate and collect the response
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")
            response_text += chunk.text
        
        print("\n" + "-" * 30)
        
        # Parse and return the response
        parsed_response, is_valid = parse_response(ReferenceNumber, response_text)
        return parsed_response, is_valid
        
    except Exception as e:
        print(f"\n✗ Error generating response for job {ReferenceNumber}: {str(e)}")
        error_response = {
            "ReferenceNumber": ReferenceNumber,
            "error": "Generation failed",
            "error_details": str(e),
            "job_title": job_title
        }
        return error_response, False

def generate_for_all_jobs(json_file_path, output_file="job_skills_combined.json"):
    """Main function to process all jobs from JSON file and save to single JSON array"""
    # Initialize the client
    client = genai.Client(
        vertexai=True,
        project="ihu-access",
        location="global",
    )
    
    # System instruction
    si_text = """You are an expert job recruiter. You are excellent at scanning job descriptions and extracting the most important skills for the job."""
    
    model = "gemini-2.5-pro"
    
    # Configuration
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        seed=0,
        max_output_tokens=65535,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
        system_instruction=[types.Part.from_text(text=si_text)],
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
    )
    
    # Load job data
    try:
        job_data_list = load_job_data(json_file_path)
        print(f"Loaded {len(job_data_list)} jobs from {json_file_path}")
        print(f"Output will be saved to: {output_file}")
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return
    
    # CHANGED: Initialize lists to collect all responses
    all_responses = []
    successful_jobs = []
    failed_jobs = []
    
    # Process each job
    for i, job_data in enumerate(job_data_list, 1):
        try:
            print(f"\n\nProcessing job {i}/{len(job_data_list)}")
            response, is_valid = process_single_job(client, model, job_data, generate_content_config)
            
            # CHANGED: Add response to the array regardless of validity
            all_responses.append(response)
            
            if is_valid:
                successful_jobs.append({
                    'ReferenceNumber': job_data['ReferenceNumber'],
                    'job_title': job_data['job_title']
                })
            else:
                failed_jobs.append({
                    'ReferenceNumber': job_data.get('ReferenceNumber', 'unknown'),
                    'job_title': job_data.get('job_title', 'unknown'),
                    'error': response.get('error', 'Unknown error')
                })
                
        except Exception as e:
            # CHANGED: Even for complete failures, add an error entry to the array
            error_info = {
                'ReferenceNumber': job_data.get('ReferenceNumber', 'unknown'),
                'job_title': job_data.get('job_title', 'unknown'),
                'error': 'Processing failed',
                'error_details': str(e)
            }
            all_responses.append(error_info)
            failed_jobs.append(error_info)
            print(f"Error processing job {job_data.get('ReferenceNumber', 'unknown')}: {str(e)}")
            continue
    
    # CHANGED: Save all responses to a single JSON file
    save_success = save_all_responses_to_file(all_responses, output_file)
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total jobs processed: {len(job_data_list)}")
    print(f"Successful: {len(successful_jobs)}")
    print(f"Failed: {len(failed_jobs)}")
    print(f"File save: {'✓ Success' if save_success else '✗ Failed'}")
    
    if successful_jobs:
        print(f"\n✓ Successfully processed jobs:")
        for job in successful_jobs:
            print(f"  - {job['ReferenceNumber']}: {job['job_title']}")
    
    if failed_jobs:
        print(f"\n✗ Failed jobs:")
        for job in failed_jobs:
            print(f"  - {job['ReferenceNumber']}: {job['job_title']} ({job['error']})")
    
    print(f"\nAll responses saved in: {output_file}")
    print(f"{'='*60}")
    
    return all_responses, successful_jobs, failed_jobs

# ============================================================================
# EXECUTE
# ============================================================================
if __name__ == "__main__":
    # Replace 'jobs.json' with the path to your JSON file
    json_file_path = "jobs.json"
    output_file = "job_responses_combined.json"  # CHANGED: Single output file instead of directory
    
    all_responses, successful, failed = generate_for_all_jobs(json_file_path, output_file)


