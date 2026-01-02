r"""
HOW TO RUN

--> It is better to move all files to Downloads (manually) as the script will run faster than on OneDrive

# 1) Occupations pass
python scripts/3_llm_reranker/3_1_LLM_pick_skills_full_details.py `
  --process occupations `
  --input "C:\Users\jasmi\Downloads\bert_cleaned.json" `
  --output "C:\Users\jasmi\Downloads\llm_opportunity_responses_occupations.ndjson" `
  --project ihu-access --location global --model gemini-2.5-pro `
  --ndjson --compact-after `
  --max_output_tokens 4000
  
# 2) Skills OPTIONAL pool
python scripts/3_llm_reranker/3_1_LLM_pick_skills_full_details.py `
  --process skills_optional `
  --input "C:\Users\jasmi\Downloads\bert_cleaned_with_occupation_skills_firstbatch.json" `
  --output "C:\Users\jasmi\Downloads\llm_opportunity_responses_skills_optional.ndjson" `
  --ndjson --compact-after  

# 3) Skills ESSENTIAL pool
python scripts/3_llm_reranker/3_1_LLM_pick_skills_full_details.py `
  --process skills_essential `
  --input "C:\Users\jasmi\Downloads\bert_cleaned_with_occupation_skills_firstbatch.json" `
  --output "C:\Users\jasmi\Downloads\llm_opportunity_responses_skills_essential.ndjson" `
  --ndjson --compact-after

# (Convenience) Run both skills passes back-to-back with one command
python scripts/3_llm_reranker/3_1_LLM_pick_skills_full_details.py `
  --process skills_both `
  --input "C:\Users\jasmi\Downloads\bert_cleaned_with_occupation_final.json" `
  --output-optional "C:\Users\jasmi\Downloads\llm_opportunity_responses_skills_optional.ndjson" `
  --output-essential "C:\Users\jasmi\Downloads\llm_opportunity_responses_skills_essential.ndjson" `
  --ndjson --compact-after

"""

import argparse
import json
import ijson
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Vertex AI
import vertexai
from google.api_core import exceptions as google_exceptions
from vertexai.generative_models import GenerationConfig, GenerativeModel


# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_or_initialize_results(output_file: Path) -> Tuple[List[Dict[str, Any]], set]:
    """
    JSON-array mode only: load existing array and return (records, processed_ids).
    Treat 'Skipped' as terminal so we don't retry them endlessly.
    """
    if output_file.exists():
        logging.info(f"Found existing output file. Loading previous results from {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                processed_ids = {
                    r['opportunity_ref_id']
                    for r in existing_results
                    if ('error' not in r) or (r.get('error') == 'Skipped')
                }
                logging.info(f"Loaded {len(existing_results)} previous results. Found {len(processed_ids)} already processed jobs.")
                return existing_results, processed_ids
            except json.JSONDecodeError:
                logging.warning(f"Could not decode JSON from {output_file}. Starting fresh.")
                return [], set()
    return [], set()


def save_results_to_file(all_responses: List[Dict[str, Any]], output_file: Path):
    """
    JSON-array mode only: safely writes data by using a temporary file and an atomic replace.
    Includes retry logic to handle file locks from sync services like OneDrive.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + '.tmp')

    max_retries = 5
    for attempt in range(max_retries):
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(all_responses, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            temp_file.replace(output_file)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Save failed on attempt {attempt + 1}/{max_retries}. Retrying in 1 second. Error: {e}")
                time.sleep(1)
            else:
                logging.error(f"FATAL: Could not save results to {output_file} after {max_retries} attempts. Error: {e}")
                if temp_file.exists():
                    temp_file.unlink()


# ============================================================================
# 2. NDJSON HELPERS
# ============================================================================

def append_ndjson_line(path: Path, obj: dict):
    """
    Append one JSON object as a single NDJSON line, fsync for durability.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_processed_from_ndjson(path: Path) -> Tuple[set, int]:
    """
    Return (processed_ids, total_lines_or_items). Treat 'Skipped' as terminal.
    Detects if the file is actually a JSON array and imports from it.
    """
    processed, total = set(), 0
    if not path.exists():
        return processed, total

    with open(path, "r", encoding="utf-8") as f:
        # Peek the first non-whitespace char
        first_non_ws = None
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break
        f.seek(0)

        # If it's a JSON array, load and extract IDs
        if first_non_ws == '[':
            try:
                arr = json.load(f)
                total = len(arr)
                for rec in arr:
                    ref_id = rec.get("opportunity_ref_id")
                    if ref_id and (('error' not in rec) or (rec.get('error') == 'Skipped')):
                        processed.add(ref_id)
                logging.info(f"Detected JSON-array file while in NDJSON mode; "
                             f"imported {len(processed)} processed IDs from array.")
                return processed, total
            except Exception:
                # If array parsing fails, fall back to line-by-line
                f.seek(0)

        # NDJSON path: one object per line
        for total, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ref_id = rec.get("opportunity_ref_id")
                if ref_id and (('error' not in rec) or (rec.get('error') == 'Skipped')):
                    processed.add(ref_id)
            except Exception:
                continue

    return processed, total


def compact_ndjson_to_json_array(ndjson_file: Path, out_file: Optional[Path] = None) -> Path:
    """
    Deduplicate by opportunity_ref_id, keep the LAST occurrence,
    and write a compact JSON array to out_file (or ndjson_file.with_suffix('.compact.json')).
    """
    if out_file is None:
        out_file = ndjson_file.with_suffix(".compact.json")

    latest = {}
    if ndjson_file.exists():
        with open(ndjson_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ref = rec.get("opportunity_ref_id")
                    if ref:
                        latest[ref] = rec  # last one wins
                except Exception:
                    continue

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(list(latest.values()), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(out_file)
    return out_file


# ============================================================================
# 3. PROMPT CREATION
# ============================================================================

def _format_labeled_list(items: List[Dict[str, str]]) -> str:
    """Format list of {'label','description'} to '- **Label**: desc' lines."""
    return "\n".join(
        [f"- **{x.get('label','').strip()}**: {x.get('description','').strip()}" for x in items if x.get('label')]
    )

def create_occupation_prompt(full_details: str, potential_occupations: List[Dict[str, str]]) -> str:
    """
    Occupation prompt (unchanged style).
    """
    occupations_text = _format_labeled_list(potential_occupations)

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
2.  Rank the potential occupations from most to least likely to be the best fit, and return the top 20 in "ranked_occupations".
3.  Choose the single occupation that best represents the core function of the opportunity. final_choice.occupation must appear in ranked_occupations.
4.  Provide clear, concise reasoning for your final choice, referencing specific parts of the opportunity details and how they align with the official occupation description.
5.  **Use Provided List Only**: Do not make up any occupations. Choose only from the list shown above.
6.  **Strict JSON Output**: Format your response as a single, valid JSON object. Do not include any explanatory text or markdown outside the JSON.
7.  **Verbatim Matching**: The occupation string you return must be an **exact, verbatim** copy of a occupation label from the list above.

Do not make up any occupations. If there is a list provided, only use those.

**Example JSON Output:**
{{
  "ranked_occupations": [
    {{"occupation": "Software Engineer", "rank": 1}},
    {{"occupation": "Data Scientist", "rank": 2}}
  ],
  "final_choice": {{
    "occupation": "Software Engineer",
    "reasoning": "Short justification..."
  }}
}}"""


def create_skills_prompt_essential_with_descriptions(
    full_details: str,
    potential_skills: List[Dict[str, str]]
) -> str:
    """
    Essential skills prompt (keeps original style/voice but narrows the task to 'required' only).
    """
    skills_text = _format_labeled_list(potential_skills)

    return f"""You are an expert technical recruiter with deep knowledge of the job market. Your task is to analyze a job opportunity and identify skills that are **non-negotiable at the time of application** (i.e., strictly essential to be hired).

**Opportunity Details:**
---
{full_details}
---

**List of Potential Skills to Choose From (each with description):**
{skills_text}

**Instructions:**
1.  **Analyze the Role**: Carefully read the opportunity details to understand what the employer is looking for in a candidate *at the time of application*.
2.  **Identify "Essential Skills"**: From the provided list, select only the skills that an employer **must** see to consider hiring the candidate. These are the non-negotiable prerequisites.
3.  **Exclude "Learn-after-hire"**: Do **not** include skills that the job ad states or implies will be taught after hiring.
4.  **Use Provided List Only**: Do not make up any skills. Choose only from the list shown above.
5.  **Strict JSON Output**: Format your response as a single, valid JSON object. Do not include any explanatory text or markdown outside the JSON.
6.  **Verbatim Matching**: The skill strings you return must be an **exact, verbatim** copy of a skill label from the list above.

**Output JSON shape:**
{{
  "essential_skills": ["...", "..."]
}}"""


def create_skills_prompt_optional_with_descriptions(
    full_details: str,
    potential_skills: List[Dict[str, str]]
) -> str:
    """
    Optional/important skills prompt (keeps original style/voice but narrows the task to 'important but not strictly required').
    """
    skills_text = _format_labeled_list(potential_skills)

    return f"""You are an expert technical recruiter with deep knowledge of the job market. Your task is to analyze a job opportunity and select skills that are **important for the role but not strictly required to be hired**.

**Opportunity Details:**
---
{full_details}
---

**List of Potential Skills to Choose From (each with description):**
{skills_text}

**Instructions:**
1.  **Analyze the Role**: Carefully read the opportunity details.
2.  **Identify "Optional Skills"**: From the provided list, select skills that would make an applicant **stronger or more competitive**, while **not being strictly required** to be hired.
3.  **Order and Coverage**: Please return a reasonable number of optional skills, ordered from most to least important for the role.
4.  **Exclude Non-skills**: Exclude benefits, conditions, years of experience counts, employer names, and traits.
5.  **Use Provided List Only**: Do not make up any skills. Choose only from the list shown above.
6.  **Strict JSON Output**: Return a single valid JSON object (no additional text or markdown).
7.  **Verbatim Matching**: The strings must exactly match a skill label from the list above.

**Output JSON shape:**
{{
  "optional_skills": ["...", "...", "..."]
}}"""


# ============================================================================
# 4. ROBUST JSON PARSING
# ============================================================================

from json import JSONDecodeError

def _strip_code_fences(text: str) -> str:
    """
    Remove ```...``` fences if present, tolerating optional language markers.
    """
    s = (text or "").strip()
    if s.startswith("```") and s.endswith("```"):
        inner = s[3:-3].lstrip()
        # Drop an optional first line like 'json'
        if "\n" in inner:
            first, rest = inner.split("\n", 1)
            # If the first line is a likely language token, drop it
            if first.strip().lower() in {"json", "js", "javascript", "python"}:
                return rest
        return inner
    return s


def _extract_first_json_block(s: str) -> Optional[str]:
    """
    Return the first top-level JSON object/array as a substring, or None.
    Handles quotes and escapes so braces/brackets inside strings are ignored.
    """
    i, n = 0, len(s)
    in_string = False
    escape = False
    start: Optional[int] = None
    depth_obj = 0  # {}
    depth_arr = 0  # []

    while i < n:
        ch = s[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        # not in string
        if ch == '"':
            in_string = True
            i += 1
            continue

        if ch == '{':
            if start is None:
                start = i
            depth_obj += 1
        elif ch == '[':
            if start is None:
                start = i
            depth_arr += 1
        elif ch == '}':
            if depth_obj > 0:
                depth_obj -= 1
        elif ch == ']':
            if depth_arr > 0:
                depth_arr -= 1

        if start is not None and depth_obj == 0 and depth_arr == 0:
            # closed the first complete JSON value
            end = i + 1
            return s[start:end]

        i += 1

    return None


def parse_llm_response(response_text: str) -> Tuple[Dict[str, Any], bool]:
    """
    1) Try direct JSON load (after removing code fences).
    2) If that fails, extract the first complete top-level JSON block and parse that.
    3) If all fails, return an error payload with raw text for debugging.
    """
    s = _strip_code_fences(response_text or "")

    try:
        parsed = json.loads(s.strip())
        return parsed, True
    except JSONDecodeError:
        pass

    candidate = _extract_first_json_block(s)
    if candidate:
        try:
            parsed = json.loads(candidate)
            return parsed, True
        except JSONDecodeError as e:
            return {
                "error": "Invalid JSON response",
                "error_details": f"First JSON block could not be parsed: {e}",
                "raw_response": response_text
            }, False

    return {
        "error": "Invalid JSON response",
        "error_details": "No JSON object/array found in model response.",
        "raw_response": response_text
    }, False


# ============================================================================
# 5. MODEL CALLS & PER-JOB PROCESSING
# ============================================================================

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
        google_exceptions.DeadlineExceeded,
        google_exceptions.Aborted,
    )
    for attempt in range(max_retries):
        try:
            response = client.generate_content(contents=contents, generation_config=config)
            return response.text
        except retriable_errors as e:
            logging.warning(f"API call failed with retriable error: {e}. Attempt {attempt + 1} of {max_retries}.")
            if attempt + 1 == max_retries:
                logging.error("Max retries reached. Failing the API call.")
                raise
            # exponential backoff with jitter
            sleep_s = (2 ** attempt) + (0.25 * (attempt + 1))
            time.sleep(sleep_s)
    raise Exception("API call failed after all retries.")


def process_single_job(
    client: GenerativeModel,
    job_data: Dict[str, Any],
    gen_config: GenerationConfig,
    process_type: str,
) -> Tuple[Dict[str, Any], bool]:
    opportunity_group_id = job_data['opportunity_group_id']
    opportunity_ref_id = job_data['opportunity_ref_id']
    opportunity_title = job_data['opportunity_title']
    base_response = {
        "opportunity_group_id": opportunity_group_id,
        "opportunity_ref_id": opportunity_ref_id,
        "opportunity_title": opportunity_title,
        "process_type": process_type,
    }

    prompt_text = None

    if process_type == 'occupations':
        occ_labels = job_data.get('potential_occupations', [])
        occ_descs = job_data.get('potential_occupations_descriptions', [])
        if not occ_labels:
            logging.warning(f"Skipping OCCUPATION for {opportunity_ref_id}: No potential occupations provided.")
            base_response.update({"error": "Skipped", "error_details": "No potential occupations in input."})
            return base_response, False

        # pad descriptions to match length
        occ_descs = list(occ_descs) if occ_descs else []
        if len(occ_descs) < len(occ_labels):
            occ_descs.extend([''] * (len(occ_labels) - len(occ_descs)))

        combined_occupations = [{'label': l, 'description': d} for l, d in zip(occ_labels, occ_descs)]
        prompt_text = create_occupation_prompt(job_data['full_details'], combined_occupations)

    elif process_type == 'skills_essential':
        labels = job_data.get('potential_essential_skills', [])
        descs  = job_data.get('potential_essential_skills_descriptions', [])
        if not labels:
            logging.warning(f"Skipping SKILLS_ESSENTIAL for {opportunity_ref_id}: No potential_essential_skills provided.")
            base_response.update({"error": "Skipped", "error_details": "No potential_essential_skills in input."})
            return base_response, False

        descs = list(descs) if descs else []
        if len(descs) < len(labels):
            descs.extend([''] * (len(labels) - len(descs)))
        skills_with_desc = [{'label': l, 'description': d} for l, d in zip(labels, descs)]

        prompt_text = create_skills_prompt_essential_with_descriptions(
            job_data['full_details'], skills_with_desc
        )

    elif process_type == 'skills_optional':
        labels = job_data.get('potential_optional_skills', [])
        descs  = job_data.get('potential_optional_skills_descriptions', [])
        if not labels:
            logging.warning(f"Skipping SKILLS_OPTIONAL for {opportunity_ref_id}: No potential_optional_skills provided.")
            base_response.update({"error": "Skipped", "error_details": "No potential_optional_skills in input."})
            return base_response, False

        descs = list(descs) if descs else []
        if len(descs) < len(labels):
            descs.extend([''] * (len(labels) - len(descs)))
        skills_with_desc = [{'label': l, 'description': d} for l, d in zip(labels, descs)]

        prompt_text = create_skills_prompt_optional_with_descriptions(
            job_data['full_details'], skills_with_desc
        )

    else:
        raise ValueError("Invalid process_type specified.")

    # Call model
    try:
        logging.info(f"Generating LLM response for {opportunity_ref_id}...")
        response_text = call_vertexai_with_retry(client, prompt_text, gen_config)
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
# 6. MAIN EXECUTION (STREAMING/RETRY + NDJSON/ARRAY MODES)
# ============================================================================

def process_all_jobs(
    input_file: Path,
    output_file: Path,
    process_type: str,
    config: Dict[str, Any]
):
    """
    Memory-efficient main loop:
      - NDJSON mode: append one line per job (fastest) + optional end compaction.
      - JSON-array mode: skip already-processed, rewrite array (your original behavior).
    """
    logging.info(f"\n{'='*60}\n--- STARTING {process_type.upper()} PROCESSING ---\n{'='*60}")

    # Init Vertex AI
    vertexai.init(project=config['project'], location=config['location'])
    client = GenerativeModel(config['model_name'])
    generation_config = GenerationConfig(
        temperature=config['temperature'],
        top_p=config['top_p'],
        max_output_tokens=config['max_output_tokens'],
    )

    # ---- NDJSON MODE ----
    if config.get("ndjson"):
        processed_ids, total_lines = load_processed_from_ndjson(output_file)
        logging.info(f"NDJSON mode: {len(processed_ids)} processed across {total_lines} lines so far in {output_file}")

        with open(input_file, 'rb') as f:
            for i, job_data in enumerate(ijson.items(f, 'item'), 1):
                ref = job_data['opportunity_ref_id']
                title = job_data.get('opportunity_title', '')
                if ref in processed_ids:
                    logging.info(f"[NDJSON] Skipping already-processed {ref}: {title}")
                    continue

                resp, _ok = process_single_job(client, job_data, generation_config, process_type)
                append_ndjson_line(output_file, resp)
                processed_ids.add(ref)

        if config.get("compact_after"):
            out = compact_ndjson_to_json_array(output_file)
            logging.info(f"Compacted NDJSON into JSON array: {out}")
        return

    # ---- JSON-ARRAY MODE (fallback) ----
    all_responses, processed_ids = load_or_initialize_results(output_file)
    response_map = {res['opportunity_ref_id']: res for res in all_responses}

    # track how many unsaved items since last write, and announce plan
    dirty_since_save = 0
    logging.info(
        f"JSON-array mode with batch saves: will write to disk every {config.get('save_every', 25)} new jobs."
    )

    processed_count = 0
    with open(input_file, 'rb') as f:
        jobs_iterator = ijson.items(f, 'item')
        for i, job_data in enumerate(jobs_iterator, 1):
            processed_count = i
            job_ref_id = job_data['opportunity_ref_id']
            job_title = job_data['opportunity_title']

            logging.info(f"\n--- Processing job {i} ({job_ref_id}: {job_title}) ---")

            if job_ref_id in processed_ids:
                logging.info(f"Job {job_ref_id} has already been processed. Skipping.")
                continue

            response, _ = process_single_job(client, job_data, generation_config, process_type)

            # Save/update in-memory map
            response_map[job_ref_id] = response
            dirty_since_save += 1

            # NEW: batch save every N jobs
            if dirty_since_save >= max(1, int(config.get('save_every', 25))):
                total = len(response_map)
                logging.info(f"[SAVE] Batch save triggered after {dirty_since_save} new jobs "
                             f"(total records now {total}). Writing {output_file} ...")
                save_results_to_file(list(response_map.values()), output_file)
                logging.info(f"[SAVE] Done: {output_file} now has {total} records.")
                dirty_since_save = 0

    # NEW: final flush for any remainder
    if dirty_since_save > 0:
        total = len(response_map)
        logging.info(f"[SAVE] Final save of remaining {dirty_since_save} jobs "
                     f"(total records {total}). Writing {output_file} ...")
        save_results_to_file(list(response_map.values()), output_file)
        logging.info(f"[SAVE] Done: {output_file} now has {total} records.")


    # Summary
    final_results = list(response_map.values())
    successful_jobs = [r for r in final_results if 'error' not in r]
    failed_jobs = [r for r in final_results if 'error' in r]

    logging.info(f"\n\n{'='*60}\n{process_type.upper()} PROCESSING SUMMARY\n{'='*60}")
    logging.info(f"Total jobs streamed: {processed_count}")
    logging.info(f"Total results in output file: {len(final_results)}")
    logging.info(f"Successful: {len(successful_jobs)}")
    logging.info(f"Failed/Skipped: {len(failed_jobs)}")
    if failed_jobs:
        logging.warning("\n--- Failed/Skipped Jobs ---")
        for job in failed_jobs:
            error_details = job.get('error_details', 'N/A')
            logging.warning(f"  - {job.get('opportunity_ref_id','?')}: {job.get('opportunity_title','?')} ({job.get('error','?')}: {str(error_details)[:200]})")
    logging.info(f"\nFinal results are saved in: {output_file}\n{'='*60}")


def run_with_retry(process_type: str, input_file: Path, output_file: Path, config: Dict[str, Any]):
    """
    Drive one pass and re-run until all items are processed or max_reruns reached.
    Works for both NDJSON and JSON-array modes.
    """
    # Count total jobs once
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

        if config.get("ndjson"):
            # Read processed IDs from NDJSON
            processed_ids, _ = load_processed_from_ndjson(output_file)
        else:
            # JSON-array mode
            _, processed_ids = load_or_initialize_results(output_file)

        if len(processed_ids) >= total_jobs:
            logging.info(f"All {total_jobs} jobs successfully processed for {process_type}. Run complete.")
            break
        else:
            rerun_count += 1
            logging.warning(f"Run {rerun_count}/{config['max_reruns']} incomplete. {len(processed_ids)}/{total_jobs} jobs processed. Retrying in 10s...")
            time.sleep(10)
    else:
        logging.error(f"Max reruns reached for {process_type}, but not all jobs were processed. Please check for persistent errors.")


# ============================================================================
# 7. CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM post-processing for occupations and/or skills (optional/essential) with streaming + retry."
    )
    parser.add_argument("--process",
                        choices=["occupations", "skills_optional", "skills_essential", "skills_both"],
                        required=True,
                        help="Which pass to run.")
    parser.add_argument("--input", type=Path, required=True, help="Path to input JSON (array) file.")
    parser.add_argument("--output", type=Path, help="Output file (for single-pass runs).")
    parser.add_argument("--output-optional", type=Path, help="Output for skills_optional (when --process skills_both).")
    parser.add_argument("--output-essential", type=Path, help="Output for skills_essential (when --process skills_both).")

    # NDJSON / compaction options
    parser.add_argument("--ndjson", action="store_true",
                        help="Append NDJSON lines instead of rewriting a JSON array (fastest).")
    parser.add_argument("--compact-after", action="store_true",
                        help="After the run (NDJSON mode), write a compact JSON array next to the NDJSON file.")
    
    # JSON-array batching (ignored in --ndjson mode)
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="JSON-array mode only: write the output file after every N new jobs instead of after each job (default: 25)."
    )

    # Model / project knobs with sane defaults
    parser.add_argument("--project", default="ihu-access")
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", dest="model_name", default="gemini-2.5-pro")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_output_tokens", type=int, default=8192)
    parser.add_argument("--max_reruns", type=int, default=5)

    args = parser.parse_args()

    # Defaults for outputs if not provided
    if args.process in ("occupations", "skills_optional", "skills_essential"):
        if not args.output:
            suffix = {
                "occupations": "job_responses_occupations.json",
                "skills_optional": "job_responses_skills_optional.ndjson" if args.ndjson else "job_responses_skills_optional.json",
                "skills_essential": "job_responses_skills_essential.ndjson" if args.ndjson else "job_responses_skills_essential.json",
            }[args.process]
            args.output = args.input.parent / suffix

    if args.process == "skills_both":
        if not args.output_optional:
            args.output_optional = args.input.parent / ("job_responses_skills_optional.ndjson" if args.ndjson else "job_responses_skills_optional.json")
        if not args.output_essential:
            args.output_essential = args.input.parent / ("job_responses_skills_essential.ndjson" if args.ndjson else "job_responses_skills_essential.json")

    config = {
        "model_name": args.model_name,
        "project": args.project,
        "location": args.location,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "max_reruns": args.max_reruns,
        "ndjson": args.ndjson,
        "compact_after": args.compact_after,
        "save_every": args.save_every, 
    }


    # Dispatch
    if args.process == "occupations":
        run_with_retry("occupations", args.input, args.output, config)

    elif args.process == "skills_optional":
        run_with_retry("skills_optional", args.input, args.output, config)

    elif args.process == "skills_essential":
        run_with_retry("skills_essential", args.input, args.output, config)

    elif args.process == "skills_both":
        run_with_retry("skills_optional", args.input, args.output_optional, config)
        run_with_retry("skills_essential", args.input, args.output_essential, config)

    else:
        raise ValueError("Unknown process type")


if __name__ == "__main__":
    main()
