# Project set up
# FIRST: Maker sure we are in the correct working directory
Set-Location "C:\Users\jasmi\Documents\GitHub\tabiya-livelihoods-classifier"

# STEP 1: Once opened VS Code, open Terminal (View > Terminal or CTRL + ~) and run:
.\classifier_setup.ps1
# That should activate virtual environment which I set up previously as per https://github.com/tabiya-tech/tabiya-livelihoods-classifier/blob/main/README.md#set-up-virtualenv

# STEP 2: Get new jobs via Harambee API and update jobs database
jobs_API_and_formatting.py

# STEP 3: View > Command Palette > Python: Select interpreter > CHOOSE VENV

# STEP 4: In Powershell in the virtual environment, run
python entity_extraction_loop.py
# TODO must change to batching in system, i.e. so that it only re-analyzes new entries

# STEP 5: Once done, clean the output using the cleaning script
clean_bert_results.py

# STEP 6: Then: Give to LLMs (script to be written)
LLM_pick_skills_full_details.py

# STEP 7: Reshape LLM output and merge opportunity information back in
clean_LLM_create_opp-db.py

# STEP 7.5: For robustness, double check the pilot database for any jobs that couldn't map the skills to uuids. 
debug_find_jobs_hallucinated_skills.py
# STEP 7.6: Then rerun LLM file with that bert_cleaned_rerun json
LLM_pick_skills_full_details_rerun.py


# STEP 8: Clean pre-existing jobseeker skillsets for jobseeker database
reshape_jobseeker_database.py

# Finally, run computation (will actually be one on cloud, so this file is just for testing)
match_skills_compute.py


# Things I could improve in the whole pipeline # Today July 30
# In Bert cleaning, check if Bert found any occupations in job description, and move those over to the occupations list (delete duplicates)
# In bert cleaning, add occupation parents to occupation list (but add directly, no need to run LLM twice here) PROBLEM: uuid in bert file is not same as in skill_hierarchy filepip install ijson
# In bert cleaning, add another list item with only the skill parent (delete duplicates) --> add to LLM script to use this instead


# LATER
# Ask Gemini to help improve my prompt, note that I specifically want to get at "key necessary skills and requirements the employer would be looking for in an applicant"
# Do I currently tell LLM how many skills to return? Do i have it choose how many are most important?

# WEIGHTED MATCHING idea
# Weighted Matching: Using these matrices to enable a multi-level matching process. An initial, coarse-grained match can be performed at a higher level of the hierarchy (e.g., Level 2 occupations to Level 2 skills). This provides a broad-strokes view of how a candidate's skill set aligns with a job's general requirements. This aggregated view is computationally less intensive and easier for users to interpret. The system can then allow for a "drill-down" into more granular skill-level matches within the high-scoring aggregated groups. This strategy provides the benefits of ESCO's richness without the unmanageable complexity of using the entire flat list for every operation.