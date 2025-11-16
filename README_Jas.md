# PROJECT SETUP
# FIRST: Maker sure we are in the correct working directory
Set-Location "C:\Users\jasmi\Documents\GitHub\tabiya-livelihoods-classifier"

# STEP 1: Once opened VS Code, open Terminal (View > Terminal or CTRL + ~) and run:
.\classifier_setup.ps1
# That should activate virtual environment which I set up previously as per https://github.com/tabiya-tech/tabiya-livelihoods-classifier/blob/main/README.md#set-up-virtualenv

# IMPORT JOBS
# STEP 2: Get new jobs via Harambee API and update jobs database -- THIS IS CURRENTLY STILL DONE IN R, THIS SCRIPT IS NOT CHECKED
1_1_harambee_jobs_API_and_formatting.py

# STEP 3: View > Command Palette > Python: Select interpreter > CHOOSE VENV

# STEP 4: In Powershell in the virtual environment, run
2_1_entity_extraction_loop.py
# TODO must change to batching in system, i.e. so that it only re-analyzes new entries

# STEP 5: Once done, clean the output using the cleaning script
2_2_clean_bert_results.py

# LLM RERANKER
# STEP 6: Then: Give to LLMs -- Run only for occupations
3_1_LLM_pick_skills_full_details.py

# STEP 7: Reshape LLM occupation output and merge opportunity information back in
3_2a_clean_LLM_create_opp-db.py

# STEP 8: Then: Give bert_cleaned_with_occupation_final.json to LLMs -- Run for essential and optional skills (can do at the same time or after each other)
3_1_LLM_pick_skills_full_details.py

# STEP 9: For robustness, double check the opportunity database for any jobs that couldn't map the skills to uuids. 
3_3_debug_find_jobs_hallucinated_skills.py
# STEP 10: Then rerun LLM file with that bert_cleaned_rerun json
3_4_LLM_pick_skills_full_details_rerun.py

# OTHER
# STEP 11: Clean pre-existing jobseeker skillsets for jobseeker database
reshape_jobseeker_database.py

# OLD MATCHING FILES IN CASE I NEED TO LOOK BACK
# Simple match
match_skills_simple.py
# Match based on path structure, using manual weights for types of distances (direct match, same hierarchy, related)
match_skills_path_manual_weights.py
# Computationally driven path distances
# must install dependencies before running
pip install -r requirements.txt
match_skills_path_node_embedding.py
# Econ theory driven matching "surplus approach"
match_skills_path_node_surplus.py
# This final one combines the surplus approach for market allocation (aggregate info I need) with the recommender approach by Bied et al.
match_skills_path_node_surplus_and_biedetal.py