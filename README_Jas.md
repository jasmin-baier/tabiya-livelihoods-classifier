# PROJECT SETUP
# FIRST: Maker sure we are in the correct working directory
Set-Location "C:\Users\jasmi\Documents\GitHub\tabiya-livelihoods-classifier"

# STEP 1: Once opened VS Code, open Terminal (View > Terminal or CTRL + ~) and run:
.\classifier_setup.ps1
# That should activate virtual environment which I set up previously as per https://github.com/tabiya-tech/tabiya-livelihoods-classifier/blob/main/README.md#set-up-virtualenv

# IMPORT JOBS
# STEP 2: Get new jobs via Harambee API and update jobs database
1_1_harambee_jobs_API_and_formatting.py

# STEP 3: View > Command Palette > Python: Select interpreter > CHOOSE VENV

# STEP 4: In Powershell in the virtual environment, run
2_1_entity_extraction_loop.py
# TODO must change to batching in system, i.e. so that it only re-analyzes new entries

# STEP 5: Once done, clean the output using the cleaning script
2_2_clean_bert_results.py

# LLM RERANKER
# STEP 6: Then: Give to LLMs
3_1_LLM_pick_skills_full_details.py

# STEP 7: Reshape LLM output and merge opportunity information back in
3_2_clean_LLM_create_opp-db.py

# STEP 7.5: For robustness, double check the pilot database for any jobs that couldn't map the skills to uuids. 
3_3_debug_find_jobs_hallucinated_skills.py
# STEP 7.6: Then rerun LLM file with that bert_cleaned_rerun json
3_4_LLM_pick_skills_full_details_rerun.py

# A script that translates final databases to skill_groups and occupation_groups to try matching then -- but note that here the LLM PICKING happened still at the lower level
3_5_map_DBs_to_groups.py

# Alternatively, this script already PICKS skill_groups, which can then be directly compared to the skill_groups in the jobseeker database
LLM_pick_skills_GROUPS_full_details

# OTHER
# STEP 8: Clean pre-existing jobseeker skillsets for jobseeker database
reshape_jobseeker_database.py

# Finally, run computation (will actually be one on cloud, so this file is just for testing)
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