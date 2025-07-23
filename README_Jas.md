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

# Once done, clean the output using the cleaning script
clean_bert_results.py

# Then: Give to LLMs (script to be written)
LLM_pick_skills.py

# Finally, run computation
match_skills_compute.py