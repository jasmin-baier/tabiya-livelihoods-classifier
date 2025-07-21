# Project set up
# FIRST: Maker sure we are in the correct working directory
Set-Location "C:\Users\jasmi\Documents\GitHub\tabiya-livelihoods-classifier"
# STEP 1: Once opened VS Code, open Terminal (View > Terminal or CTRL + ~) and run:
.\classifier_setup.ps1
# That should activate virtual environment which I set up previously as per https://github.com/tabiya-tech/tabiya-livelihoods-classifier/blob/main/README.md#set-up-virtualenv

# STEP 2: View > Command Palette > Python: Select interpreter > CHOOSE VENV

# STEP 3: In Powershell in the virtual environment, run
# python entity_extraction.py
# OR
python entity_extraction_loop.py

# Once done, clean the output using the cleaning script
clean_bert_results.py

# Then: Give to LLMs (script to be written)

# Finally, run computation
match_skills_compute.py