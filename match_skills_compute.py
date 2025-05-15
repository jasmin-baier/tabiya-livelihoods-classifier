import os
import json

# Set base directory
basedir = "C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass"

# Imagine I had skills from jobs
with open(os.path.normpath(os.path.join(basedir, "data", "pre_study", "harambee_jobs_2024-01-01_to_2025-04-30.json")), "r", encoding="utf-8") as file:
    jobs = json.load(file)

# skills_jobs # TODO missing

# load taxonomy


# define / extract "skills from current conversation"
# TODO missing

# load "other people's skills"
with open(os.path.normpath(os.path.join(basedir, "data", "pre_study", "discovered_skills_ajira.json")), "r", encoding="utf-8") as file:
    skills_compass = json.load(file)

# ===== EXPLORE DATA =====

# Print number of observations
len(jobs)
len(skills_compass)
print(type(skills_compass))
print(skills_compass[0])  # Print first entry to inspect structure
print(json.dumps(skills_compass[0], indent=4))  # Nicely formatted output
