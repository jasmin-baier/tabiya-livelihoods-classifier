import pandas as pd
import os
import random
from scipy import stats


# TODO regarding the concern that there will be no matches: Since there is a skills-hierarchy and skill_groups, can I first find "more coarse" skills and then compare?
# TODO discuss skills entity structure that is being pulled from conversation
## What I will get is an array with uuids BUT
# TODO also ask which uuids I should best compare, since I can cleanly choose which appear in the jobs_vector -- the latest or the uuid history?
# [TODO Ask how difficult to set up RAG for LLM layer of skills extraction from job ads]
# TODO: Bring LLM output into correct format for this script to work
# TODO compare to random 100, not all people
# TODO: re naming convention: call whichever belief I use in the end just "prior_belief"
# TODO: maybe add in choose which jobs, i.e. only newest or only where active=True, EXCEPT have minimum of ~500-1000

# Define the base path for the data files.
# Using os.path.join ensures compatibility across different operating systems.
base_path = os.path.join("C:", os.sep, "Users", "jasmi", "OneDrive - Nexus365", 
                         "Documents", "PhD - Oxford BSG", "Paper writing projects", 
                         "Ongoing", "Compass", "data", "pre_study")

# Define the full paths for each of the CSV files.
file_path_skills_others = os.path.join(base_path, "dummy_data_skills_temporary.csv") # instead of csv, can I just take it from the current users of compass on harambee deployment; then always randomly choose 100; at ground 0 say "if I don't have enough users here, start picking from..." pilot one's as "baseline" as csv; note that this is a list of lists
#file_path_skills = os.path.join(base_path, "dummy_data_skills_others.csv")
file_path_baseline = os.path.join(base_path, "dummy_data_baseline_beliefs.csv")
file_path_jobs = os.path.join(base_path, "dummy_data_jobs.csv") 

df_skills_others = pd.read_csv(file_path_skills_others)
df_bl = pd.read_csv(file_path_baseline)
df_jobs = pd.read_csv(file_path_jobs)

# Define the column structure for the new DataFrame.
columns = [
    "harambee_id",
    "uuid_1",
    "uuid_2",
    "uuid_3",
    "uuid_history_11",
    "uuid_history_12",
    "uuid_history_13",
    "uuid_origin_1"
]

# Create a dictionary to hold the data for the single row.
# We'll initialize all values to None, as no data is being loaded yet.
# TODO here load skills entity and clean before creating the DataFrame
current_person_skills = {col: [None] for col in columns}

# Create a new pandas DataFrame from the dictionary.
# This DataFrame will have one row with the specified columns.
df_current_person_skills = pd.DataFrame(current_person_skills)

# --- STEP 1: Compute the current person's "truth" ---
# for each row in df_jobs, check how many of the columns with names that include "uuid" overlap with values in the columns with names that include "uuid" in df_current_person_skills, then divide by the total length of columns with names that include "uuid" in df_jobs

# Identify all columns in each DataFrame that contain "uuid" in their name.
uuid_cols_jobs = [col for col in df_jobs.columns if 'uuid' in col]
uuid_cols_skills = [col for col in df_current_person_skills.columns if 'uuid' in col] # THIS WILL ALREADY BE INPUT INTO FUNCTION

# Get the set of unique skill UUIDs for the current person.
# Using a set provides fast lookups. .values.flatten() gets all values from the uuid columns.
if not df_current_person_skills.empty:
    person_skill_uuids = set(df_current_person_skills[uuid_cols_skills].values.flatten())
else:
    person_skill_uuids = set() # TODO how can I robustly ensure that this is not empty?

# TODO: make sure that things don't happen outside of function; each function should have specific input, and should only access things that are part of its input
# TODO: function should not worry about structure of its input, but only what it's responsible for, so the input should already be in correct structure

# Define a function to calculate the overlap for a single row from df_jobs.
def calculate_overlap_score(row):
    """
    Checks how many UUIDs in a job row overlap with a person's skill UUIDs,
    and normalizes the count by the total number of UUID columns in the jobs data.
    """
    # Get the set of UUIDs from the current job row.
    job_uuids = set(row[uuid_cols_jobs].values)
    
    # Find the common UUIDs between the job and the person's skills.
    overlapping_uuids = job_uuids.intersection(person_skill_uuids)
    
    # Avoid division by zero if there are no UUID columns in df_jobs.
    if len(uuid_cols_jobs) == 0:
        return 0.0
        
    # Calculate the final score.
    score = len(overlapping_uuids) / len(uuid_cols_jobs)
    return score

# Apply the function to each row in df_jobs to compute the score.
# The result is stored in a new 'overlap_score' column.
# TODO problem here is that this is also dependent on the data structure, so should change that too
# TODO consider doing all of this without pandas; separate getting data into write structure into separate file, here ONLY have the calculation
# TODO vectorize instead of loop
df_jobs['overlap_score'] = df_jobs.apply(calculate_overlap_score, axis=1)

# Offline check while writing script: Display the head of the jobs DataFrame with the new score column to verify.
#print("\nOverlap calculation complete. Resulting df_jobs (head):")
#display_cols = [col for col in ['ReferenceNumber', 'job_title'] if col in df_jobs.columns] + ['overlap_score']
#print(df_jobs[display_cols].head())

# Find for how many jobs the overlap score is above 0.5
# TODO have a define threshold rather than hardcode the number
num_jobs_above_threshold = df_jobs[df_jobs['overlap_score'] > 0.5].shape[0]
# Get percentage of jobs with overlap score above 0.5
percentage_above_threshold = (num_jobs_above_threshold / df_jobs.shape[0]) * 100 # A number between 0 and 100

# Offline check while writing script:
#print(f"\nNumber and Percentage of jobs with overlap score above 0.5: {num_jobs_above_threshold} ({percentage_above_threshold:.2f}%)")

# --- STEP 2: Compare to other job seekers ---
# Check where in the distribution this value is: percentage_above_threshold compared to the vector of the column "truth" in df_skills_others
truth_distribution = df_skills_others['truth'].dropna()

# Calculate the percentile rank of our value within the "truth" distribution.
# Note: The 'truth' column values must be in the same scale (0-100) as percentage_above_threshold.
percentile = stats.percentileofscore(truth_distribution, percentage_above_threshold)

# --- STEP 3 & 4: Compare to baseline beliefs and assign group ---
if not df_current_person_skills.empty and 'harambee_id' in df_current_person_skills.columns:
    # Get the harambee_id from the current person's skill data
    current_harambee_id = df_current_person_skills['harambee_id'].iloc[0]
    # Find the corresponding row in the beliefs dataframe
    person_belief_row = df_bl[df_bl['harambee_id'] == current_harambee_id]

    # Get belief value and calculate the distance tot he truth
    if not person_belief_row.empty and 'belief_percentage_jobs' in person_belief_row.columns:
        belief_value = person_belief_row['belief_percentage_jobs'].iloc[0]
        difference = percentage_above_threshold - belief_value

        #print(f"Belief percentage for jobs from baseline data: {belief_value:.2f}%")
        #print(f"Difference (Truth % - Belief %): {difference:.2f}")
        
        # Above or below (negative or positive)
        underconfident = difference > 0
        
        # Absolute distance
        # Check if the absolute value of the difference is larger than 20 # TODO might change this threshold after pilot testing
        high_difference = abs(difference) > 20

        #print(f"Is underconfident: {underconfident}")
        #print(f"Is high difference: {high_difference}")

        # COMPUTE GROUP
        # First, there is a 50% chance that they are assigned to either group 1 or 2. Then once that is set they are seperated again in the following way:
            # Group 1a: if high_difference = True
            # Group 1b: if high_difference = False
            # Group 2a: if underconfident = True
            # Group 2b: if underconfident = False
        assigned_group = ""

        if random.random() < 0.5:
            # Path for Group 1: Assignment based on 'high_difference'
            if high_difference:
                assigned_group = "Group 1a"
            else:
                assigned_group = "Group 1b"
        else:
            # Path for Group 2: Assignment based on 'underconfident'
            if underconfident:
                assigned_group = "Group 2a"
            else:
                assigned_group = "Group 2b"
        
        print(f"Final Assigned Group: {assigned_group}")

    else:
        print(f"\nWarning: Could not find matching belief data for harambee_id '{current_harambee_id}' or 'belief_percentage_jobs' column is missing.")
else:
    print("\nWarning: 'harambee_id' not found in skills data, cannot calculate belief difference.")