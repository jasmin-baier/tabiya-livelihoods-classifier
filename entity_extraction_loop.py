import csv
import pandas as pd
from inference.linker import EntityLinker

# TODO: Currently returns a LOT of skills (around 500 per job using full job description) -- probably overkill
# TODO: I am using extracted_skills1 in other files now. Better might be to merge extracted_skills2 & extracted_requirements

# Initialize the entity linker
pipeline = EntityLinker(k=20, output_format='uuid') # for occupations, get 20
pipeline_skills = EntityLinker(k=100, output_format='uuid') # for skills, get 100

# Import Clean Harambee Jobs csv
basedir = "C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/"
filepath = basedir + "data/pre_study/harambee_jobs_clean" + ".csv"
df = pd.read_csv(filepath)
#print(df.columns)
print("Number of rows in the dataset before extraction:")
print(df.shape[0])
#df = df.head(5) # remove again but for testing purposes only

# Make sure relevant variables don't have NAs, as linker.py throws an error if they do
df['full_details'] = df['full_details'].fillna('')
df['job_description'] = df['job_description'].fillna('')
df['job_requirements'] = df['job_requirements'].fillna('')


# Open CSV file for writing results
filepath_write = basedir + "data/pre_study/BERT_extracted_occupations_skills_uuid" + ".csv"
with open(filepath_write, mode="w", newline="", encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["GroupSourceID", "ReferenceNumber", "job_title", "job_description", "job_requirements", "full_details", "extracted_occupation", "extracted_skills1", "extracted_skills2", "extracted_requirements"])  # Header

    # Process each row in the Jobs Database
    for snippet in df.itertuples(index=False):
        extracted_occ = pipeline(snippet.job_title) # Extract occupation from job title
        extracted_skills1 = pipeline_skills(snippet.full_details) # Extract skills from full ad
        extracted_skills2 = pipeline_skills(snippet.job_description) # Extract skills from just job_description (check later which was better)
        extracted_requirements = pipeline_skills(snippet.job_requirements) # Extract skills from job_requirements
        writer.writerow([snippet.GroupSourceID, snippet.ReferenceNumber, snippet.job_title, snippet.job_description, snippet.job_requirements, snippet.full_details, extracted_occ, extracted_skills1, extracted_skills2, extracted_requirements])

# Do the same with labels mostly for exploration
pipeline_skills = EntityLinker(k=100, output_format='preffered_label')

filepath_write = basedir + "data/pre_study/BERT_extracted_occupations_skills_labels" + ".csv"
with open(filepath_write, mode="w", newline="", encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["GroupSourceID", "ReferenceNumber", "job_title", "job_description", "job_requirements", "full_details", "extracted_occupation", "extracted_skills1", "extracted_skills2", "extracted_requirements"])  # Header

    # Process each row in the Jobs Database
    for snippet in df.itertuples(index=False):
        extracted_occ = pipeline(snippet.job_title) # Extract occupation from job title
        extracted_skills1 = pipeline_skills(snippet.full_details) # Extract skills from full ad
        extracted_skills2 = pipeline_skills(snippet.job_description) # Extract skills from just job_description (check later which was better)
        extracted_requirements = pipeline_skills(snippet.job_requirements) # Extract skills from job_requirements
        writer.writerow([snippet.GroupSourceID, snippet.ReferenceNumber, snippet.job_title, snippet.job_description, snippet.job_requirements, snippet.full_details, extracted_occ, extracted_skills1, extracted_skills2, extracted_requirements])

# TODO later in cleaning just remove occupation from the last two extractions.
# TODO later add evaluation bit

print("Extraction completed! Results saved to C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study/ 🎉")