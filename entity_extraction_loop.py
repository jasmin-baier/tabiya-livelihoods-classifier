import csv
import pandas as pd
from inference.linker import EntityLinker

# TODO this doesn't currently move over all other job columns, like date posted etc.
# TODO: I am using extracted_skills1 in other files now. Better might be to merge extracted_skills2 & extracted_requirements? Though first quality check what is classified in requirements
# TODO Once I have a finalized version, need to make more efficient by only extracting from jobs I haven't already done so; either to recognize in this file, or earlier and only providing new jobs, and appending to database in clean form later

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


#### Shut the below doesn for now, I prefer to first get uuids and map them during cleaning, as the below is inconsistent and sometimes still gets uuids
# Do the same with labels mostly for exploration
#pipeline_skills = EntityLinker(k=25, output_format='preffered_label')

#filepath_write = basedir + "data/pre_study/BERT_extracted_occupations_skills_labels" + ".csv"
#with open(filepath_write, mode="w", newline="", encoding='utf-8') as file:
#    writer = csv.writer(file)
#    writer.writerow(["GroupSourceID", "ReferenceNumber", "job_title", "job_description", "job_requirements", "full_details", "extracted_occupation", "extracted_skills1", "extracted_skills2", "extracted_requirements"])  # Header

    # Process each row in the Jobs Database
#    for snippet in df.itertuples(index=False):
#        extracted_occ = pipeline(snippet.job_title) # Extract occupation from job title
#        extracted_skills1 = pipeline_skills(snippet.full_details) # Extract skills from full ad
#        extracted_skills2 = pipeline_skills(snippet.job_description) # Extract skills from just job_description (check later which was better)
#        extracted_requirements = pipeline_skills(snippet.job_requirements) # Extract skills from job_requirements
#        writer.writerow([snippet.GroupSourceID, snippet.ReferenceNumber, snippet.job_title, snippet.job_description, snippet.job_requirements, snippet.full_details, extracted_occ, extracted_skills1, extracted_skills2, extracted_requirements])

print("Extraction completed! Results saved to C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study/ 🎉")