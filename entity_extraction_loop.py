import csv
import pandas as pd
from inference.linker import EntityLinker

# TODO this doesn't currently move over all other job columns, like date posted etc.
# TODO: I am using  extracted_skills2 & extracted_requirements in other files now, so might as well drop extracted_skills 1, but I haven't done rigorous check of differences
# TODO Once I have a finalized version, need to make more efficient by only extracting from jobs I haven't already done so; either to recognize in this file, or earlier and only providing new jobs, and appending to database in clean form later
# TODO check with Apostolos, is it true that bert only checks first 100 words?

# NOTE: I adapted linker.py; I added a counter

# Initialize the entity linker
pipeline = EntityLinker(k=25, output_format='uuid') # for occupations, get 25
pipeline_skills = EntityLinker(k=50, output_format='uuid') # for skills, get 50

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
df['opportunity_description'] = df['opportunity_description'].fillna('')
df['opportunity_requirements'] = df['opportunity_requirements'].fillna('')


# Open CSV file for writing results
filepath_write = basedir + "data/pre_study/BERT_extracted_occupations_skills_uuid" + ".csv"
with open(filepath_write, mode="w", newline="", encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["opportunity_group_id", "opportunity_ref_id", "opportunity_title", "opportunity_description", "opportunity_requirements", "full_details", "extracted_occupation", "extracted_skills1", "extracted_skills2", "extracted_requirements"])  # Header

    # Process each row in the Jobs Database
    for snippet in df.itertuples(index=False):
        extracted_occ = pipeline(snippet.opportunity_title) # Extract occupation from job title
        extracted_skills1 = pipeline_skills(snippet.full_details) # Extract skills from full ad
        extracted_skills2 = pipeline_skills(snippet.opportunity_description) # Extract skills from just opportunity_description (check later which was better)
        extracted_requirements = pipeline_skills(snippet.opportunity_requirements) # Extract skills from opportunity_requirements
        writer.writerow([snippet.opportunity_group_id, snippet.opportunity_ref_id, snippet.opportunity_title, snippet.opportunity_description, snippet.opportunity_requirements, snippet.full_details, extracted_occ, extracted_skills1, extracted_skills2, extracted_requirements])


#### Shut the below doesn for now, I prefer to first get uuids and map them during cleaning, as the below is inconsistent and sometimes still gets uuids
# Do the same with labels mostly for exploration
#pipeline_skills = EntityLinker(k=25, output_format='preffered_label')

#filepath_write = basedir + "data/pre_study/BERT_extracted_occupations_skills_labels" + ".csv"
#with open(filepath_write, mode="w", newline="", encoding='utf-8') as file:
#    writer = csv.writer(file)
#    writer.writerow(["opportunity_group_id", "opportunity_ref_id", "opportunity_title", "opportunity_description", "opportunity_requirements", "full_details", "extracted_occupation", "extracted_skills1", "extracted_skills2", "extracted_requirements"])  # Header

    # Process each row in the Jobs Database
#    for snippet in df.itertuples(index=False):
#        extracted_occ = pipeline(snippet.opportunity_title) # Extract occupation from job title
#        extracted_skills1 = pipeline_skills(snippet.full_details) # Extract skills from full ad
#        extracted_skills2 = pipeline_skills(snippet.opportunity_description) # Extract skills from just opportunity_description (check later which was better)
#        extracted_requirements = pipeline_skills(snippet.opportunity_requirements) # Extract skills from opportunity_requirements
#        writer.writerow([snippet.opportunity_group_id, snippet.opportunity_ref_id, snippet.opportunity_title, snippet.opportunity_description, snippet.opportunity_requirements, snippet.full_details, extracted_occ, extracted_skills1, extracted_skills2, extracted_requirements])

print("Extraction completed! Results saved to C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study/ 🎉")