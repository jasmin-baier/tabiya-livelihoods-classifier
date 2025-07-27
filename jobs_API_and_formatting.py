"""
Skills Job Matching Exploration
Author: Jasmin Baier
Date: 2025-04-30
"""

# TODO: This was translated from R, need to quality check and ensure it works as intended.
# TODO: Double double check that I am not adding duplicates -- if a job with an existing group+reference ID was updated only use the new one (wouldn't appear as duplicate by checking the whole row since maybe dates were extended). In raw harambee jobs file check how unique the IDs are, e.g. can there be a job with the same ID in two locations?

import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import re
from bs4 import BeautifulSoup
import warnings
from scipy.spatial.distance import cosine
from itertools import chain

# Clear warnings and set options
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set seed for reproducibility
np.random.seed(167)

# Base directory
basedir = "C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/"

# ============================================================================
# Call Jobs from Harambee API
# ============================================================================

def get_harambee_jobs():
    """
    Fetch jobs from Harambee API
    API documentation: https://developers.sayouth.org.za/Home/ApiDocs?version=1.0
    """
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://api.sayouth.org.za/Opportunity/All?date_from=2022-01-01&date_to={today}&num_records=10000"
    
    headers = {
        'accept': 'application/json',
        'X-API-VERSION': '1.0',
        'X-API-KEY': 'P_hdIxyAWTVI8xrVuFDxXllNoecr0Sj78wID'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        print("Successfully fetched data from API")
        
        # Save the data
        filename = f"harambee_jobs_2024-01-01_to_{today}.json"
        with open(os.path.join(basedir, "data/pre_study", filename), 'w') as f:
            json.dump(data, f)
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

# Fetch new data
data = get_harambee_jobs()

# ============================================================================
# Merge new export with master database
# ============================================================================

def merge_with_master_database(new_data):
    """
    Merge new export with existing master database
    """
    master_file = os.path.join(basedir, "data/pre_study/harambee_jobs_master_database.json")
    
    try:
        # Load existing master database
        with open(master_file, 'r') as f:
            existing_data = json.load(f)
        print("Loaded existing master database")
    except FileNotFoundError:
        print("Master database not found, creating new one")
        existing_data = []
    
    # Merge and remove duplicates
    # Convert to strings for comparison to handle duplicates
    existing_str = [json.dumps(item, sort_keys=True) for item in existing_data]
    new_str = [json.dumps(item, sort_keys=True) for item in new_data]
    
    # Combine and get unique items
    all_items_str = list(set(existing_str + new_str))
    merged_data = [json.loads(item) for item in all_items_str]
    
    # Save merged database
    with open(master_file, 'w') as f:
        json.dump(merged_data, f)
    
    print(f"Merged database contains {len(merged_data)} unique records")
    return merged_data

# Merge with master database
df = merge_with_master_database(data)

# ============================================================================
# Clean list and write clean database
# ============================================================================

def safe_extract(item, key):
    """Safely extract field from item, handling None values"""
    value = item.get(key)
    if value is None:
        return None
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value

def process_job_list(job_list):
    """
    Function to safely extract fields from each job item
    """
    processed_jobs = []
    
    for item in job_list:
        # Extract reference number (remove non-digits)
        ref_num = safe_extract(item, 'ReferenceNumber')
        if ref_num:
            ref_num = re.sub(r'\D', '', str(ref_num))
            ref_num = int(ref_num) if ref_num else None
        
        # Create full_details string
        details_parts = [
            f"CompanyName: {safe_extract(item, 'CompanyName') or 'NA'}",
            f"JobTitle: {safe_extract(item, 'JobTitle') or 'NA'}",
            f"RoleDescription: {safe_extract(item, 'RoleDescription') or 'NA'}",
            f"ContractType: {safe_extract(item, 'ContractType') or 'NA'}",
            f"DatePosted: {safe_extract(item, 'DatePosted') or 'NA'}",
            f"OpportunityClosingDate: {safe_extract(item, 'OpportunityClosingDate') or 'NA'}",
            f"City: {safe_extract(item, 'City') or 'NA'}",
            f"Province: {safe_extract(item, 'Province') or 'NA'}",
            f"OpportunityDuration: {safe_extract(item, 'OpportunityDuration') or 'NA'}",
            f"IsOnline: {safe_extract(item, 'IsOnline') or 'NA'}",
            f"RoleRequirements: {safe_extract(item, 'RoleRequirements') or 'NA'}",
            f"CertificationType: {safe_extract(item, 'CertificationType') or 'NA'}"
        ]
        
        # Extract salary (remove non-numeric characters)
        salary = safe_extract(item, 'Salary')
        if salary:
            salary = re.sub(r'[^0-9.]', '', str(salary))
            salary = float(salary) if salary else None
        
        job_data = {
            'opportunity_group_id': safe_extract(item, 'GroupSourceID'),
            'opportunity_ref_id': ref_num,
            'full_details': '; '.join(details_parts),
            'opportunity_title': safe_extract(item, 'JobTitle'),
            'opportunity_description': safe_extract(item, 'RoleDescription'),
            'opportunity_requirements': safe_extract(item, 'RoleRequirements'),
            'company_name': safe_extract(item, 'CompanyName'),
            'contract_type': safe_extract(item, 'ContractType'),
            'date_posted': safe_extract(item, 'DatePosted'),
            'certification_type': safe_extract(item, 'CertificationType'),
            'certification_description': safe_extract(item, 'CertificationDescription'),
            'date_closing': safe_extract(item, 'OpportunityClosingDate'),
            'city': safe_extract(item, 'City'),
            'province': safe_extract(item, 'Province'),
            'latitude': safe_extract(item, 'Latitude'),
            'longitude': safe_extract(item, 'Longitude'),
            'salary_type': safe_extract(item, 'SalaryType'),
            'salary': salary,
            'opportunity_url': safe_extract(item, 'OpportunityUrl'),
            'opportunity_duration': safe_extract(item, 'OpportunityDuration'),
            'is_online': safe_extract(item, 'IsOnline')
        }
        
        processed_jobs.append(job_data)
    
    return pd.DataFrame(processed_jobs)

# Process the job list
df_clean = process_job_list(df)

# Remove duplicates
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

# Drop location and certification_description columns (mostly NA)
columns_to_drop = ['certification_description']
df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])

# Convert date columns
df_clean['date_posted'] = pd.to_datetime(df_clean['date_posted'], errors='coerce')
df_clean['date_closing'] = pd.to_datetime(df_clean['date_closing'], errors='coerce')

# Data exploration
print("\nData exploration:")
print("Missing values per column:")
print(df_clean.isnull().sum())

print(f"\nNumber of unique companies: {df_clean['company_name'].nunique()}")
print(f"Number of unique cities: {df_clean['city'].nunique()}")

print("\nUnique values in categorical columns:")
for col in ['contract_type', 'province', 'salary_type', 'opportunity_duration', 'is_online']:
    if col in df_clean.columns:
        print(f"{col}: {df_clean[col].value_counts().to_dict()}")

print(f"\nDate posted range: {df_clean['date_posted'].min()} to {df_clean['date_posted'].max()}")
print(f"Date closing range: {df_clean['date_closing'].min()} to {df_clean['date_closing'].max()}")
print(f"Salary statistics: {df_clean['salary'].describe()}")

# ============================================================================
# Clean HTML text
# ============================================================================

def clean_html_text(text):
    """
    Function to clean HTML and extra formatting
    """
    if pd.isna(text) or text is None:
        return ""
    
    try:
        # Parse HTML
        soup = BeautifulSoup(f"<html>{text}</html>", 'html.parser')
        clean_text = soup.get_text()
        
        # Clean up text
        clean_text = re.sub(r'[\r\n]+', ' ', clean_text)  # Remove newlines
        clean_text = re.sub(r'\s+', ' ', clean_text)      # Normalize spaces
        clean_text = re.sub(r'&[a-zA-Z]+;', ' ', clean_text)  # Remove HTML entities
        clean_text = re.sub(r'[^\w\s]$', '', clean_text)  # Trim trailing punctuation
        clean_text = clean_text.strip()  # Trim spaces
        
        return clean_text
    except:
        return str(text) if text is not None else ""

# Apply HTML cleaning to text columns
text_columns = ['full_details', 'opportunity_description', 'opportunity_requirements']
for col in text_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(clean_html_text)

# Fill NaN values in text columns with empty strings
df_clean['opportunity_description'] = df_clean['opportunity_description'].fillna('')
df_clean['opportunity_requirements'] = df_clean['opportunity_requirements'].fillna('')

# Remove duplicates
df_clean = df_clean.drop_duplicates()
## Create a score based on variables I care about, keep only later/longer entries
df_clean["composite_score"] = (
    pd.to_datetime(df_clean["date_closing"]).astype("int64") // 1_000_000 * 1e3 +  # prioritize closing date
    pd.to_datetime(df_clean["date_posted"]).astype("int64") // 1_000_000 * 1e3 +  # then posted date
    df_clean["opportunity_description"].str.len().fillna(0) +                    # longer descriptions
    df_clean["full_details"].str.len().fillna(0)                                 # longer full details
)
df_clean = (
    df_clean.sort_values("composite_score", ascending=False)
            .drop_duplicates(subset=["opportunity_group_id", "opportunity_ref_id", "opportunity_title"])
            .reset_index(drop=True)
)

# Save clean dataset
clean_file = os.path.join(basedir, "data/pre_study/harambee_jobs_clean.csv")
df_clean.to_csv(clean_file, index=False)
print(f"\nClean dataset saved to: {clean_file}")

# Reload clean dataset
df_clean = pd.read_csv(clean_file)

