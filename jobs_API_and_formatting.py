#!/usr/bin/env python3
"""
Skills Job Matching Exploration
Author: Jasmin Baier
Date: 2025-04-30
Translated from R to Python
"""

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
            'GroupSourceID': safe_extract(item, 'GroupSourceID'),
            'ReferenceNumber': ref_num,
            'full_details': '; '.join(details_parts),
            'job_title': safe_extract(item, 'JobTitle'),
            'job_description': safe_extract(item, 'RoleDescription'),
            'job_requirements': safe_extract(item, 'RoleRequirements'),
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
text_columns = ['full_details', 'job_description', 'job_requirements']
for col in text_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(clean_html_text)

# Fill NaN values in text columns with empty strings
df_clean['job_description'] = df_clean['job_description'].fillna('')
df_clean['job_requirements'] = df_clean['job_requirements'].fillna('')

# Save clean dataset
clean_file = os.path.join(basedir, "data/pre_study/harambee_jobs_clean.csv")
df_clean.to_csv(clean_file, index=False)
print(f"\nClean dataset saved to: {clean_file}")

# Reload clean dataset
df_clean = pd.read_csv(clean_file)

# ============================================================================
# BERT Analysis Section - Data Exploration
# ============================================================================

def load_bert_data():
    """
    Load BERT extracted data
    """
    try:
        bert_uuid = pd.read_csv(os.path.join(basedir, "data/pre_study/2025-05-28_BERT_extracted_occupations_skills_uuid.csv"))
        bert_labels = pd.read_csv(os.path.join(basedir, "data/pre_study/2025-05-28_BERT_extracted_occupations_skills_labels.csv"))
        return bert_uuid, bert_labels
    except FileNotFoundError as e:
        print(f"BERT data files not found: {e}")
        return None, None

# Load BERT data
bert_uuid, bert_labels = load_bert_data()

if bert_uuid is not None and bert_labels is not None:
    print("\nBERT data loaded successfully")
    
    # Compare extractions
    print("\nComparing BERT extractions:")
    
    # Check if occupation extractions are identical
    occ_identical = bert_uuid['extracted_occupation'].equals(bert_labels['extracted_occupation'])
    print(f"Occupation extractions identical: {occ_identical}")
    
    # Check if skills extractions are identical
    skills1_identical = bert_uuid['extracted_skills1'].equals(bert_labels['extracted_skills1'])
    skills2_identical = bert_uuid['extracted_skills2'].equals(bert_labels['extracted_skills2'])
    print(f"Skills1 extractions identical: {skills1_identical}")
    print(f"Skills2 extractions identical: {skills2_identical}")
    
    # Fuzzy matching function
    def fuzzy_match(a, b, threshold=0.1):
        """Calculate cosine similarity between strings"""
        if pd.isna(a) or pd.isna(b):
            return False
        try:
            # Simple token-based comparison
            tokens_a = set(str(a).lower().split())
            tokens_b = set(str(b).lower().split())
            if not tokens_a or not tokens_b:
                return a == b
            intersection = tokens_a.intersection(tokens_b)
            union = tokens_a.union(tokens_b)
            similarity = len(intersection) / len(union) if union else 0
            return similarity > (1 - threshold)
        except:
            return str(a) == str(b)
    
    # Compare occupation and skills1
    bert_uuid['identical_flag_occskill1'] = (bert_uuid['extracted_occupation'] == bert_uuid['extracted_skills1'])
    bert_labels['identical_flag_occskill1'] = (bert_labels['extracted_occupation'] == bert_labels['extracted_skills1'])
    
    # Fuzzy comparison
    bert_uuid['fuzzy_flag_occskill1'] = [fuzzy_match(a, b) for a, b in 
                                        zip(bert_uuid['extracted_occupation'], bert_uuid['extracted_skills1'])]
    bert_labels['fuzzy_flag_occskill1'] = [fuzzy_match(a, b) for a, b in 
                                          zip(bert_labels['extracted_occupation'], bert_labels['extracted_skills1'])]
    
    # Compare skills1 and skills2
    bert_uuid['identical_flag_skills'] = (bert_uuid['extracted_skills1'] == bert_uuid['extracted_skills2'])
    bert_labels['identical_flag_skills'] = (bert_labels['extracted_skills1'] == bert_labels['extracted_skills2'])
    
    bert_uuid['fuzzy_flag_skills'] = [fuzzy_match(a, b) for a, b in 
                                     zip(bert_uuid['extracted_skills1'], bert_uuid['extracted_skills2'])]
    bert_labels['fuzzy_flag_skills'] = [fuzzy_match(a, b) for a, b in 
                                       zip(bert_labels['extracted_skills1'], bert_labels['extracted_skills2'])]
    
    # Tabulate results
    print("\nBERT UUID comparison results:")
    comparison_cols = ['identical_flag_occskill1', 'fuzzy_flag_occskill1', 'identical_flag_skills', 'fuzzy_flag_skills']
    for col in comparison_cols:
        if col in bert_uuid.columns:
            print(f"{col}: {bert_uuid[col].value_counts().to_dict()}")
    
    print("\nBERT Labels comparison results:")
    for col in comparison_cols:
        if col in bert_labels.columns:
            print(f"{col}: {bert_labels[col].value_counts().to_dict()}")

# ============================================================================
# Skills File Cleaning (Ajira)
# ============================================================================

def clean_skills_data():
    """
    Clean Ajira skills data
    """
    try:
        with open(os.path.join(basedir, "data/pre_study/discovered_skills_ajira.json"), 'r') as f:
            skills = json.load(f)
        
        # Flatten the nested structure
        skills_clean_list = []
        
        for i, item in enumerate(skills):
            flat_item = {'original_index': i}
            
            def flatten_dict(d, parent_key='', sep='_'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    elif isinstance(v, list):
                        for idx, list_item in enumerate(v):
                            list_key = f"{new_key}_{idx+1}"
                            items.append((list_key, list_item))
                    else:
                        items.append((new_key, v))
                return dict(items)
            
            flat_item.update(flatten_dict(item))
            skills_clean_list.append(flat_item)
        
        skills_clean = pd.DataFrame(skills_clean_list)
        
        # Remove unwanted columns
        columns_to_remove = [
            'experience_id', 'preferred_label', 'description', 'skill_type', 
            'model_id', 'conversation_phase'
        ]
        
        # Remove columns that start with 'alt_labels'
        alt_label_cols = [col for col in skills_clean.columns if col.startswith('alt_labels')]
        columns_to_remove.extend(alt_label_cols)
        
        # Remove existing columns from the removal list
        existing_cols_to_remove = [col for col in columns_to_remove if col in skills_clean.columns]
        skills_clean = skills_clean.drop(columns=existing_cols_to_remove)
        
        # Remove columns that are entirely NA
        skills_clean = skills_clean.dropna(axis=1, how='all')
        
        # Rename uuid_history columns if they exist
        uuid_columns = {
            'uuid_history.uuid_history_1': 'uuid_history_1',
            'uuid_history.uuid_history_2': 'uuid_history_2',
            'uuid_history.uuid_history_3': 'uuid_history_3'
        }
        skills_clean = skills_clean.rename(columns=uuid_columns)
        
        # Pivot to wide format by conversation_id if that column exists
        if 'conversation_id' in skills_clean.columns:
            # Add row numbers within each conversation
            skills_clean = skills_clean.sort_values('conversation_id').reset_index(drop=True)
            skills_clean['row_num'] = skills_clean.groupby('conversation_id').cumcount() + 1
            
            # Pivot to wide format
            id_cols = ['conversation_id']
            value_cols = [col for col in skills_clean.columns if col not in id_cols + ['row_num']]
            
            skills_wide = skills_clean.pivot_table(
                index='conversation_id',
                columns='row_num',
                values=value_cols,
                aggfunc='first'
            ).reset_index()
            
            # Flatten column names
            skills_wide.columns = [f"{col[0]}_{col[1]}" if col[1] != '' and col[0] != 'conversation_id' else col[0] 
                                  for col in skills_wide.columns]
            
            skills_clean = skills_wide
        
        # Save cleaned skills data
        skills_file = os.path.join(basedir, "data/pre_study/dummy_data_skills_temporary.csv")
        skills_clean.to_csv(skills_file, index=False)
        print(f"Skills data saved to: {skills_file}")
        
        return skills_clean
        
    except FileNotFoundError:
        print("Skills file not found")
        return None
    except Exception as e:
        print(f"Error processing skills data: {e}")
        return None

# Clean skills data
skills_data = clean_skills_data()

print("\n" + "="*80)
print("TRANSLATION COMPLETE")
print("="*80)
print(f"Clean jobs dataset shape: {df_clean.shape}")
if skills_data is not None:
    print(f"Clean skills dataset shape: {skills_data.shape}")
print("\nAll data processing steps have been translated from R to Python")
print("Files saved in the same directory structure as the original R code")
