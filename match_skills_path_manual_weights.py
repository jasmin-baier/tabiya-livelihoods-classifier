import pandas as pd
import json
import networkx as nx
import os
from tqdm import tqdm # Import tqdm for a progress bar

# NOTE: change jobseeker and opportunity data paths at the bottom if needed
#    TAXONOMY_DATA_PATH = './taxonomy' 
#    MAIN_DATA_PATH = 'C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study'

def load_data(taxonomy_path='.', main_data_path='.'):
    """
    Loads all necessary files from their respective directories into pandas DataFrames.
    """
    print("Loading datasets...")
    try:
        # Load taxonomy files from the taxonomy_path
        skill_hierarchy = pd.read_csv(os.path.join(taxonomy_path, 'skill_hierarchy.csv'))
        skill_relations = pd.read_csv(os.path.join(taxonomy_path, 'skill_to_skill_relations.csv'))
        skills = pd.read_csv(os.path.join(taxonomy_path, 'skills.csv'))

        # Load main data files (jobseekers, opportunities) from the main_data_path
        with open(os.path.join(main_data_path, 'pilot_jobseeker_database.json'), 'r') as f:
            jobseekers = json.load(f)
        with open(os.path.join(main_data_path, 'pilot_opportunity_database.json'), 'r') as f:
            opportunities = json.load(f)

        print("Datasets loaded successfully.")
        return skills, skill_hierarchy, skill_relations, jobseekers, opportunities
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Make sure your file paths are correct.")
        return None, None, None, None, None

def create_uuid_to_id_map(skills_df):
    """
    Creates a dictionary to map historical UUIDs to the canonical skill ID.
    """
    print("Creating UUID to ID mapping...")
    uuid_map = {}
    skills_df = skills_df.dropna(subset=['ID', 'UUIDHISTORY'])
    
    for _, row in skills_df.iterrows():
        canonical_id = row['ID']
        uuids = row['UUIDHISTORY'].strip().split('\n')
        for uuid in uuids:
            if uuid:
                uuid_map[uuid.strip()] = canonical_id
                
    print(f"Map created with {len(uuid_map)} UUID entries.")
    return uuid_map

def build_weighted_skill_graph(skills_df, hierarchy_df, relations_df):
    """
    Builds the skill taxonomy graph with manually defined edge weights.
    - Hierarchical links have a weight of 1.0.
    - "Related" links have a weight of 1.5.
    """
    print("Building the weighted skill taxonomy graph...")
    G = nx.Graph()

    # Add all skills as nodes using the canonical 'ID'
    for _, skill in skills_df.iterrows():
        G.add_node(skill['ID'])

    # Add hierarchical relationships with weight 1.0
    for _, row in hierarchy_df.iterrows():
        if G.has_node(row['PARENTID']) and G.has_node(row['CHILDID']):
            G.add_edge(row['PARENTID'], row['CHILDID'], weight=1.0)

    # Add "related" skill relationships with weight 1.5
    for _, row in relations_df.iterrows():
        if G.has_node(row['REQUIRINGID']) and G.has_node(row['REQUIREDID']):
            G.add_edge(row['REQUIRINGID'], row['REQUIREDID'], weight=1.5)
            
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def get_jobseeker_skills_uuid(jobseeker_data):
    """Extracts a simple list of skill UUIDs from a jobseeker's record."""
    return [skill['uuid'] for skill in jobseeker_data.get('skills', []) if skill.get('uuid')]

def get_opportunity_skills_uuid(opportunity_data):
    """Extracts a simple list of skill UUIDs from an opportunity's record."""
    if not opportunity_data.get('skills'):
        return []
    return [skill['uuid'] for skill in opportunity_data.get('skills', []) if skill.get('uuid')]

def calculate_manual_match_score(jobseeker_skill_ids, opportunity_skill_ids, graph, max_dist=5.0):
    """
    Calculates a match score based on shortest path distance in the weighted graph.
    The score is an inverse of the distance, normalized by the max_dist.
    """
    if not opportunity_skill_ids:
        return 0.0

    total_score = 0.0
    
    valid_jobseeker_skills = [s_id for s_id in jobseeker_skill_ids if s_id in graph]
    valid_opportunity_skills = [s_id for s_id in opportunity_skill_ids if s_id in graph]

    if not valid_jobseeker_skills or not valid_opportunity_skills:
        return 0.0

    for req_skill_id in valid_opportunity_skills:
        min_dist = float('inf')
        for js_skill_id in valid_jobseeker_skills:
            if req_skill_id == js_skill_id:
                dist = 0.0
            elif nx.has_path(graph, source=js_skill_id, target=req_skill_id):
                dist = nx.shortest_path_length(graph, source=js_skill_id, target=req_skill_id, weight='weight')
            else:
                dist = float('inf')
            
            if dist < min_dist:
                min_dist = dist
        
        # Convert distance to a similarity score (0 to 1)
        score = max(0, 1 - (min_dist / max_dist))
        total_score += score
    
    avg_score = total_score / len(valid_opportunity_skills)
    return avg_score

def run_full_analysis_manual(all_jobseekers, all_opportunities, graph, threshold, uuid_map, output_path):
    """
    Calculates scores for all jobseeker-opportunity pairs using the manual weighted model and saves results.
    """
    print("Running full analysis for all jobseekers using manually weighted model...")
    detailed_scores = []
    aggregate_summaries = []

    opportunities_with_skills = [opp for opp in all_opportunities if get_opportunity_skills_uuid(opp)]
    total_opportunities_considered = len(opportunities_with_skills)

    if total_opportunities_considered == 0:
        print("No opportunities with skills found. Cannot generate analysis.")
        return

    for jobseeker in tqdm(all_jobseekers, desc="Processing Jobseekers"):
        jobseeker_id = jobseeker.get('compass_id')
        jobseeker_uuids = get_jobseeker_skills_uuid(jobseeker)
        jobseeker_skill_ids = [uuid_map[uuid] for uuid in jobseeker_uuids if uuid in uuid_map]

        if not jobseeker_skill_ids:
            aggregate_summaries.append({
                'jobseeker_id': jobseeker_id,
                'opportunities_matched': 0,
                'total_opportunities_considered': total_opportunities_considered,
                'aggregate_match_percent': 0.0
            })
            continue

        match_count = 0
        for opp in opportunities_with_skills:
            opp_id = opp.get('opportunity_ref_id')
            opp_uuids = get_opportunity_skills_uuid(opp)
            opp_skill_ids = [uuid_map[uuid] for uuid in opp_uuids if uuid in uuid_map]
            
            score = calculate_manual_match_score(jobseeker_skill_ids, opp_skill_ids, graph)
            
            detailed_scores.append({
                'jobseeker_id': jobseeker_id,
                'opportunity_id': opp_id,
                'match_score': score
            })
            
            if score >= threshold:
                match_count += 1
        
        percentage_match = (match_count / total_opportunities_considered) * 100
        aggregate_summaries.append({
            'jobseeker_id': jobseeker_id,
            'opportunities_matched': match_count,
            'total_opportunities_considered': total_opportunities_considered,
            'aggregate_match_percent': percentage_match
        })

    print("\nSaving results to CSV files...")
    scores_df = pd.DataFrame(detailed_scores)
    summary_df = pd.DataFrame(aggregate_summaries)

    scores_output_path = os.path.join(output_path, 'jobseeker_opportunity_scores_manual.csv')
    summary_output_path = os.path.join(output_path, 'jobseeker_aggregate_summary_manual.csv')

    scores_df.to_csv(scores_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print(f"Detailed scores saved to: {scores_output_path}")
    print(f"Aggregate summaries saved to: {summary_output_path}")

def main():
    """Main execution function."""
    # --- Define File Paths ---
    TAXONOMY_DATA_PATH = './taxonomy' 
    MAIN_DATA_PATH = 'C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study'

    skills_df, hierarchy_df, relations_df, jobseekers, opportunities = load_data(
        taxonomy_path=TAXONOMY_DATA_PATH,
        main_data_path=MAIN_DATA_PATH
    )
    
    if skills_df is None: return

    uuid_to_id_map = create_uuid_to_id_map(skills_df)
    skill_graph = build_weighted_skill_graph(skills_df, hierarchy_df, relations_df)
    
    if not jobseekers or not opportunities:
        print("No jobseekers or opportunities to analyze.")
        return
        
    # A score of 0.8 is a reasonable threshold for this distance-based approach.
    MATCH_THRESHOLD = 0.8

    run_full_analysis_manual(jobseekers, opportunities, skill_graph, MATCH_THRESHOLD, uuid_to_id_map, MAIN_DATA_PATH)
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()

