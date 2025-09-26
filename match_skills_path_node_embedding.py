import pandas as pd
import json
import networkx as nx
import numpy as np
import os
import argparse # Import argparse for command-line arguments
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# NOTE: change jobseeker and opportunity data paths at the bottom if needed
#    TAXONOMY_DATA_PATH = './taxonomy' 
#    MAIN_DATA_PATH = 'C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study'
# remove _extract , and change opp db to pilot_opportunity_database_unique.json

# --- NEW IMPORTS for PyTorch Geometric ---
# Make sure you have PyTorch and PyG installed
try:
    import torch
    from torch_geometric.nn import Node2Vec
except ImportError:
    print("\nPyTorch or PyTorch Geometric not found.")
    print("Please install them. For most systems (without a dedicated NVIDIA GPU):")
    print("pip install torch")
    print("pip install torch_geometric")
    print("\nFor systems with a GPU, please see the PyTorch website for specific installation instructions.")
    exit()

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

        with open(os.path.join(main_data_path, 'pilot_jobseeker_database.json'), 'r', encoding='utf-8') as f:
            jobseekers = json.load(f)
        with open(os.path.join(main_data_path, 'pilot_opportunity_database_unique.json'), 'r', encoding='utf-8') as f:
            opportunities = json.load(f)

        print("Datasets loaded successfully.")
        return skills, skill_hierarchy, skill_relations, jobseekers, opportunities
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Make sure your file paths are correct.")
        return None, None, None, None, None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}. One of your JSON files might be corrupted.")
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

def build_unweighted_skill_graph(skills_df, hierarchy_df, relations_df):
    """
    Builds an unweighted graph and creates mappings for PyG.
    """
    print("Building the unweighted skill taxonomy graph...")
    G = nx.Graph()

    node_list = skills_df['ID'].unique().tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    
    for node_id in node_list:
        G.add_node(node_to_idx[node_id])

    for _, row in hierarchy_df.iterrows():
        parent = node_to_idx.get(row['PARENTID'])
        child = node_to_idx.get(row['CHILDID'])
        if parent is not None and child is not None:
            G.add_edge(parent, child)

    for _, row in relations_df.iterrows():
        requiring = node_to_idx.get(row['REQUIRINGID'])
        required = node_to_idx.get(row['REQUIREDID'])
        if requiring is not None and required is not None:
            G.add_edge(requiring, required)
            
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, node_to_idx, node_list

def train_and_save_model(graph, node_to_idx, model_path, map_path):
    """
    Trains a Node2Vec model and saves it along with the node mapping.
    """
    print("Training skill embedding model with PyTorch Geometric...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    edge_index = torch.tensor(list(graph.edges)).t().contiguous()

    model = Node2Vec(
        edge_index, embedding_dim=64, walk_length=30, context_size=10,
        walks_per_node=20, num_negative_samples=1, p=1.0, q=1.0, sparse=True,
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    model.train()
    for epoch in range(1, 51):
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader, desc=f"Epoch {epoch}/50"):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss / len(loader):.4f}")

    print("Model training complete.")
    
    # --- SAVE THE MODEL AND MAPPING ---
    print("Saving model and mappings...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    with open(map_path, 'w') as f:
        json.dump(node_to_idx, f)
    print(f"Model saved to {model_path}")
    print(f"Node map saved to {map_path}")
    
    return model

def load_embedding_model(graph, model_path, map_path):
    """
    Loads a pre-trained Node2Vec model and its node mapping.
    """
    print("Loading pre-trained model and mappings...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open(map_path, 'r') as f:
        node_to_idx = json.load(f)
        
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    
    # Re-initialize model with the same architecture
    model = Node2Vec(
        edge_index, embedding_dim=64, walk_length=30, context_size=10,
        walks_per_node=20, num_negative_samples=1, p=1.0, q=1.0, sparse=True,
    ).to(device)
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set model to evaluation mode
    
    print("Model and mappings loaded successfully.")
    return model, node_to_idx

def get_jobseeker_skills_uuid(jobseeker_data):
    """Extracts a simple list of skill UUIDs from a jobseeker's record."""
    return [skill['uuid'] for skill in jobseeker_data.get('skills', []) if skill.get('uuid')]

def get_opportunity_skills_uuid(opportunity_data):
    """Extracts a simple list of skill UUIDs from an opportunity's record."""
    if not opportunity_data.get('skills'):
        return []
    return [skill['uuid'] for skill in opportunity_data.get('skills', []) if skill.get('uuid')]

def pre_process_entities_for_embedding(entities, id_getter_func, uuid_map, model, node_to_idx):
    """
    Pre-processes entities to fetch their skill vectors from the loaded/trained model.
    """
    print(f"Pre-processing entities to fetch skill vectors...")
    entity_vectors = {}
    with torch.no_grad():
        all_embeddings = model.embedding.weight.cpu().numpy()

    for entity in tqdm(entities, desc="Fetching skill vectors"):
        entity_id = entity.get(id_getter_func)
        uuids = (get_jobseeker_skills_uuid(entity) if id_getter_func == 'compass_id' 
                 else get_opportunity_skills_uuid(entity))
        
        skill_ids = [uuid_map[uuid] for uuid in uuids if uuid in uuid_map]
        valid_skill_indices = [node_to_idx[s_id] for s_id in skill_ids if s_id in node_to_idx]
        
        if valid_skill_indices:
            entity_vectors[entity_id] = all_embeddings[valid_skill_indices]
        else:
            entity_vectors[entity_id] = np.array([])
            
    return entity_vectors

def calculate_embedding_match_score(jobseeker_vectors, opportunity_vectors):
    """
    Calculates a match score using vectorized cosine similarity.
    """
    if jobseeker_vectors.size == 0 or opportunity_vectors.size == 0:
        return 0.0
    similarity_matrix = cosine_similarity(opportunity_vectors, jobseeker_vectors)
    max_sim_per_req_skill = similarity_matrix.max(axis=1)
    avg_similarity = max_sim_per_req_skill.mean()
    return avg_similarity

def run_full_analysis_embedding(all_jobseekers, all_opportunities, model, threshold, uuid_map, node_to_idx, output_path):
    """
    Calculates scores for all jobseeker-opportunity pairs.
    """
    jobseeker_skill_vectors = pre_process_entities_for_embedding(all_jobseekers, 'compass_id', uuid_map, model, node_to_idx)
    opportunity_skill_vectors = pre_process_entities_for_embedding(all_opportunities, 'opportunity_ref_id', uuid_map, model, node_to_idx)

    print("\nRunning full analysis for all jobseekers using embedding model...")
    detailed_scores, aggregate_summaries = [], []

    opportunities_with_skills = {
        opp_id: vectors for opp_id, vectors in opportunity_skill_vectors.items() if vectors.size > 0
    }
    total_opportunities_considered = len(opportunities_with_skills)

    if total_opportunities_considered == 0:
        print("No opportunities with skills found. Cannot generate analysis.")
        return

    for jobseeker in tqdm(all_jobseekers, desc="Processing Jobseekers"):
        jobseeker_id = jobseeker.get('compass_id')
        js_vectors = jobseeker_skill_vectors.get(jobseeker_id)

        if js_vectors is None or js_vectors.size == 0:
            aggregate_summaries.append({
                'jobseeker_id': jobseeker_id, 'opportunities_matched': 0,
                'total_opportunities_considered': total_opportunities_considered, 'aggregate_match_percent': 0.0
            })
            continue

        match_count = 0
        for opp_id, opp_vectors in opportunities_with_skills.items():
            score = calculate_embedding_match_score(js_vectors, opp_vectors)
            detailed_scores.append({
                'jobseeker_id': jobseeker_id, 'opportunity_id': opp_id, 'match_score': score
            })
            if score >= threshold:
                match_count += 1
        
        percentage_match = (match_count / total_opportunities_considered) * 100
        aggregate_summaries.append({
            'jobseeker_id': jobseeker_id, 'opportunities_matched': match_count,
            'total_opportunities_considered': total_opportunities_considered, 'aggregate_match_percent': percentage_match
        })

    print("\nSaving results to CSV files...")
    scores_df = pd.DataFrame(detailed_scores)
    summary_df = pd.DataFrame(aggregate_summaries)

    scores_output_path = os.path.join(output_path, 'jobseeker_opportunity_scores_embedding_pyg.csv')
    summary_output_path = os.path.join(output_path, 'jobseeker_aggregate_summary_embedding_pyg.csv')

    scores_df.to_csv(scores_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print(f"Detailed scores saved to: {scores_output_path}")
    print(f"Aggregate summaries saved to: {summary_output_path}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run skill matching with a Node2Vec model.")
    parser.add_argument("--retrain", action="store_true", help="Force the model to be retrained, even if a saved version exists.")
    args = parser.parse_args()

    # --- Define File Paths ---
    TAXONOMY_DATA_PATH = './taxonomy' 
    MAIN_DATA_PATH = 'C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study'
    MODEL_ARTIFACTS_PATH = './trained_model'
    MODEL_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'skill_embedding_model.pt')
    NODE_MAP_PATH = os.path.join(MODEL_ARTIFACTS_PATH, 'node_to_idx.json')

    skills_df, hierarchy_df, relations_df, jobseekers, opportunities = load_data(
        taxonomy_path=TAXONOMY_DATA_PATH, main_data_path=MAIN_DATA_PATH
    )
    if skills_df is None: return

    uuid_to_id_map = create_uuid_to_id_map(skills_df)
    skill_graph, node_to_idx, _ = build_unweighted_skill_graph(skills_df, hierarchy_df, relations_df)
    
    # --- CORE LOGIC: Train or Load Model ---
    if args.retrain or not os.path.exists(MODEL_PATH):
        print("\n--- Training new model ---")
        embedding_model = train_and_save_model(skill_graph, node_to_idx, MODEL_PATH, NODE_MAP_PATH)
        # We use the node_to_idx generated during training
    else:
        print("\n--- Loading existing model ---")
        embedding_model, node_to_idx = load_embedding_model(skill_graph, MODEL_PATH, NODE_MAP_PATH)
        
    if not jobseekers or not opportunities:
        print("No jobseekers or opportunities to analyze.")
        return
        
    MATCH_THRESHOLD = 0.75

    run_full_analysis_embedding(jobseekers, opportunities, embedding_model, MATCH_THRESHOLD, uuid_to_id_map, node_to_idx, MAIN_DATA_PATH)
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()

