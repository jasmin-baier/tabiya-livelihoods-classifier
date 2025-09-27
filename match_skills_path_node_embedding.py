import pandas as pd
import json
import networkx as nx
import numpy as np
import os
import argparse # Import argparse for command-line arguments
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")          # important on servers/headless; put BEFORE pyplot
import matplotlib.pyplot as plt
from pathlib import Path
from decimal import Decimal

# NOTE: change jobseeker and opportunity data paths at the bottom if needed
#    TAXONOMY_DATA_PATH = './taxonomy' 
#    MAIN_DATA_PATH = 'C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study'
# MATCH_THRESHOLD = 0.5 ; technically 0.75 would be more robust, but this didn't lead to any matches...

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

def create_id_to_label_map(skills_df):
    """
    Creates a dictionary to map canonical skill ID to its preferred label for readability.
    """
    print("Creating ID to Label mapping for explanations...")
    id_to_label_map = {}
    skills_df = skills_df.dropna(subset=['ID', 'PREFERREDLABEL'])
    for _, row in skills_df.iterrows():
        id_to_label_map[row['ID']] = row['PREFERREDLABEL']
    return id_to_label_map

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
            G.add_edge(parent, child, etype='is_a')

    for _, row in relations_df.iterrows():
        requiring = node_to_idx.get(row['REQUIRINGID'])
        required = node_to_idx.get(row['REQUIREDID'])
        if requiring is not None and required is not None:
            G.add_edge(requiring, required, etype='related')
            
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
    
    model = Node2Vec(
        edge_index, embedding_dim=64, walk_length=30, context_size=10,
        walks_per_node=20, num_negative_samples=1, p=1.0, q=1.0, sparse=True,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("Model and mappings loaded successfully.")
    return model, node_to_idx

def get_jobseeker_skills_uuid(jobseeker_data):
    """
    Extracts a simple list of skill UUIDs from a jobseeker's record.
    """
    if not jobseeker_data.get('skills'):
        return []
    
    all_uuids = []
    for skill in jobseeker_data.get('skills', []):
        uuid_value = skill.get('uuid')
        if not uuid_value:
            continue
        
        if isinstance(uuid_value, list):
            all_uuids.extend(uuid_value)
        else:
            all_uuids.append(str(uuid_value))
            
    return all_uuids

def get_opportunity_skills_uuid(opportunity_data):
    """
    Extracts a simple list of skill UUIDs from an opportunity's record.
    """
    if not opportunity_data.get('skills'):
        return []

    all_uuids = []
    for skill in opportunity_data.get('skills', []):
        uuid_value = skill.get('uuid')
        if not uuid_value:
            continue
        
        if isinstance(uuid_value, list):
            all_uuids.extend(uuid_value)
        else:
            all_uuids.append(str(uuid_value))

    return all_uuids

def pre_process_entities_for_embedding(entities, id_getter_func, uuid_map, model, node_to_idx):
    """
    Pre-processes entities to fetch their skill vectors from the loaded/trained model.
    """
    print(f"Pre-processing entities to fetch skill vectors...")
    entity_vectors = {}
    with torch.no_grad():
        all_embeddings = model.embedding.weight.cpu().numpy()

    for entity in tqdm(entities, desc="Fetching skill vectors"):
        raw_entity_id = entity.get(id_getter_func)
        if isinstance(raw_entity_id, list):
            if not raw_entity_id: continue
            entity_id = raw_entity_id[0]
        else:
            entity_id = raw_entity_id

        if not entity_id: continue
        
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
    # Force stable, string IDs (no floats/scientific notation)
    scores_df['jobseeker_id']  = scores_df['jobseeker_id'].map(lambda v: str(int(v)) if pd.notnull(v) else None)
    scores_df['opportunity_id'] = scores_df['opportunity_id'].map(lambda v: str(int(v)) if pd.notnull(v) else None)

    summary_df = pd.DataFrame(aggregate_summaries)

    scores_output_path = os.path.join(output_path, 'jobseeker_opportunity_scores_embedding_pyg.csv')
    summary_output_path = os.path.join(output_path, 'jobseeker_aggregate_summary_embedding_pyg.csv')

    scores_df.to_csv(scores_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print(f"Detailed scores saved to: {scores_output_path}")
    print(f"Aggregate summaries saved to: {summary_output_path}")

def _uuid_to_graph_idx(skill_uuid, uuid_to_id_map, node_to_idx):
    # uuid -> taxonomy ID -> graph index
    tax_id = uuid_to_id_map.get(skill_key, skill_key)
    return node_to_idx.get(tax_id)

def _shortest_path_with_labels(G, s_idx, t_idx):
    try:
        path = nx.shortest_path(G, s_idx, t_idx)
    except nx.NetworkXNoPath:
        return None, None
    # Edge labels by etype along the path
    etypes = []
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v) or {}
        etypes.append(data.get('etype', 'edge'))
    return path, etypes

def visualize_match_paths(
    jobseeker_skills, opportunity_skills, 
    best_pairs,                      # list[(req_uuid, js_uuid, sim)]
    G, uuid_to_id_map, node_to_idx, id_to_label_map,
    out_png_path, topk_paths_per_pair=1, max_pairs=6
):
    """
    Build a small subgraph comprising shortest paths between the top skill pairs.
    Colors:
      - Jobseeker skills: blue nodes
      - Opportunity skills: orange nodes
      - Intermediates: light gray
    Edge colors:
      - is_a: green
      - related: purple
    """
    # --- pick top-N pairs to keep the figure readable
    best_pairs = sorted(best_pairs, key=lambda x: x[2], reverse=True)[:max_pairs]

    # --- collect nodes/edges
    nodes_role = {}  # node_idx -> role ("js", "opp", "mid")
    edges_type_count = defaultdict(int)
    H = nx.Graph()

    for req_uuid, js_uuid, sim in best_pairs:
        s_idx = _uuid_to_graph_idx(js_uuid, uuid_to_id_map, node_to_idx)
        t_idx = _uuid_to_graph_idx(req_uuid, uuid_to_id_map, node_to_idx)
        if s_idx is None or t_idx is None:
            continue

        path, etypes = _shortest_path_with_labels(G, s_idx, t_idx)
        if not path:
            continue

        # Add nodes with roles
        nodes_role.setdefault(s_idx, 'js')
        nodes_role.setdefault(t_idx, 'opp')
        for n in path:
            nodes_role.setdefault(n, 'mid')

        # Add edges with etype + count
        for (u, v), et in zip(zip(path[:-1], path[1:]), etypes):
            H.add_edge(u, v, etype=et)
            edges_type_count[et] += 1

    if H.number_of_nodes() == 0:
        print("No paths to visualize.")
        return

    # --- draw
    pos = nx.spring_layout(H, seed=42, k=0.5)  # stable layout
    node_colors = []
    node_labels = {}
    for n in H.nodes():
        role = nodes_role.get(n, 'mid')
        if role == 'js':
            node_colors.append('#1f77b4')  # blue
        elif role == 'opp':
            node_colors.append('#ff7f0e')  # orange
        else:
            node_colors.append('#d3d3d3')  # light gray

        # back-map to label (graph idx -> taxonomy ID -> label)
        tax_id = list(node_to_idx.keys())[list(node_to_idx.values()).index(n)]
        node_labels[n] = id_to_label_map.get(tax_id, str(tax_id))

    # edge colors by etype
    edge_colors = []
    for u, v, data in H.edges(data=True):
        et = data.get('etype', 'edge')
        edge_colors.append('#2ca02c' if et == 'is_a' else '#9467bd')  # green vs purple

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_nodes(H, pos, node_size=600, node_color=node_colors, linewidths=0.5, edgecolors='black')
    nx.draw_networkx_labels(H, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edges(H, pos, width=2, edge_color=edge_colors, alpha=0.9)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200)
    plt.close()

    # quick textual summary to pair with the image
    total_edges = sum(edges_type_count.values())
    if total_edges:
        share_is_a = edges_type_count['is_a'] / total_edges
        share_rel  = edges_type_count['related'] / total_edges
        print(f"Explanation graph saved to {out_png_path}")
        print(f"Path composition → is_a: {share_is_a:.1%}, related: {share_rel:.1%}")

def plot_alignment_heatmap(req_ids, js_ids, sim_matrix, id_to_label_map, out_png):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(min(18, 1 + 0.35*len(js_ids)), min(12, 1 + 0.35*len(req_ids))))
    plt.imshow(sim_matrix, aspect='auto', interpolation='nearest')
    plt.colorbar(label='cosine similarity')
    plt.xticks(range(len(js_ids)), [id_to_label_map.get(i,i) for i in js_ids], rotation=90, fontsize=8)
    plt.yticks(range(len(req_ids)), [id_to_label_map.get(i,i) for i in req_ids], fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def edge_label_with_direction(G, u, v):
    d = G.get_edge_data(u, v) or {}
    if d.get('etype') != 'is_a':
        return 'related'
    # you added edges as: G.add_edge(p, c, etype='is_a', parent=p, child=c)
    if d.get('child') == u and d.get('parent') == v:
        return 'is_a↑'   # going to a broader concept (child -> parent)
    if d.get('parent') == u and d.get('child') == v:
        return 'is_a↓'   # going to a narrower concept (parent -> child)
    return 'is_a'        # fallback

def describe_path(G, path, id_to_label_map, node_to_idx):
    labels = []
    for n in path:
        tax_id = next(k for k, idx in node_to_idx.items() if idx == n)
        labels.append(id_to_label_map.get(tax_id, str(tax_id)))
    parts = []
    for (u, v), lbl in zip(zip(path[:-1], path[1:]), labels):
        parts.append(lbl)
        et = edge_label_with_direction(G, u, v)
        parts.append(f" —{et}→ ")
    parts.append(labels[-1])
    return ''.join(parts)

def explain_match_details(jobseeker_id, opportunity_id, all_jobseekers, all_opportunities, model, uuid_map, node_to_idx, id_to_label_map, skill_graph=None, out_dir="."):
    """
    Provides a skill-by-skill breakdown of a specific jobseeker-opportunity match.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Match Explanation ---")

    def normalize_id(value):
        if value is None:
            return None
        try:
            return str(int(value))        
        except (ValueError, TypeError):
            # last resort: strip decimals/scientific notation safely
            s = str(value)
            if "e+" in s.lower() or "e-" in s.lower():
                try:
                    return str(int(Decimal(s)))
                except Exception:
                    pass
            return ''.join(ch for ch in s if ch.isdigit()) or s

    jobseeker_id = normalize_id(jobseeker_id)
    opportunity_id = normalize_id(opportunity_id)

    jobseeker = next((js for js in all_jobseekers
                      if normalize_id(js.get('compass_id')) == jobseeker_id), None)
    opportunity = next((opp for opp in all_opportunities
                        if normalize_id(opp.get('opportunity_ref_id')) == opportunity_id), None)

    if not jobseeker or not opportunity:
        print(f"Could not find Jobseeker {jobseeker_id} or Opportunity {opportunity_id}")
        return

    js_uuids = get_jobseeker_skills_uuid(jobseeker)
    opp_uuids = get_opportunity_skills_uuid(opportunity)

    js_skill_ids = [uuid_map[uuid] for uuid in js_uuids if uuid in uuid_map]
    opp_skill_ids = [uuid_map[uuid] for uuid in opp_uuids if uuid in uuid_map]
    
    with torch.no_grad():
        all_embeddings = model.embedding.weight.cpu().numpy()

    js_indices = [node_to_idx[s_id] for s_id in js_skill_ids if s_id in node_to_idx]
    opp_indices = [node_to_idx[s_id] for s_id in opp_skill_ids if s_id in node_to_idx]

    if not js_indices or not opp_indices:
        print("No common skills found in the model for this pair.")
        return

    js_vectors = all_embeddings[js_indices]
    opp_vectors = all_embeddings[opp_indices]
    
    js_skill_ids_with_vectors = [s_id for s_id in js_skill_ids if s_id in node_to_idx]
    opp_skill_ids_with_vectors = [s_id for s_id in opp_skill_ids if s_id in node_to_idx]

    similarity_matrix = cosine_similarity(opp_vectors, js_vectors)

    # --- save alignment heatmap ---
    heatmap_path = out_dir / f"alignment_heatmap_js_{jobseeker_id}_opp_{opportunity_id}.png"

    plot_alignment_heatmap(
        req_ids=opp_skill_ids_with_vectors,
        js_ids=js_skill_ids_with_vectors,
        sim_matrix=similarity_matrix,
        id_to_label_map=id_to_label_map,
        out_png=heatmap_path
    )
    print(f"[Saved] {heatmap_path}")
    
    print(f"\nBreakdown for Jobseeker '{jobseeker_id}' and Opportunity '{opportunity_id}':\n")
    print(f"{'Opportunity Skill':<50} | {'Best Match from Jobseeker':<50} | {'Similarity':<10}")
    print("-" * 115)

    for i, req_skill_id in enumerate(opp_skill_ids_with_vectors):
        best_match_idx = similarity_matrix[i].argmax()
        best_score = similarity_matrix[i].max()
        best_match_skill_id = js_skill_ids_with_vectors[best_match_idx]
        
        req_label = id_to_label_map.get(req_skill_id, req_skill_id)
        match_label = id_to_label_map.get(best_match_skill_id, best_match_skill_id)
        
        print(f"{req_label:<50} | {match_label:<50} | {best_score:.4f}")
    
    # --- print top-K path stories (shortest paths) ---
    TOP_K = 8  # keep the console readable
    pairs = []
    for i, req_id in enumerate(opp_skill_ids_with_vectors):
        j = int(np.argmax(similarity_matrix[i]))
        pairs.append((req_id, js_skill_ids_with_vectors[j], float(similarity_matrix[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:TOP_K]

    if skill_graph is not None:
        print("\nMost explanatory paths through the skills graph:")
        for req_id, js_id, sim in pairs:
            s_idx = node_to_idx.get(js_id)
            t_idx = node_to_idx.get(req_id)
            if s_idx is None or t_idx is None:
                continue
            try:
                path = nx.shortest_path(skill_graph, s_idx, t_idx)
                story = describe_path(skill_graph, path, id_to_label_map, node_to_idx)
                print(f"  • {story}  (cos={sim:.3f})")
            except nx.NetworkXNoPath:
                pass

    pairs = []
    for i, req_skill_id in enumerate(opp_skill_ids_with_vectors):
        j = similarity_matrix[i].argmax()
        sim = float(similarity_matrix[i, j])
        best_match_skill_id = js_skill_ids_with_vectors[j]
        # store UUIDs (not graph indices) so we can map later
        pairs.append((req_skill_id, best_match_skill_id, sim))

    # Pick a filename and draw
    out_png = out_dir / f"explain_graph_js_{jobseeker_id}_opp_{opportunity_id}.png"

    # IMPORTANT: pass the actual graph object
    visualize_match_paths(
        jobseeker_skills=js_skill_ids,              # or js_skill_ids_with_vectors
        opportunity_skills=opp_skill_ids,           # or opp_skill_ids_with_vectors
        best_pairs=pairs,
        G=skill_graph,
        uuid_to_id_map=uuid_map,
        node_to_idx=node_to_idx,
        id_to_label_map=id_to_label_map,
        out_png_path=str(out_png),
        topk_paths_per_pair=1,
        max_pairs=6
    )





def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run skill matching with a Node2Vec model.")
    parser.add_argument("--retrain", action="store_true", help="Force the model to be retrained, even if a saved version exists.")
    args = parser.parse_args()

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
    id_to_label_map = create_id_to_label_map(skills_df)
    skill_graph, node_to_idx, _ = build_unweighted_skill_graph(skills_df, hierarchy_df, relations_df)
    
    if args.retrain or not os.path.exists(MODEL_PATH):
        print("\n--- Training new model ---")
        embedding_model = train_and_save_model(skill_graph, node_to_idx, MODEL_PATH, NODE_MAP_PATH)
    else:
        print("\n--- Loading existing model ---")
        embedding_model, node_to_idx = load_embedding_model(skill_graph, MODEL_PATH, NODE_MAP_PATH)
        
    if not jobseekers or not opportunities:
        print("No jobseekers or opportunities to analyze.")
        return
        
    MATCH_THRESHOLD = 0.5

    run_full_analysis_embedding(jobseekers, opportunities, embedding_model, MATCH_THRESHOLD, uuid_to_id_map, node_to_idx, MAIN_DATA_PATH)
    
    try:
        TOP_N = 10
        results_fp = os.path.join(MAIN_DATA_PATH, "jobseeker_opportunity_scores_embedding_pyg.csv")

        # Keep IDs as strings to avoid float/scientific notation issues
        results_df = pd.read_csv(results_fp, dtype={"jobseeker_id": "string", "opportunity_id": "string"})
        results_df["match_score"] = pd.to_numeric(results_df["match_score"], errors="coerce")

        # (optional) if your CSV might have duplicates
        # results_df = results_df.drop_duplicates(subset=["jobseeker_id", "opportunity_id"])

        top_n = results_df.nlargest(TOP_N, "match_score")

        out_dir = os.path.join(MAIN_DATA_PATH, f"explanations_top{TOP_N}")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for _, row in top_n.iterrows():
            js_id  = row["jobseeker_id"]
            opp_id = row["opportunity_id"]
            score  = row["match_score"]
            print(f"\n=== Explaining JS {js_id} vs OPP {opp_id} (score={score:.3f}) ===")
            try:
                explain_match_details(
                    js_id, opp_id,
                    jobseekers, opportunities,
                    embedding_model, uuid_to_id_map, node_to_idx, id_to_label_map,
                    skill_graph=skill_graph,      # make sure explain_match_details accepts this
                    out_dir=out_dir               # and this
                )
            except Exception as e:
                print(f"Skipping pair due to error: {e}")

    except FileNotFoundError:
        print("\nCould not find results file to generate explanation. Run analysis first.")
    except Exception as e:
        print(f"\nAn error occurred during explanation generation: {e}")


    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()

