#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taxonomy-aware matching over ESCO-like extracts.

Inputs (all in --data_dir):
- pilot_jobseeker_database.json
- pilot_opportunity_database.json
- skills.csv
- skill_groups.csv
- skill_hierarchy.csv
- skill_to_skill_relations.csv
- occupation_to_skill_relations.csv
- occupations.csv
- occupation_hierarchy.csv
- occupation_groups.csv

Outputs (written to --data_dir):
- pairwise_matching_results_with_explanations.csv
- jobseeker_coverage_summary.csv
- matching_config.json
- matching_results.xlsx   (optional; if xlsxwriter is available)
"""

import os, json, math, argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx

# -----------------------------
# Tunable parameters
# -----------------------------
HIER_COST = 1.0           # cost for hierarchy edges (skill/skillgroup)
REL_COST = 1.5            # cost for non-hierarchical skill↔skill relations
LAMBDA = 0.7              # kernel parameter in exp(-λ d)
DIST_CUTOFF = 3           # Dijkstra cutoff for speed
GROUP_RECALL_GATE = 1     # a group's covered if a requirement in it has d <= 1
THETA = {"ess": 1.0, "opt": 0.5, "group": 0.2, "gap_pen": 1.0}  # surplus weights

# -----------------------------
# Utilities
# -----------------------------
def extract_uuid_from_uri(uri: str):
    return uri.rsplit("/", 1)[-1] if isinstance(uri, str) else None

def build_crosswalks(skills_df: pd.DataFrame, occs_df: pd.DataFrame):
    # Skills
    skills_df["SKILL_UUID"] = skills_df["ORIGINURI"].map(extract_uuid_from_uri)
    skill_uuid_to_id = dict(zip(skills_df["SKILL_UUID"], skills_df["ID"]))
    skill_label_to_ids = defaultdict(list)
    for sid, lab in zip(
        skills_df["ID"],
        skills_df["PREFERREDLABEL"].fillna("").str.strip().str.lower()
    ):
        if lab:
            skill_label_to_ids[lab].append(sid)
    skill_id_to_label = dict(zip(skills_df["ID"], skills_df["PREFERREDLABEL"].fillna("")))
    skill_id_to_type  = dict(zip(skills_df["ID"], skills_df["SKILLTYPE"].fillna("")))

    # Occupations
    occs_df["OCC_UUID"] = occs_df["ORIGINURI"].map(extract_uuid_from_uri)
    occ_uuid_to_id  = dict(zip(occs_df["OCC_UUID"], occs_df["ID"]))
    occ_label_to_id = {
        (lab.strip().lower()): oid
        for oid, lab in zip(occs_df["ID"], occs_df["PREFERREDLABEL"].fillna(""))
        if lab
    }

    return {
        "skill_uuid_to_id": skill_uuid_to_id,
        "skill_label_to_ids": skill_label_to_ids,
        "skill_id_to_label": skill_id_to_label,
        "skill_id_to_type":  skill_id_to_type,
        "occ_uuid_to_id":    occ_uuid_to_id,
        "occ_label_to_id":   occ_label_to_id,
    }

def map_skill_uuid_or_label(entry: dict, cw: dict):
    """Resolve a JSON skill {'uuid','preferred_label'} -> internal skill ID."""
    if not isinstance(entry, dict):
        return None
    u = entry.get("uuid")
    if u and u in cw["skill_uuid_to_id"]:
        return cw["skill_uuid_to_id"][u]
    lab = entry.get("preferred_label")
    if isinstance(lab, str):
        labn = lab.strip().lower()
        ids = cw["skill_label_to_ids"].get(labn, [])
        if ids:
            return ids[0]  # deterministic if duplicates
    return None

def map_occ_uuid_or_label(entry: dict, cw: dict):
    """Resolve a JSON occupation {'uuid','preferred_label'} -> internal occupation ID."""
    if not isinstance(entry, dict):
        return None
    u = entry.get("uuid")
    if u and u in cw["occ_uuid_to_id"]:
        return cw["occ_uuid_to_id"][u]
    lab = entry.get("preferred_label")
    if isinstance(lab, str):
        return cw["occ_label_to_id"].get(lab.strip().lower())
    return None

def build_graph(skills_df, skill_groups_df, skill_hier_df, skill_rel_df, skill_id_to_label):
    G = nx.Graph()
    # nodes
    for sid in skills_df["ID"]:
        G.add_node(f"S:{sid}", kind="skill", label=skill_id_to_label.get(sid, ""))
    sg_label = dict(zip(skill_groups_df["ID"], skill_groups_df["PREFERREDLABEL"].fillna("").astype(str)))
    for gid in skill_groups_df["ID"]:
        G.add_node(f"G:{gid}", kind="skillgroup", label=sg_label.get(gid, ""))
    # hierarchy edges
    for _, row in skill_hier_df.iterrows():
        pt = str(row["PARENTOBJECTTYPE"]).strip().lower()
        ct = str(row["CHILDOBJECTTYPE"]).strip().lower()
        pid, cid = row["PARENTID"], row["CHILDID"]
        if pd.isna(pid) or pd.isna(cid):
            continue
        if   pt == "skill"      and ct == "skill":       u, v = f"S:{pid}", f"S:{cid}"
        elif pt == "skillgroup" and ct == "skill":       u, v = f"G:{pid}", f"S:{cid}"
        elif pt == "skill"      and ct == "skillgroup":  u, v = f"S:{pid}", f"G:{cid}"
        elif pt == "skillgroup" and ct == "skillgroup":  u, v = f"G:{pid}", f"G:{cid}"
        else:
            continue
        G.add_edge(u, v, weight=HIER_COST, etype="hierarchy")
    # non-hierarchical relations (undirected)
    for _, row in skill_rel_df.iterrows():
        req_id, reqd_id = row["REQUIRINGID"], row["REQUIREDID"]
        if pd.isna(req_id) or pd.isna(reqd_id):
            continue
        u, v = f"S:{req_id}", f"S:{reqd_id}"
        if not (G.has_node(u) and G.has_node(v)):
            continue
        if G.has_edge(u, v):
            G[u][v]["weight"] = min(G[u][v]["weight"], REL_COST)
            G[u][v]["etype"]  = G[u][v].get("etype", "mixed") + "|relation"
        else:
            G.add_edge(u, v, weight=REL_COST, etype=f"relation:{row.get('RELATIONTYPE','')}")
    return G

def skill_to_groups_map(skill_hier_df):
    m = defaultdict(set)
    for _, row in skill_hier_df.iterrows():
        if str(row["PARENTOBJECTTYPE"]).strip().lower() == "skillgroup" and str(row["CHILDOBJECTTYPE"]).strip().lower() == "skill":
            gid, sid = row["PARENTID"], row["CHILDID"]
            if pd.notna(gid) and pd.notna(sid):
                m[sid].add(gid)
    return m

def occ_canonical_profiles(occ_skill_df):
    occ_ess, occ_opt = defaultdict(set), defaultdict(set)
    for _, row in occ_skill_df.iterrows():
        occ_id, skill_id = row["OCCUPATIONID"], row["SKILLID"]
        rel = str(row["RELATIONTYPE"]).strip().lower()
        if pd.isna(occ_id) or pd.isna(skill_id) or not rel:
            continue
        (occ_ess if rel == "essential" else occ_opt)[occ_id].add(skill_id)
    return occ_ess, occ_opt

def dists_from_skillset(G: nx.Graph, skill_ids, max_dist=DIST_CUTOFF):
    """Multi-source Dijkstra distances from seeker skill IDs to all skill nodes (truncated)."""
    sources = [f"S:{sid}" for sid in skill_ids if G.has_node(f"S:{sid}")]
    dist = {}
    if not sources:
        return {}
    for src in sources:
        this = nx.single_source_dijkstra_path_length(G, src, cutoff=max_dist, weight="weight")
        for node, d in this.items():
            if d <= max_dist and (node not in dist or d < dist[node]):
                dist[node] = d
    # Keep only skill nodes; strip prefix
    out = {}
    for node, d in dist.items():
        if node.startswith("S:"):
            out[node[2:]] = min(out.get(node[2:], float("inf")), d)
    return out

def kernel_exp(d, lam=LAMBDA):
    return 0.0 if math.isinf(d) else math.exp(-lam * d)

def coverage_score(req_ids, match_dists, lam=LAMBDA):
    if not req_ids:
        return 0.0
    vals = [kernel_exp(match_dists.get(r, float("inf")), lam=lam) for r in req_ids]
    return float(np.mean(vals)) if vals else 0.0

def pass_gate_essentials(ess_ids, match_dists, gate=0):
    if not ess_ids:
        return True
    return all(match_dists.get(r, float("inf")) <= gate for r in ess_ids)

def group_recall(req_ids, match_dists, skill_to_groups, gate=GROUP_RECALL_GATE):
    if not req_ids:
        return 0.0
    groups, covered = set(), set()
    for r in req_ids:
        gs = skill_to_groups.get(r, set())
        groups.update(gs)
        if match_dists.get(r, float("inf")) <= gate and gs:
            covered.update(gs)
    return len(covered) / len(groups) if groups else 0.0

# -----------------------------
# Main pipeline
# -----------------------------
def main(data_dir):
    # Load files
    skills_df        = pd.read_csv(os.path.join(data_dir, "skills.csv"))
    skill_groups_df  = pd.read_csv(os.path.join(data_dir, "skill_groups.csv"))
    skill_hier_df    = pd.read_csv(os.path.join(data_dir, "skill_hierarchy.csv"))
    skill_rel_df     = pd.read_csv(os.path.join(data_dir, "skill_to_skill_relations.csv"))
    occ_skill_df     = pd.read_csv(os.path.join(data_dir, "occupation_to_skill_relations.csv"))
    occs_df          = pd.read_csv(os.path.join(data_dir, "occupations.csv"))
    with open(os.path.join(data_dir, "pilot_jobseeker_database.json"), "r", encoding="utf-8") as fh:
        seekers = json.load(fh)
    with open(os.path.join(data_dir, "pilot_opportunity_database_unique.json"), "r", encoding="utf-8") as fh:
        opps = json.load(fh)

    # Crosswalks and graph
    cw = build_crosswalks(skills_df, occs_df)
    G  = build_graph(skills_df, skill_groups_df, skill_hier_df, skill_rel_df, cw["skill_id_to_label"])
    s2g = skill_to_groups_map(skill_hier_df)
    occ_ess, occ_opt = occ_canonical_profiles(occ_skill_df)
    occ_id_to_label = dict(zip(occs_df["ID"], occs_df["PREFERREDLABEL"].fillna("")))

    # Prepare opportunities (posting skills if present; else occupation canonical profile)
    def opp_required_skills(op):
        req_skill_ids = set()
        if op.get("skills"):
            for s in op["skills"]:
                sid = map_skill_uuid_or_label(s, cw)
                if sid:
                    req_skill_ids.add(sid)
        occ = op.get("occupation") or {}
        occ_id = map_occ_uuid_or_label(occ, cw)
        ess_ids, opt_ids = set(), set()
        if not req_skill_ids and occ_id:
            ess_ids = set(occ_ess.get(occ_id, set()))
            opt_ids = set(occ_opt.get(occ_id, set()))
            req_skill_ids = ess_ids | opt_ids
        else:
            ess_ids = set(req_skill_ids)  # treat posting-provided skills as essential
        return {"occupation_id": occ_id, "req": list(req_skill_ids), "ess": list(ess_ids), "opt": list(opt_ids)}

    opps_prepared = []
    for op in opps:
        meta = opp_required_skills(op)
        opps_prepared.append({
            "job_id": op.get("opportunity_ref_id") or op.get("opportunity_group_id") or op.get("id"),
            "title":  op.get("opportunity_title"),
            "active": bool(op.get("active", True)),
            "occ_id": meta["occupation_id"],
            "req":    meta["req"],
            "ess":    meta["ess"],
            "opt":    meta["opt"],
        })

    # Pairwise computations
    pair_rows = []
    coverage_counters = defaultdict(lambda: {"total": 0, "pass_exact": 0, "pass_onehop": 0})

    seeker_skill_ids_cache = {}
    dist_map_cache = {}

    def get_seeker_skill_ids(seeker):
        sid = seeker.get("compass_id")
        if sid in seeker_skill_ids_cache:
            return seeker_skill_ids_cache[sid]
        sids = []
        for s in seeker.get("skills", []):
            mid = map_skill_uuid_or_label(s, cw)
            if mid:
                sids.append(mid)
        sids = list(set(sids))
        seeker_skill_ids_cache[sid] = sids
        return sids

    def get_dist_map(seeker_id):
        if seeker_id in dist_map_cache:
            return dist_map_cache[seeker_id]
        seeker_obj = next(s for s in seekers if s.get("compass_id") == seeker_id)
        sids = get_seeker_skill_ids(seeker_obj)
        dist_map = dists_from_skillset(G, sids, max_dist=DIST_CUTOFF)
        dist_map_cache[seeker_id] = dist_map
        return dist_map

    for seeker in seekers:
        seeker_id = seeker.get("compass_id")
        _ = get_dist_map(seeker_id)  # warm cache
        for op in opps_prepared:
            if not op["active"]:
                continue
            req_ids = op["req"]
            if not req_ids:  # skip postings that could not be mapped at all
                continue

            ess_ids, opt_ids = op["ess"], op["opt"]
            dist_map   = get_dist_map(seeker_id)
            match_dists = {r: dist_map.get(r, float("inf")) for r in req_ids}

            cov_ess = coverage_score(ess_ids, match_dists, lam=LAMBDA) if ess_ids else 0.0
            cov_opt = coverage_score(opt_ids, match_dists, lam=LAMBDA) if opt_ids else 0.0
            grp_rec = group_recall(req_ids, match_dists, s2g, gate=GROUP_RECALL_GATE)
            gap_ess = sum(1 for r in ess_ids if match_dists.get(r, float("inf")) > GROUP_RECALL_GATE)

            surplus = THETA["ess"]*cov_ess + THETA["opt"]*cov_opt + THETA["group"]*grp_rec - THETA["gap_pen"]*bool(gap_ess)
            pass_exact  = pass_gate_essentials(ess_ids, match_dists, gate=0)
            pass_onehop = pass_gate_essentials(ess_ids, match_dists, gate=1)

            coverage_counters[seeker_id]["total"]      += 1
            coverage_counters[seeker_id]["pass_exact"] += int(pass_exact)
            coverage_counters[seeker_id]["pass_onehop"]+= int(pass_onehop)

            # Explainability snippets
            overlaps = []
            for r in ess_ids + [x for x in req_ids if x not in ess_ids]:
                d = match_dists.get(r, float("inf"))
                if d <= 1:
                    overlaps.append({"skill_id": r, "distance": float(d), "label": cw["skill_id_to_label"].get(r, "")})
            overlaps = sorted(overlaps, key=lambda x: (x["distance"], x["label"]))[:5]

            gaps = []
            for r in ess_ids:
                d = match_dists.get(r, float("inf"))
                if d > 1 and d < float("inf"):
                    gaps.append({"skill_id": r, "approx_distance": float(d), "label": cw["skill_id_to_label"].get(r, "")})
            gaps = sorted(gaps, key=lambda x: x["approx_distance"])[:3]

            pair_rows.append({
                "seeker_id": seeker_id,
                "job_id": op["job_id"],
                "job_title": op["title"],
                "occupation_id": op["occ_id"],
                "occupation_label": occ_id_to_label.get(op["occ_id"], None),
                "n_req": len(req_ids),
                "n_ess": len(ess_ids),
                "n_opt": len(opt_ids),
                "cov_ess": round(cov_ess, 4),
                "cov_opt": round(cov_opt, 4),
                "group_recall": round(grp_rec, 4),
                "gap_essentials": int(gap_ess),
                "surplus": round(surplus, 4),
                "pass_exact": bool(pass_exact),
                "pass_onehop": bool(pass_onehop),
                "top_overlaps": json.dumps(overlaps, ensure_ascii=False),
                "top_gaps": json.dumps(gaps, ensure_ascii=False),
            })

    pair_df = pd.DataFrame(pair_rows)

    # Aggregate “you have the key skills for XX% of opportunities”
    summary_rows = []
    for seeker in seekers:
        sid = seeker.get("compass_id")
        c = coverage_counters[sid]
        tot = c["total"]
        exact_pct  = 100.0 * c["pass_exact"]  / tot if tot else 0.0
        onehop_pct = 100.0 * c["pass_onehop"] / tot if tot else 0.0
        summary_rows.append({
            "seeker_id": sid,
            "total_opportunities": tot,
            "pass_exact_count": c["pass_exact"],
            "pass_onehop_count": c["pass_onehop"],
            "key_skills_coverage_pct_exact": round(exact_pct, 2),
            "key_skills_coverage_pct_onehop": round(onehop_pct, 2),
        })
    summary_df = pd.DataFrame(summary_rows)

    # Write outputs
    out_pair    = os.path.join(data_dir, "pairwise_matching_results_with_explanations.csv")
    out_summary = os.path.join(data_dir, "jobseeker_coverage_summary.csv")
    pair_df.to_csv(out_pair, index=False)
    summary_df.to_csv(out_summary, index=False)

    # Also save config used
    cfg = {
        "graph_edge_weights": {"hierarchy": HIER_COST, "skill_relation": REL_COST},
        "kernel": "exp(-lambda * d)", "lambda": LAMBDA,
        "distance_cutoff": DIST_CUTOFF,
        "group_recall_gate": GROUP_RECALL_GATE,
        "theta": THETA,
        "assumptions": [
            "If a posting lists skills, they are treated as essential.",
            "If a posting lacks skills, use the occupation's canonical essential/optional sets.",
            "Skills/occupations are mapped by UUID when possible; otherwise by exact preferred_label (case-insensitive).",
            "Graph = skills + skillgroups; hierarchy edges cost 1.0; non-hierarchical relations cost 1.5; distances truncated.",
            "Coverage uses per-requirement minimum distance from the seeker."
        ]
    }
    with open(os.path.join(data_dir, "matching_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)

    # Optional Excel workbook (if xlsxwriter is available)
    try:
        xlsx_path = os.path.join(data_dir, "matching_results.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xlw:
            pair_df.to_excel(xlw, sheet_name="Pairwise", index=False)
            summary_df.to_excel(xlw, sheet_name="CoverageSummary", index=False)
    except Exception:
        pass

    print("Wrote:", out_pair, out_summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".", help="Directory with the provided JSON/CSV extracts")
    args = parser.parse_args()
    main(args.data_dir)
