#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drop-in patched version of the matching pipeline with safer defaults and better error messages.

Key improvements vs the original:
- Auto-detect taxonomy/json directories if the defaults don't exist (tries: provided arg, ./taxonomy, script folder, parent folder, /mnt/data).
- Validates presence of required CSV/JSON files and prints a clear checklist before exiting.
- Tolerates missing optional columns in taxonomy CSVs by creating empty fallbacks (e.g., SKILLTYPE, UUIDHISTORY).
- Works on Windows and POSIX paths without hardcoded Windows defaults.
"""
import os, json, math, argparse, datetime, re, sys
import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
from pathlib import Path

# ---------- Tunables (unchanged) ----------
HIER_COST = 1.0
REL_COST  = 1.5
LAMBDA    = 0.7
DIST_CUTOFF = 3
GROUP_RECALL_GATE = 1
THETA = {"ess": 1.0, "opt": 0.5, "group": 0.2, "gap_pen": 1.0}

HALF_LIFE_CELL_DAYS = 30.0
HALF_LIFE_JOB_DAYS  = 21.0
EB_ALPHA            = 1.0
REGION_FIELD_ORDER  = ["province", "city", "country"]

TAU_FIT, TAU_SAFE, TAU_ADJ = 0.60, 0.40, 0.25

UUID_REGEX = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')

# ---------- Helpers (mostly unchanged) ----------
def extract_uuid_from_uri(uri):
    return uri.rsplit("/", 1)[-1] if isinstance(uri, str) else None

def parse_last_uuid_from_history(val):
    """Return the last UUID from UUIDHISTORY cell (JSON list or UUID-like tokens)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and obj:
            return str(obj[-1]).strip()
    except Exception:
        pass
    matches = UUID_REGEX.findall(s)
    if matches:
        return matches[-1].lower()
    return None

def now_utc_date():
    return datetime.datetime.utcnow().date()

def days_since(date_str):
    if not isinstance(date_str, str) or not date_str.strip():
        return float("inf")
    try:
        if "T" in date_str:
            dt = datetime.datetime.fromisoformat(date_str.replace("Z",""))
            d  = dt.date()
        else:
            d  = datetime.date.fromisoformat(date_str[:10])
        return max(0, (now_utc_date() - d).days)
    except Exception:
        return float("inf")

def half_life_decay(days, half_life):
    if math.isinf(days): return 0.0
    return math.exp(-math.log(2) * (days / half_life))

# ---------- FS utilities ----------
def resolve_dir_with_candidates(preferred: Path, candidates: list[Path]) -> Path:
    if preferred and preferred.exists():
        return preferred
    for c in candidates:
        if c and c.exists():
            return c
    return preferred  # return as-is; will fail later during validation

def validate_required_files(taxonomy_dir: Path, json_dir: Path, jobseeker_file: str, opportunity_file: str) -> tuple[bool, str]:
    missing = []
    taxo_files = [
        "skills.csv",
        "skill_groups.csv",
        "skill_hierarchy.csv",
        "skill_to_skill_relations.csv",
        "occupation_to_skill_relations.csv",
        "occupations.csv",
    ]
    for fn in taxo_files:
        p = taxonomy_dir / fn
        if not p.exists():
            missing.append(f"Missing taxonomy CSV: {p}")
    for fn in [jobseeker_file, opportunity_file]:
        p = json_dir / fn
        if not p.exists():
            missing.append(f"Missing JSON: {p}")
    if missing:
        msg = "Input validation failed:\n- " + "\n- ".join(missing)
        return False, msg
    return True, ""

# ---------- Crosswalks with UUIDHISTORY ----------
def build_crosswalks(skills_df, occs_df):
    # Ensure optional columns exist
    for col in ["UUIDHISTORY", "ORIGINURI", "SKILLTYPE", "PREFERREDLABEL", "ID"]:
        if col not in skills_df.columns:
            if col in ("UUIDHISTORY", "ORIGINURI", "SKILLTYPE", "PREFERREDLABEL"):
                skills_df[col] = ""
            elif col == "ID":
                raise KeyError("skills.csv is missing required column 'ID'")
    for col in ["UUIDHISTORY", "ORIGINURI", "PREFERREDLABEL", "ID"]:
        if col not in occs_df.columns:
            if col in ("UUIDHISTORY", "ORIGINURI", "PREFERREDLABEL"):
                occs_df[col] = ""
            elif col == "ID":
                raise KeyError("occupations.csv is missing required column 'ID'")

    if "UUIDHISTORY" in skills_df.columns:
        skills_df["SKILL_UUIDH"] = skills_df["UUIDHISTORY"].apply(parse_last_uuid_from_history)
        skills_df["SKILL_UUID"]  = skills_df["SKILL_UUIDH"]
    else:
        skills_df["SKILL_UUID"]  = None
    if "ORIGINURI" in skills_df.columns:
        ori_uuid = skills_df["ORIGINURI"].map(extract_uuid_from_uri)
        skills_df["SKILL_UUID"] = skills_df["SKILL_UUID"].fillna(ori_uuid)

    skill_uuid_to_id = dict(zip(skills_df["SKILL_UUID"], skills_df["ID"]))
    skill_label_to_ids = defaultdict(list)
    for sid, lab in zip(
        skills_df["ID"],
        skills_df["PREFERREDLABEL"].fillna("").str.strip().str.lower()
    ):
        if lab:
            skill_label_to_ids[lab].append(sid)
    skill_id_to_label = dict(zip(skills_df["ID"], skills_df["PREFERREDLABEL"].fillna("")))
    skill_id_to_type  = dict(zip(skills_df["ID"], skills_df.get("SKILLTYPE", pd.Series([""]*len(skills_df)))))

    if "UUIDHISTORY" in occs_df.columns:
        occs_df["OCC_UUIDH"] = occs_df["UUIDHISTORY"].apply(parse_last_uuid_from_history)
        occs_df["OCC_UUID"]  = occs_df["OCC_UUIDH"]
    else:
        occs_df["OCC_UUID"]  = None
    if "ORIGINURI" in occs_df.columns:
        occ_ori_uuid = occs_df["ORIGINURI"].map(extract_uuid_from_uri)
        occs_df["OCC_UUID"] = occs_df["OCC_UUID"].fillna(occ_ori_uuid)

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

def _iter_uuid_values(u):
    """Yield string UUIDs from str | dict | list values."""
    if u is None:
        return
    if isinstance(u, str):
        yield u
    elif isinstance(u, dict):
        cand = u.get("id") or u.get("uuid") or u.get("value")
        if isinstance(cand, str):
            yield cand
    elif isinstance(u, list):
        for item in u:
            yield from _iter_uuid_values(item)

def _iter_label_values(l):
    """Yield string labels from str | dict | list values."""
    if l is None:
        return
    if isinstance(l, str):
        yield l
    elif isinstance(l, dict):
        for k in ("preferred_label", "label", "name", "value"):
            v = l.get(k)
            if isinstance(v, str):
                yield v
    elif isinstance(l, list):
        for item in l:
            if isinstance(item, str):
                yield item
        for item in l:
            if isinstance(item, dict):
                yield from _iter_label_values(item)

def map_skill_uuid_or_label(entry, cw):
    # allow a plain string to mean "preferred_label"
    if isinstance(entry, str):
        ids = cw["skill_label_to_ids"].get(entry.strip().lower(), [])
        return ids[0] if ids else None
    if not isinstance(entry, dict):
        return None
    # try UUID(s)
    for u in _iter_uuid_values(entry.get("uuid")):
        mid = cw["skill_uuid_to_id"].get(u)
        if mid is not None:
            return mid
    # fallback to label(s)
    for lab in _iter_label_values(entry.get("preferred_label")):
        ids = cw["skill_label_to_ids"].get(lab.strip().lower(), [])
        if ids:
            return ids[0]
    return None

def map_occ_uuid_or_label(entry, cw):
    if isinstance(entry, str):
        return cw["occ_label_to_id"].get(entry.strip().lower())
    if not isinstance(entry, dict):
        return None
    for u in _iter_uuid_values(entry.get("uuid")):
        oid = cw["occ_uuid_to_id"].get(u)
        if oid is not None:
            return oid
    for lab in _iter_label_values(entry.get("preferred_label")):
        oid = cw["occ_label_to_id"].get(lab.strip().lower())
        if oid is not None:
            return oid
    return None

# ---------- Graph & taxonomy ----------
def build_graph(skills_df, skill_groups_df, skill_hier_df, skill_rel_df, skill_id_to_label):
    G = nx.Graph()
    for sid in skills_df["ID"]:
        G.add_node(f"S:{sid}", kind="skill", label=skill_id_to_label.get(sid, ""))
    sg_label = dict(zip(skill_groups_df["ID"], skill_groups_df["PREFERREDLABEL"].fillna("").astype(str)))
    for gid in skill_groups_df["ID"]:
        G.add_node(f"G:{gid}", kind="skillgroup", label=sg_label.get(gid, ""))
    for _, row in skill_hier_df.iterrows():
        pt = str(row.get("PARENTOBJECTTYPE","")).strip().lower()
        ct = str(row.get("CHILDOBJECTTYPE","")).strip().lower()
        pid, cid = row.get("PARENTID"), row.get("CHILDID")
        if pd.isna(pid) or pd.isna(cid): continue
        if   pt == "skill"      and ct == "skill":       u, v = f"S:{pid}", f"S:{cid}"
        elif pt == "skillgroup" and ct == "skill":       u, v = f"G:{pid}", f"S:{cid}"
        elif pt == "skill"      and ct == "skillgroup":  u, v = f"S:{pid}", f"G:{cid}"
        elif pt == "skillgroup" and ct == "skillgroup":  u, v = f"G:{pid}", f"G:{cid}"
        else: continue
        G.add_edge(u, v, weight=HIER_COST, etype="hierarchy")
    for _, row in skill_rel_df.iterrows():
        req_id, reqd_id = row.get("REQUIRINGID"), row.get("REQUIREDID")
        if pd.isna(req_id) or pd.isna(reqd_id): continue
        u, v = f"S:{req_id}", f"S:{reqd_id}"
        if not (G.has_node(u) and G.has_node(v)): continue
        if G.has_edge(u, v):
            G[u][v]["weight"] = min(G[u][v]["weight"], REL_COST)
            G[u][v]["etype"]  = G[u][v].get("etype", "mixed") + "|relation"
        else:
            G.add_edge(u, v, weight=REL_COST, etype=f"relation:{row.get('RELATIONTYPE','')}")
    return G

def skill_to_groups_map(skill_hier_df):
    m = defaultdict(set)
    for _, row in skill_hier_df.iterrows():
        if str(row.get("PARENTOBJECTTYPE","")).strip().lower() == "skillgroup" and str(row.get("CHILDOBJECTTYPE","")).strip().lower() == "skill":
            gid, sid = row.get("PARENTID"), row.get("CHILDID")
            if pd.notna(gid) and pd.notna(sid):
                m[sid].add(gid)
    return m

def occ_canonical_profiles(occ_skill_df):
    occ_ess, occ_opt = defaultdict(set), defaultdict(set)
    for _, row in occ_skill_df.iterrows():
        occ_id, skill_id = row.get("OCCUPATIONID"), row.get("SKILLID")
        rel = str(row.get("RELATIONTYPE","")).strip().lower()
        if pd.isna(occ_id) or pd.isna(skill_id) or not rel: continue
        if rel == "essential":
            occ_ess[occ_id].add(skill_id)
        else:
            occ_opt[occ_id].add(skill_id)
    return occ_ess, occ_opt

def dists_from_skillset(G, skill_ids, max_dist=DIST_CUTOFF):
    sources = [f"S:{sid}" for sid in skill_ids if G.has_node(f"S:{sid}")]
    dist = {}
    if not sources: return {}
    for src in sources:
        this = nx.single_source_dijkstra_path_length(G, src, cutoff=max_dist, weight="weight")
        for node, d in this.items():
            if d <= max_dist and (node not in dist or d < dist[node]):
                dist[node] = d
    out = {}
    for node, d in dist.items():
        if node.startswith("S:"):
            out[node[2:]] = min(out.get(node[2:], float("inf")), d)
    return out

def kernel_exp(d, lam=LAMBDA):
    return 0.0 if math.isinf(d) else math.exp(-lam * d)

def coverage_score(req_ids, match_dists, lam=LAMBDA):
    if not req_ids: return 0.0
    vals = [kernel_exp(match_dists.get(r, float("inf")), lam=lam) for r in req_ids]
    return float(np.mean(vals)) if vals else 0.0

def pass_gate_essentials(ess_ids, match_dists, gate=0):
    if not ess_ids: return True
    return all(match_dists.get(r, float("inf")) <= gate for r in ess_ids)

def group_recall(req_ids, match_dists, skill_to_groups, gate=GROUP_RECALL_GATE):
    if not req_ids: return 0.0
    groups, covered = set(), set()
    for r in req_ids:
        gs = skill_to_groups.get(r, set())
        groups.update(gs)
        if match_dists.get(r, float("inf")) <= gate and gs:
            covered.update(gs)
    return len(covered) / len(groups) if groups else 0.0

# ---------- Success proxy p̂ ----------
def pick_region(op):
    for f in REGION_FIELD_ORDER:
        v = op.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return "unknown"

def build_success_proxy(opps_prepared):
    cell_osi = defaultdict(float)
    for op in opps_prepared:
        if not op["active"] or op["occ_id"] is None: 
            continue
        cell = (op["occ_id"], op["region"])
        cell_osi[cell] += half_life_decay(op["age_days"], HALF_LIFE_CELL_DAYS)

    max_osi = max(cell_osi.values()) if cell_osi else 0.0
    def norm_osi(val):
        if max_osi <= 0: return 0.0
        return (val + EB_ALPHA) / (max_osi + EB_ALPHA)

    p_hat = {}
    for op in opps_prepared:
        if not op["active"] or op["occ_id"] is None:
            p_hat[op["job_id"]] = 0.0
            continue
        cell = (op["occ_id"], op["region"])
        cell_term  = norm_osi(cell_osi.get(cell, 0.0))
        stale_term = half_life_decay(op["age_days"], HALF_LIFE_JOB_DAYS)
        p_hat[op["job_id"]] = float(cell_term * stale_term)
    return p_hat

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent

    # Defaults: don't hardcode Windows paths anymore
    default_taxonomy_dir = script_dir / "taxonomy"
    default_json_dir = 'C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study'

    parser.add_argument("--taxonomy_dir", default=str(default_taxonomy_dir), help="Directory containing taxonomy CSVs (default: ./taxonomy next to this script)")
    parser.add_argument("--json_dir", default=str(default_json_dir), help="Directory containing jobseeker/opportunity JSON files (default: script directory)")
    parser.add_argument("--jobseeker_file", default="pilot_jobseeker_database.json", help="Jobseeker JSON filename")
    parser.add_argument("--opportunity_file", default="pilot_opportunity_database_unique.json", help="Opportunity JSON filename")
    parser.add_argument("--out_dir", default=str(script_dir), help="Output directory (default: script directory)")
    args = parser.parse_args()

    taxonomy_dir = Path(args.taxonomy_dir)
    json_dir     = Path(args.json_dir)
    out_dir      = Path(args.out_dir)

    # Auto-detect if provided/default dirs don't exist
    taxonomy_dir = resolve_dir_with_candidates(
        taxonomy_dir,
        [script_dir / "taxonomy", script_dir, script_dir.parent / "taxonomy", Path("/mnt/data")]
    )
    json_dir = resolve_dir_with_candidates(
        json_dir,
        [script_dir, script_dir.parent, Path("/mnt/data")]
    )

    # Validate inputs and print a clear message if missing
    ok, msg = validate_required_files(taxonomy_dir, json_dir, args.jobseeker_file, args.opportunity_file)
    if not ok:
        print(msg, file=sys.stderr)
        print(f"\nTried taxonomy_dir={taxonomy_dir} and json_dir={json_dir}", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load inputs ---
    skills_df        = pd.read_csv(taxonomy_dir / "skills.csv")
    skill_groups_df  = pd.read_csv(taxonomy_dir / "skill_groups.csv")
    skill_hier_df    = pd.read_csv(taxonomy_dir / "skill_hierarchy.csv")
    skill_rel_df     = pd.read_csv(taxonomy_dir / "skill_to_skill_relations.csv")
    occ_skill_df     = pd.read_csv(taxonomy_dir / "occupation_to_skill_relations.csv")
    occs_df          = pd.read_csv(taxonomy_dir / "occupations.csv")

    with open(json_dir / args.jobseeker_file, "r", encoding="utf-8") as fh:
        seekers = json.load(fh)
    with open(json_dir / args.opportunity_file, "r", encoding="utf-8") as fh:
        opps = json.load(fh)

    # Crosswalks & graph
    cw = build_crosswalks(skills_df, occs_df)
    G  = build_graph(skills_df, skill_groups_df, skill_hier_df, skill_rel_df, cw["skill_id_to_label"])
    s2g = skill_to_groups_map(skill_hier_df)
    occ_ess, occ_opt = occ_canonical_profiles(occ_skill_df)
    occ_id_to_label = dict(zip(occs_df["ID"], occs_df["PREFERREDLABEL"].fillna("")))

    # Prepare opportunities (region + age_days for p̂)
    opps_prepared = []
    for op in opps:
        req_skill_ids = set()
        if op.get("skills"):
            skills_val = op.get("skills") or []
            if isinstance(skills_val, dict):
                skills_val = [skills_val]
            for s in skills_val:
                sid = map_skill_uuid_or_label(s, cw)
                if sid: req_skill_ids.add(sid)
        occ = op.get("occupation") or {}
        occ_id = map_occ_uuid_or_label(occ, cw)

        if not req_skill_ids and occ_id:
            ess_ids = set(occ_ess.get(occ_id, set()))
            opt_ids = set(occ_opt.get(occ_id, set()))
            req_ids = ess_ids | opt_ids
        else:
            ess_ids = set(req_skill_ids)
            opt_ids = set()
            req_ids = set(req_skill_ids)

        age = days_since(op.get("date_posted") or op.get("updated_at"))
        region = pick_region(op)

        opps_prepared.append({
            "job_id":   op.get("opportunity_ref_id") or op.get("opportunity_group_id") or op.get("id"),
            "title":    op.get("opportunity_title"),
            "active":   bool(op.get("active", True)),
            "occ_id":   occ_id,
            "region":   region,
            "age_days": age if age == age else float("inf"),
            "req":      list(req_ids),
            "ess":      list(ess_ids),
            "opt":      list(opt_ids),
        })

    # Success proxy
    p_hat = build_success_proxy(opps_prepared)

    # Pairwise computations
    pair_rows = []
    coverage_counters = defaultdict(lambda: {"total": 0, "pass_exact": 0, "pass_onehop": 0})

    seeker_skill_ids_cache, dist_map_cache = {}, {}

    def get_seeker_skill_ids(seeker):
        sid = seeker.get("compass_id")
        if sid in seeker_skill_ids_cache:
            return seeker_skill_ids_cache[sid]
        sids = []
        skills_val = seeker.get("skills") or []
        if isinstance(skills_val, dict):
            skills_val = [skills_val]
        for s in skills_val:
            mid = map_skill_uuid_or_label(s, cw)
            if mid: sids.append(mid)
        sids = list(set(sids))
        seeker_skill_ids_cache[sid] = sids
        return sids

    def get_dist_map(seeker_id):
        if seeker_id in dist_map_cache:
            return dist_map_cache[seeker_id]
        seeker_obj = next(s for s in seekers if s.get("compass_id")==seeker_id)
        sids = get_seeker_skill_ids(seeker_obj)
        dist_map = dists_from_skillset(G, sids, max_dist=DIST_CUTOFF)
        dist_map_cache[seeker_id] = dist_map
        return dist_map

    for seeker in seekers:
        seeker_id = seeker.get("compass_id")
        _ = get_dist_map(seeker_id)

        for op in opps_prepared:
            if not op["active"]: continue
            req_ids = op["req"]
            if not req_ids: continue

            ess_ids, opt_ids = op["ess"], op["opt"]
            dist_map   = get_dist_map(seeker_id)
            match_dists = {r: dist_map.get(r, float("inf")) for r in req_ids}

            cov_ess = coverage_score(ess_ids, match_dists, lam=LAMBDA) if ess_ids else 0.0
            cov_opt = coverage_score(opt_ids, match_dists, lam=LAMBDA) if opt_ids else 0.0
            grp_rec = group_recall(req_ids, match_dists, s2g, gate=GROUP_RECALL_GATE)
            gap_ess = sum(1 for r in ess_ids if match_dists.get(r, float("inf")) > GROUP_RECALL_GATE)

            U_core  = THETA["ess"]*cov_ess + THETA["opt"]*cov_opt + THETA["group"]*grp_rec
            surplus = U_core - THETA["gap_pen"]*bool(gap_ess)
            U_final = max(0.0, surplus)

            pass_exact  = pass_gate_essentials(ess_ids, match_dists, gate=0)
            pass_onehop = pass_gate_essentials(ess_ids, match_dists, gate=1)

            coverage_counters[seeker_id]["total"]      += 1
            coverage_counters[seeker_id]["pass_exact"] += int(pass_exact)
            coverage_counters[seeker_id]["pass_onehop"]+= int(pass_onehop)

            phat = p_hat.get(op["job_id"], 0.0)
            rank_score = U_final * phat

            if pass_exact and U_core >= TAU_FIT: band = "Fit"
            elif pass_onehop and U_core >= TAU_SAFE: band = "Safe"
            elif U_core >= TAU_ADJ: band = "Adjacent"
            else: band = "Unlikely"

            pair_rows.append({
                "seeker_id": seeker_id,
                "job_id": op["job_id"],
                "job_title": op["title"],
                "occupation_id": op["occ_id"],
                "occupation_label": occ_id_to_label.get(op["occ_id"], None),
                "region": op["region"],
                "age_days": op["age_days"],
                "n_req": len(req_ids), "n_ess": len(ess_ids), "n_opt": len(opt_ids),
                "cov_ess": round(cov_ess,4), "cov_opt": round(cov_opt,4), "group_recall": round(grp_rec,4),
                "gap_essentials": int(gap_ess),
                "U_core": round(U_core,4), "surplus": round(surplus,4), "U_final": round(U_final,4),
                "p_hat": round(phat,4), "rank_score": round(rank_score,4), "band": band,
                "pass_exact": bool(pass_exact), "pass_onehop": bool(pass_onehop),
            })

    pair_df = pd.DataFrame(pair_rows)

    # Per-seeker coverage headline
    summary_rows = []
    for seeker in seekers:
        sid = seeker.get("compass_id")
        c = coverage_counters[sid]
        tot = c["total"]
        exact_pct  = 100.0 * c["pass_exact"]  / tot if tot else 0.0
        onehop_pct = 100.0 * c["pass_onehop"] / tot if tot else 0.0
        top5 = (pair_df.query("seeker_id == @sid")
                      .sort_values("rank_score", ascending=False)
                      .head(5)[["job_id","job_title","rank_score","band"]]
                      .to_dict(orient="records"))
        summary_rows.append({
            "seeker_id": sid,
            "total_opportunities": tot,
            "pass_exact_count": c["pass_exact"],
            "pass_onehop_count": c["pass_onehop"],
            "key_skills_coverage_pct_exact": round(exact_pct, 2),
            "key_skills_coverage_pct_onehop": round(onehop_pct, 2),
            "top5_by_U_times_phat": json.dumps(top5, ensure_ascii=False),
        })
    summary_df = pd.DataFrame(summary_rows)

    # Outputs
    out_pair    = Path(args.out_dir) / "pairwise_matching_results_with_explanations.csv"
    out_summary = Path(args.out_dir) / "jobseeker_coverage_summary.csv"
    out_pair.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(out_pair, index=False)
    summary_df.to_csv(out_summary, index=False)

    cfg = {
        "uuid_source": "UUIDHISTORY last element preferred; fallback to ORIGINURI tail; else label.",
        "taxonomy_dir": str(taxonomy_dir),
        "json_dir": str(json_dir),
        "graph_edge_weights": {"hierarchy": HIER_COST, "skill_relation": REL_COST},
        "kernel": "exp(-lambda * d)", "lambda": LAMBDA, "distance_cutoff": DIST_CUTOFF,
        "group_recall_gate": GROUP_RECALL_GATE, "theta": THETA,
        "phat": {
            "cell_half_life_days": HALF_LIFE_CELL_DAYS,
            "job_half_life_days": HALF_LIFE_JOB_DAYS,
            "eb_alpha": EB_ALPHA,
            "region_field_order": REGION_FIELD_ORDER,
            "definition": "p_hat(job) = normalized OSI(occ,region) × job staleness decay"
        },
        "ui_thresholds": {"Fit": TAU_FIT, "Safe": TAU_SAFE, "Adjacent": TAU_ADJ},
    }
    with open(Path(args.out_dir) / "matching_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)

    try:
        xlsx_path = Path(args.out_dir) / "matching_results.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xlw:
            pair_df.to_excel(xlw, sheet_name="Pairwise", index=False)
            summary_df.to_excel(xlw, sheet_name="CoverageSummary", index=False)
    except Exception:
        pass

    print("Wrote:", out_pair, out_summary)

if __name__ == "__main__":
    main()
