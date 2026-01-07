
r"""
enrich_bert_with_occupation_skills.py

Purpose
-------
Merge the chosen occupation ("final_choice") from an NDJSON file into a
bert_cleaned JSON file keyed by `opportunity_group_id`. Then:
 - map the chosen occupation's preferred label -> occupation ID (from occupations.csv),
 - gather all skills related to that occupation (from occupation_to_skill_relations.csv),
 - map those skill IDs to {preferred label, description, origin UUID}, where UUID is
   the **last element of UUIDHISTORY** in skills.csv,
 - merge these skills into the existing bert_cleaned fields:

      * potential_essential_skills
      * potential_essential_skills_uuids
      * potential_essential_skills_descriptions
      * potential_optional_skills
      * potential_optional_skills_uuids
      * potential_optional_skills_descriptions

   RELATIONTYPE rules:
     * "essential" -> merge into essential triplet
     * "optional"  -> merge into optional triplet
     * blank/missing -> merge into **both** essential and optional triplets

All merges are de-duplicated (by UUID if available, else by normalized label).
If the JSON already contains essential/optional skills, they are merged and de-duplicated.

Use --only-enriched to write *only* the opportunities whose opportunity_group_id
is present in the NDJSON (i.e., "found in NDJSON"). Otherwise (default), keep all.

Usage
-----
python scripts/3_llm_reranker/3_2a_clean_LLM_occupations_add_skills.py `
  --ndjson "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\data\pre_study\llm_opportunity_responses_occupations.ndjson" `
  --json   "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\data\pre_study\bert_cleaned.json" `
  --occupations "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\Tabiya South Africa v1.0.1-rc.1\occupations.csv" `
  --relations   "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\Tabiya South Africa v1.0.1-rc.1\occupation_to_skill_relations.csv" `
  --skills      "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\Tabiya South Africa v1.0.1-rc.1\skills.csv" `
  --out         "C:\Users\jasmi\Downloads\bert_cleaned_with_occupation_skills.json" `
  #--only-enriched # DON'T USE FOR MAIN RUN; only for testing with subset NDJSON

"""


import argparse
import csv
import json
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set


# ----------------------------- Helpers: I/O ---------------------------------- #

def load_ndjson(path: Path) -> List[Dict[str, Any]]:
    """Robust NDJSON loader (one JSON object per line; ignores blank lines/comments)."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {ln} of {path}: {e}") from e
    return rows


def load_json_any(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSON file that might be:
      - a JSON array of objects,
      - a single JSON object (wrapped in {"data":[...]} or similar),
      - (fallback) newline-delimited JSON masquerading as .json.
    Returns a list of dicts (records).
    """
    text = path.read_text(encoding="utf-8").strip()
    # Try standard JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return list(obj)
        if isinstance(obj, dict):
            for key in ("data", "items", "records", "results"):
                if key in obj and isinstance(obj[key], list):
                    return list(obj[key])
            return [obj]
    except json.JSONDecodeError:
        pass

    # Fallback: treat as ndjson
    rows: List[Dict[str, Any]] = []
    for ln, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {ln} of {path}: {e}") from e
    return rows


def write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    """Pretty (but compact) JSON writer."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------------ Helpers: schema sniffing --------------------------- #

def norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def find_first_key(d: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    """Return the first present key from candidates (case-insensitive) or None."""
    if not isinstance(d, dict):
        return None
    lowered = {k.lower(): k for k in d.keys()}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def extract_opportunity_group_id(rec: Dict[str, Any]) -> Optional[str]:
    key = find_first_key(rec, [
        "opportunity_group_id", "opportunityGroupId", "group_id", "opportunity_group_uuid"
    ])
    if key:
        return str(rec.get(key))
    return None


def extract_final_choice_label(rec: Dict[str, Any]) -> Optional[str]:
    """
    Try a few common patterns:
      - rec["final_choice"] -> string OR dict with 'occupation'
      - rec["final_choice_occupation"] -> string
      - rec["occupation"] (if context implies it's the chosen one)
    """
    k = find_first_key(rec, ["final_choice", "finalChoice"])
    if k:
        val = rec.get(k)
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, dict):
            kk = find_first_key(val, ["occupation", "label", "preferred_label", "preferredLabel", "PREFERREDLABEL"])
            if kk and isinstance(val[kk], str):
                return val[kk].strip()

    k2 = find_first_key(rec, ["final_choice_occupation", "finalChoiceOccupation"])
    if k2 and isinstance(rec.get(k2), str):
        return rec[k2].strip()

    k3 = find_first_key(rec, ["occupation"])
    if k3 and isinstance(rec.get(k3), str):
        return rec[k3].strip()

    return None


def detect_column(row: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    return find_first_key(row, candidates)


def load_csv_as_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


# ---------------------- Helpers: label/ID lookups ---------------------------- #

def build_occupation_label_to_id(rows: List[Dict[str, Any]]) -> Tuple[str, str, Dict[str, str]]:
    """Return (id_col, label_col, mapping normalized_label -> occupation_id)."""
    if not rows:
        return "id", "preferred_label", {}

    sample = rows[0]
    id_col = detect_column(sample, ["id", "uuid", "occupation_id", "concept_id", "identifier"])
    label_col = detect_column(sample, ["preferred_label", "preferredLabel", "prefLabel", "label", "name", "PREFERREDLABEL"])

    if not id_col or not label_col:
        raise ValueError(
            f"Could not detect occupation ID/label columns. "
            f"Found columns: {list(sample.keys())}"
        )

    mapping: Dict[str, str] = {}
    for r in rows:
        oid = str(r.get(id_col, "")).strip()
        lab = str(r.get(label_col, "")).strip()
        if not oid or not lab:
            continue
        mapping.setdefault(norm(lab), oid)
    return id_col, label_col, mapping


def detect_skill_columns(sample: Dict[str, Any]) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Return (id_col, label_col, desc_col, uuidhist_col).
    desc_col and uuidhist_col may be None if missing.
    """
    id_col = detect_column(sample, ["id", "uuid", "skill_id", "concept_id", "identifier"])
    label_col = detect_column(sample, ["PREFERREDLABEL","preferred_label", "preferredLabel", "prefLabel", "label", "name"])
    desc_col = detect_column(sample, ["DESCRIPTION","description", "preferred_description", "PreferredDescription", "definition", "Definition", "desc"])
    uuidhist_col = detect_column(sample, ["UUIDHISTORY", "uuid_history", "uuidhistory", "UuidHistory", "uuidHistory"])
    if not id_col or not label_col:
        raise ValueError(
            f"Could not detect skill ID/label columns in skills.csv. "
            f"Found: {list(sample.keys())}"
        )
    return id_col, label_col, desc_col, uuidhist_col


_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}")


def _last_uuid_from_history(val: Any) -> Optional[str]:
    """
    Parse UUIDHISTORY and return the LAST UUID.
    Accepts JSON arrays, or delimited strings (',', ';', '|', ' ', '->', '»').
    Falls back to the last non-empty token that looks like a UUID; returns None if none found.
    """
    if val is None:
        return None

    # JSON array?
    if isinstance(val, list):
        seq = val
    else:
        s = str(val).strip()
        if not s:
            return None
        # try JSON
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                seq = parsed
            else:
                seq = None
        except Exception:
            seq = None

        if seq is None:
            # split on common delimiters
            tmp = re.split(r"[,\;\|\s>»]+", s)
            seq = [t for t in tmp if t]

    # Walk from the end to find a UUID-like token
    for token in reversed(seq):
        t = str(token).strip()
        if not t:
            continue
        # If it's like "uuid:xxxxx", take last segment
        maybe = t.split(":")[-1]
        if _UUID_RE.fullmatch(maybe):
            return maybe.lower()
        if _UUID_RE.search(maybe):
            return _UUID_RE.search(maybe).group(0).lower()
    return None


def build_skill_payload(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Build mapping: skill_id -> {label, desc, uuid_origin}
      - label from preferred_label (or similar)
      - desc from description/definition (if available)
      - uuid_origin = last element of UUIDHISTORY (parsed robustly)
    """
    if not rows:
        return {}

    id_col, label_col, desc_col, uuidhist_col = detect_skill_columns(rows[0])
    mapping: Dict[str, Dict[str, Optional[str]]] = {}
    for r in rows:
        sid = str(r.get(id_col, "")).strip()
        if not sid:
            continue
        label = str(r.get(label_col, "")).strip() or None
        desc = str(r.get(desc_col, "")).strip() if desc_col else None
        uuid_origin = _last_uuid_from_history(r.get(uuidhist_col)) if uuidhist_col else None
        mapping[sid] = {"label": label, "desc": desc, "uuid_origin": uuid_origin}
    return mapping


def detect_relations_columns(sample: Dict[str, Any]) -> Tuple[str, str, str]:
    occ_col = detect_column(sample, ["occupation_id", "occupationId", "occupation", "occ_id"])
    skl_col = detect_column(sample, ["skill_id", "skillId", "skill", "skl_id"])
    rel_col = detect_column(sample, ["RELATIONTYPE", "relationtype", "relation_type", "RelationType", "type"])

    if not occ_col or not skl_col:
        raise ValueError(
            f"Could not detect columns in occupation_to_skill_relations.csv. "
            f"Found: {list(sample.keys())}"
        )
    if not rel_col:
        rel_col = "__missing_relationtype__"

    return occ_col, skl_col, rel_col


# ---------------------- Helpers: merging into triplets ----------------------- #

def _collect_existing_triplet(rec: Dict[str, Any], base: str) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Read existing triplet fields and produce a dict:
      key -> {label, uuid, desc}

    key preference:
      1) uuid (lowercased) if present
      2) normalized label

    base is one of:
      - "potential_essential_skills"
      - "potential_optional_skills"
    """
    labels = rec.get(base)
    uuids = rec.get(f"{base}_uuids")
    descs = rec.get(f"{base}_descriptions")

    if not isinstance(labels, list):
        labels = []
    if not isinstance(uuids, list):
        uuids = []
    if not isinstance(descs, list):
        descs = []

    n = max(len(labels), len(uuids), len(descs))
    # extend shorter lists
    def _get(lst, i):
        return lst[i] if i < len(lst) else None

    out: Dict[str, Dict[str, Optional[str]]] = {}
    for i in range(n):
        lab = _get(labels, i)
        uid = _get(uuids, i)
        des = _get(descs, i)
        lab_s = str(lab).strip() if lab not in (None, "") else None
        uid_s = str(uid).strip().lower() if uid not in (None, "") else None
        des_s = str(des).strip() if des not in (None, "") else None

        key = uid_s if uid_s else (norm(lab_s) if lab_s else None)
        if not key:
            continue
        # keep first occurrence; fill missing fields if later entries have them
        if key not in out:
            out[key] = {"label": lab_s, "uuid": uid_s, "desc": des_s}
        else:
            if not out[key]["label"] and lab_s:
                out[key]["label"] = lab_s
            if not out[key]["uuid"] and uid_s:
                out[key]["uuid"] = uid_s
            if not out[key]["desc"] and des_s:
                out[key]["desc"] = des_s
    return out


def _merge_items(existing: Dict[str, Dict[str, Optional[str]]],
                 new_items: List[Dict[str, Optional[str]]]) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Merge new_items (each with 'label','uuid','desc') into existing (dict keyed by uuid/label).
    Deduplicate by uuid (preferred) else by normalized label.
    """
    for it in new_items:
        lab = it.get("label")
        uid = it.get("uuid")
        des = it.get("desc")
        key = (uid or "").lower().strip() if uid else (norm(lab) if lab else None)
        if not key:
            continue
        if key not in existing:
            existing[key] = {"label": lab, "uuid": (uid.lower().strip() if uid else None), "desc": des}
        else:
            if not existing[key]["label"] and lab:
                existing[key]["label"] = lab
            if not existing[key]["uuid"] and uid:
                existing[key]["uuid"] = uid.lower().strip()
            if not existing[key]["desc"] and des:
                existing[key]["desc"] = des
    return existing


def _emit_triplet_lists(merged: Dict[str, Dict[str, Optional[str]]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Turn merged dict back into parallel lists, sorted for stability by:
      (label_lower, uuid or "")
    """
    items = list(merged.values())
    items.sort(key=lambda d: (norm(d.get("label")), d.get("uuid") or ""))

    labels = [d.get("label") or "" for d in items]
    uuids = [d.get("uuid") or "" for d in items]
    descs = [d.get("desc") or "" for d in items]
    return labels, uuids, descs


# ------------------------------- Main logic --------------------------------- #

def enrich(
    ndjson_path: Path,
    json_path: Path,
    occupations_csv: Path,
    relations_csv: Path,
    skills_csv: Path,
    out_path: Path,
    only_enriched: bool = False,
) -> None:

    # 1) Load sources
    nd_rows = load_ndjson(ndjson_path)
    bert_rows = load_json_any(json_path)
    occ_rows = load_csv_as_rows(occupations_csv)
    rel_rows = load_csv_as_rows(relations_csv)
    skl_rows = load_csv_as_rows(skills_csv)

    # Extract the full set of GIDs present in NDJSON
    gids_in_ndjson: Set[str] = set()
    for r in nd_rows:
        gid = extract_opportunity_group_id(r)
        if gid:
            gids_in_ndjson.add(gid)

    # 2) Build lookups
    _, _, occ_label_to_id = build_occupation_label_to_id(occ_rows)
    skill_payload = build_skill_payload(skl_rows)  # skill_id -> {label, desc, uuid_origin}

    if rel_rows:
        occ_col, skl_col, rel_col = detect_relations_columns(rel_rows[0])
    else:
        raise ValueError("occupation_to_skill_relations.csv appears to be empty.")

    # Map: opportunity_group_id -> final_choice_label (only when label present)
    oppid_to_choice: Dict[str, str] = {}
    for r in nd_rows:
        gid = extract_opportunity_group_id(r)
        lab = extract_final_choice_label(r)
        if gid and lab:
            oppid_to_choice.setdefault(gid, lab)

    # Index relations: occupation_id -> list[(skill_id, relationtype)]
    rel_index: Dict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
    for r in rel_rows:
        occ_id = str(r.get(occ_col, "")).strip()
        skl_id = str(r.get(skl_col, "")).strip()
        rel_val = None
        if rel_col != "__missing_relationtype__":
            rel_val = str(r.get(rel_col)).strip() if r.get(rel_col) not in (None, "") else None
        if occ_id and skl_id:
            rel_index[occ_id].append((skl_id, rel_val))

    # Counters for summary
    n_with_choice = 0
    n_occ_id_found = 0
    n_added_ess = 0
    n_added_opt = 0

    # 3) Enrich bert rows
    for rec in bert_rows:
        gid = extract_opportunity_group_id(rec)
        if not gid:
            continue

        choice_label = oppid_to_choice.get(gid)
        if not choice_label:
            # No final_choice known for this record
            continue

        # Merge the final choice label in a stable field (optional, but useful)
        rec["final_choice_occupation"] = choice_label
        n_with_choice += 1

        # Map occupation label -> occupation_id
        occ_id = occ_label_to_id.get(norm(choice_label))
        if not occ_id:
            # Cannot map this occupation; skip skill derivation
            continue
        n_occ_id_found += 1

        # Gather skill IDs + relation types for this occupation
        pairs = rel_index.get(occ_id, [])

        # Build new items to merge per relation bucket
        new_ess: List[Dict[str, Optional[str]]] = []
        new_opt: List[Dict[str, Optional[str]]] = []

        for sid, reltype in pairs:
            payload = skill_payload.get(sid, {})
            item = {
                "label": payload.get("label"),
                "uuid": payload.get("uuid_origin"),
                "desc": payload.get("desc"),
            }
            reln = norm(reltype) if reltype else ""
            if reln == "essential":
                new_ess.append(item)
            elif reln == "optional":
                new_opt.append(item)
            else:
                # Blank/missing -> both
                new_ess.append(item)
                new_opt.append(item)

        # Merge into existing triplets (essential, optional)
        ess_base = "potential_essential_skills"
        opt_base = "potential_optional_skills"

        existing_ess = _collect_existing_triplet(rec, ess_base)
        existing_opt = _collect_existing_triplet(rec, opt_base)

        before_ess = len(existing_ess)
        before_opt = len(existing_opt)

        merged_ess = _merge_items(existing_ess, new_ess)
        merged_opt = _merge_items(existing_opt, new_opt)

        # Write back as parallel lists (and count additions)
        labs, uids, descs = _emit_triplet_lists(merged_ess)
        rec[ess_base] = labs
        rec[f"{ess_base}_uuids"] = uids
        rec[f"{ess_base}_descriptions"] = descs

        labs, uids, descs = _emit_triplet_lists(merged_opt)
        rec[opt_base] = labs
        rec[f"{opt_base}_uuids"] = uids
        rec[f"{opt_base}_descriptions"] = descs

        n_added_ess += max(0, len(merged_ess) - before_ess)
        n_added_opt += max(0, len(merged_opt) - before_opt)

    # 4) Choose output subset
    if only_enriched:
        out_rows = [
            rec for rec in bert_rows
            if (extract_opportunity_group_id(rec) or "") in gids_in_ndjson
        ]
    else:
        out_rows = bert_rows

    # 5) Write output
    write_json(out_path, out_rows)

    # 6) Print concise summary to stderr
    print(
        f"[done] wrote: {out_path}\n"
        f"  NDJSON opportunities found (unique gids): {len(gids_in_ndjson)}\n"
        f"  records with final_choice merged: {n_with_choice}\n"
        f"  records where occupation ID was found: {n_occ_id_found}\n"
        f"  skills added into essential triplet: {n_added_ess}\n"
        f"  skills added into optional triplet: {n_added_opt}\n"
        f"  output records written: {len(out_rows)}",
        file=sys.stderr
    )


def main():
    p = argparse.ArgumentParser(description="Enrich bert_cleaned JSON with chosen occupation and its skills.")
    p.add_argument("--ndjson", type=Path, required=False,
                   default=Path("/mnt/data/llm_opportunity_responses_occupations_excerpt.ndjson"),
                   help="NDJSON with final_choice per opportunity_group_id.")
    p.add_argument("--json", type=Path, required=False,
                   default=Path("/mnt/data/bert_cleaned_subset6.json"),
                   help="bert_cleaned JSON to enrich.")
    p.add_argument("--occupations", type=Path, required=False,
                   default=Path("/mnt/data/occupations.csv"),
                   help="CSV mapping occupation IDs to preferred labels.")
    p.add_argument("--relations", type=Path, required=False,
                   default=Path("/mnt/data/occupation_to_skill_relations.csv"),
                   help="CSV linking occupation_id to skill_id with RELATIONTYPE.")
    p.add_argument("--skills", type=Path, required=False,
                   default=Path("/mnt/data/skills.csv"),
                   help="CSV mapping skill IDs to preferred labels, descriptions, and UUIDHISTORY.")
    p.add_argument("--out", type=Path, required=False,
                   default=Path("/mnt/data/bert_cleaned_enriched.json"),
                   help="Output JSON path.")
    p.add_argument("--only-enriched", action="store_true",
                   help="If set, write only opportunities whose opportunity_group_id is present in the NDJSON. "
                        "Otherwise, keep all original JSON records (default).")
    args = p.parse_args()

    enrich(
        ndjson_path=args.ndjson,
        json_path=args.json,
        occupations_csv=args.occupations,
        relations_csv=args.relations,
        skills_csv=args.skills,
        out_path=args.out,
        only_enriched=args.only_enriched,
    )


if __name__ == "__main__":
    main()
