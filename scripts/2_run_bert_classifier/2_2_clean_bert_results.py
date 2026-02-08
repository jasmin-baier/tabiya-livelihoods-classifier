# NOTE: It is important that this script renames extracted_occupations and extracted_skills1 to "potential_occupations" and "potential_skills" for LLM file to work

"""
2_2_clean_bert_results.py (improved)

Improvements implemented (Nov 3, 2025):

- Normalize UUID case everywhere (lower-case on input, maps, and filters).
- Robust CSV reads as strings (`dtype=str`, `keep_default_na=False`) to avoid NaN coercion.
- Safer row access using `.get()` with sensible defaults; skip rows with missing essential IDs.
- `parse_extracted_items` now handles both single- and double-quoted "retrieved" payloads, plus list/dict inputs.
- Group mapping dedupes by *parent ID* (not label/description) to keep label/description in sync.
- Paths are configurable via argparse (no hard-coded Windows paths). Sensible defaults can be provided.
- Logging (with --verbose) instead of print; optional tqdm progress if installed.
- Append/skip mode: if `--append` and output exists, skip already processed opportunities
  (keyed by (opportunity_group_id, opportunity_ref_id)) and only process new rows.
- Deterministic ordering: UUID lists and group ID sets are sorted before mapping to labels/descriptions.
- Hierarchy loader hardens to mixed-case/space headers by normalizing column names.
- `build_identifier_map` lower-cases detected UUID-like tokens; adds the primary ID as a key and, if UUID-like,
  also its lower-cased form.

  How to run
  python scripts/2_run_bert_classifier/2_2_clean_bert_results.py `
  --taxonomy-dir "C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/Tabiya South Africa v1.0.1-rc.1" `
  --input-csv "C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study/BERT_extracted_occupations_skills_uuid.csv" `
  --output-json "C:/Users/jasmi/OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass/data/pre_study/bert_cleaned.json" `
  --append `
  -v


use --append to skip already processed opportunities in the output file (optional and not necessary on first run)
Add -v for verbose logs.
tqdm is optional for progress bars

"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# tqdm is optional
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore
    

# TODO check new names from line 410, change in LLM file
# TODO: still consider adding all skills from occupation --> has to happen BEFORE the LLM step. This can be an alternative opp_db, similar how the skills_group version is an alternative opp_db
# TODO: Now still using extracted_skills1, check if requirements are just a subset of skills 1; if yes OR more comprehensive, use extracted_skills2 & skill_requirements separately
# TODO I have currently commented out all extra fields, in two places (e.g. date_posted etc) --> change once I decide how to pull these fields through the whole pipeline
# TODO: opportunity DB should also have: when posted/when ends + and indicator if at date = today it is still relevant; or if it has been deleted (then also not relevant) --> for study consider keeping all relevant all the time to have larger number of jobs / have function that ensures there are at least 1000 jobs to compare to
# TODO: opportunity_ref_ids are not unique, I always need both opportunity_group_id and opportunity_ref_id
# TODO output doesn't yet actually have all the columns I want

# PRIORITY
# TODO: NOTE I need to make sure to actually just the original id; ideally I want to keep both though; but matching should use the origin; or have only origin_id but then have metadata information which taxonomy was used [for future] --> metadata should be: model/taxonomy id, uuid of the model
# NOTE push for compass to actually use south african taxonomy for main study

# NOTE  that BERT taxonomy used oldest esco uuids (oldest in Compass uuid history)

# TODO think about qualifications and opportunity_requirements
# Current judgement: Probably stay away from it, and only talk about skills matches, clearly state that it doesn't take qualifications into account
# Considerations if I did want to include qualifications
# if it mentions matric, manually add it as requirement, bert likely won't understand
# I have to map requirements to qualifications, but our taxonomy only has skills and occupations? Couldn't find secondary school certificate for example
# How can I tell the system that if someone has upper secondary qualification, they also have all of the qualifications below
# possibly have to add South African qulifications manually, similar to matric bit in; opportunity_requirements can get quite complicated though (include employment status, criminal record, driver's license, own car, matric only a few months go etc.)
# South African NQF doesn't perfectly map onto EQF, since EQF only has 8 levels: https://www.saqa.org.za/wp-content/uploads/2023/02/National-Qualifications-Framework.pdf

# NOTE: It is important that this script renames extracted_occupations and extracted_skills1 to "potential_occupations" and "potential_skills" for LLM file to work


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

UUID_PATTERN = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', re.I
)

def norm_uuid(s: Any) -> str:
    """Normalize a UUID-like string: strip and lower. Non-strings -> ''."""
    if not isinstance(s, str):
        return ""
    return s.strip().lower()

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    # keep pandas/quieter libs calmer in DEBUG
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Taxonomy loaders
# ---------------------------------------------------------------------------

def build_identifier_map(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Build a dictionary mapping from any seen identifier token (UUIDs found in any cell,
    plus values from a primary ID column if present) to the full record (row).
    - UUID-like tokens are lower-cased.
    - The 'primary id' column (ID or uuid) is also added as a key; if it looks like
      a UUID, a lower-cased alias is added as well.
    """
    if not csv_path.exists():
        logging.warning("File not found: %s", csv_path)
        return {}

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[])
    logging.info("Processing identifiers from '%s' (%d rows)...", csv_path.name, len(df))

    possible_id_cols = ["ID", "uuid", "Uuid", "UUID"]
    id_col_name = next((col for col in possible_id_cols if col in df.columns), None)

    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        record = row.to_dict()
        # 1) Collect UUID-like tokens found anywhere in the row
        for cell_value in row.values:
            if isinstance(cell_value, str) and cell_value:
                for u in UUID_PATTERN.findall(cell_value):
                    mapping[norm_uuid(u)] = record

        # 2) Also add the primary id value as a key (required for group_map lookups by numeric ID)
        if id_col_name:
            primary_id = str(row.get(id_col_name, "")).strip()
            if primary_id:
                mapping[primary_id] = record  # keep as-is (IDs may be numeric-like strings)
                if UUID_PATTERN.fullmatch(primary_id):
                    mapping[norm_uuid(primary_id)] = record  # UUID alias

    return mapping


def build_hierarchy_map(csv_path: Path) -> Dict[str, str]:
    """
    Build child->parent map from hierarchy CSV.
    Header names are normalized to uppercase+stripped to be resilient.
    """
    if not csv_path.exists():
        logging.warning("Hierarchy file not found: %s", csv_path)
        return {}

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[])
    # normalize headers
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "CHILDID" not in df.columns or "PARENTID" not in df.columns:
        logging.warning("Missing CHILDID/PARENTID columns in %s", csv_path.name)
        return {}

    df = df[(df["CHILDID"].astype(str) != "") & (df["PARENTID"].astype(str) != "")]
    hier_map: Dict[str, str] = pd.Series(df["PARENTID"].values, index=df["CHILDID"]).to_dict()
    logging.info("Loaded hierarchy from '%s' with %d relations.", csv_path.name, len(hier_map))
    return hier_map

# ---------------------------------------------------------------------------
# Parsing helpers (for the BERT output cells)
# ---------------------------------------------------------------------------

def _safe_load_literal(cell: Any) -> Any:
    """
    Try JSON first, then Python literal (list/dict), otherwise return the raw cell.
    """
    if not isinstance(cell, str):
        return cell
    text = cell.strip()
    if not text:
        return []
    # JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Python literal
    try:
        import ast
        return ast.literal_eval(text)
    except Exception:
        return text


def _extract_retrieved_list(parsed_item: Any) -> List[str]:
    """
    From a parsed list element that could be a dict with key 'retrieved' (string or list),
    or a plain string, return a list of retrieved tokens.
    """
    out: List[str] = []
    if isinstance(parsed_item, dict):
        if "retrieved" in parsed_item:
            val = parsed_item["retrieved"]
            if isinstance(val, str):
                val = val.strip()
                if val:
                    out.append(val)
            elif isinstance(val, list):
                out.extend([str(v).strip() for v in val if str(v).strip()])
    elif isinstance(parsed_item, str):
        if parsed_item.strip():
            out.append(parsed_item.strip())
    return out


def parse_extracted_items(cell: Any) -> List[str]:
    """
    Parses a cell that may contain:
      - JSON or Python list of dicts/strings (expected), or
      - A single string that embeds `'retrieved': '...'` or `"retrieved": "..."`.
    Returns a de-duplicated, order-preserving list of strings.
    """
    parsed = _safe_load_literal(cell)

    # Case 1: string with embedded retrieved="..."
    if isinstance(parsed, str):
        out: List[str] = []
        # support both single- and double-quoted keys/values
        for pat in (r"'retrieved'\s*:\s*'([^']+)'", r'"retrieved"\s*:\s*"([^"]+)"'):
            out.extend(re.findall(pat, parsed))
        # dedupe & normalize whitespace, preserve order
        seen, final = set(), []
        for s in (x.strip() for x in out if str(x).strip()):
            if s not in seen:
                seen.add(s)
                final.append(s)
        return final

    # Case 2: parsed list/dict structure
    out: List[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            out.extend(_extract_retrieved_list(item))
    elif isinstance(parsed, dict):
        out.extend(_extract_retrieved_list(parsed))

    # dedupe while preserving order
    seen, final = set(), []
    for s in (x.strip() for x in out if str(x).strip()):
        if s not in seen:
            seen.add(s)
            final.append(s)
    return final

# ---------------------------------------------------------------------------
# Group mapping helper
# ---------------------------------------------------------------------------

def _get_group_info(
    item_uuids: Sequence[str],
    item_map: Dict[str, Dict[str, Any]],
    hierarchy_map: Dict[str, str],
    group_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """
    Resolve a set of *parent group IDs* for the given item UUIDs, then map to
    synchronized (labels, descriptions). Dedup is done on parent IDs, not text.
    """
    parent_ids: set[str] = set()

    for raw_uuid in item_uuids:
        rec = item_map.get(norm_uuid(raw_uuid)) or item_map.get(str(raw_uuid))
        if not rec:
            continue
        current_id = str(rec.get("ID", "")).strip()
        if not current_id:
            continue
        parent_id = hierarchy_map.get(current_id)
        if parent_id:
            parent_ids.add(str(parent_id).strip())

    if not parent_ids:
        return [], []

    labels: List[str] = []
    descriptions: List[str] = []

    # deterministic order
    for pid in sorted(parent_ids):
        g = group_map.get(pid) or {}
        label = str(g.get("PREFERREDLABEL", "")).strip()
        desc = str(g.get("DESCRIPTION", "")).strip()
        if label:
            labels.append(label)
            descriptions.append(desc)

    return labels, descriptions

# ---------------------------------------------------------------------------
# Main transformation
# ---------------------------------------------------------------------------

def transform_csv_to_job_data(
    csv_file_path: Path,
    output_json_path: Path,
    occ_map: Dict[str, Dict[str, Any]],
    skill_map: Dict[str, Dict[str, Any]],
    occ_hierarchy_map: Dict[str, str],
    skill_hierarchy_map: Dict[str, str],
    occ_group_map: Dict[str, Dict[str, Any]],
    skill_group_map: Dict[str, Dict[str, Any]],
    append: bool = False,
) -> List[Dict[str, Any]]:
    """
    Transform the raw BERT output CSV into clean JSON, enriching with labels,
    descriptions, and group mappings from taxonomy maps.
    """
    logging.info("Reading BERT CSV: %s", csv_file_path)
    try:
        df = pd.read_csv(csv_file_path, dtype=str, keep_default_na=False, na_values=[])
    except Exception as e:
        logging.error("Failed to read CSV: %s", e)
        return []

    # --- Column renames ---
    df.rename(
        columns={
            "extracted_occupation_from_title": "potential_occupations_title_raw",
            "extracted_occupation_from_full_details": "potential_occupations_fulldescription_raw",
            "extracted_optional_skills": "potential_skills_optional_raw",
            "extracted_essential_skills": "potential_skill_essential_raw",
        },
        inplace=True,
    )

    # Build sets of valid UUIDs (lower-cased) for efficient filtering
    valid_occupation_uuids = {k.lower() for k in occ_map.keys() if UUID_PATTERN.fullmatch(str(k))}
    valid_skill_uuids = {k.lower() for k in skill_map.keys() if UUID_PATTERN.fullmatch(str(k))}

    # Append/skip mode: read existing JSON to skip already processed
    processed_keys: set[Tuple[str, str]] = set()
    existing: List[Dict[str, Any]] = []
    if append and output_json_path.exists():
        try:
            existing = json.loads(output_json_path.read_text(encoding="utf-8"))
            for d in existing:
                k = (str(d.get("opportunity_group_id", "")), str(d.get("opportunity_ref_id", "")))
                processed_keys.add(k)
            logging.info("Append mode: loaded %d existing entries, %d keys.", len(existing), len(processed_keys))
        except Exception as e:
            logging.warning("Append mode: could not load existing JSON (%s). Proceeding fresh.", e)

    def iter_rows(it):
        if tqdm is None:
            return it
        return tqdm(it, total=len(df), disable=len(df) < 2000, desc="Processing rows")

    job_data_list: List[Dict[str, Any]] = []

    for _, row in iter_rows(df.iterrows()):
        try:
            ogid = str(row.get("opportunity_group_id", "")).strip()
            orid = str(row.get("opportunity_ref_id", "")).strip()
            if not ogid or not orid:
                logging.debug("Skipping row with missing IDs (group_id=%r, ref_id=%r)", ogid, orid)
                continue

            key = (ogid, orid)
            if key in processed_keys:
                continue  # skip already processed

            # 1) Parse raw cells
            occ_title_raw = parse_extracted_items(row.get("potential_occupations_title_raw"))
            occ_full_raw  = parse_extracted_items(row.get("potential_occupations_fulldescription_raw"))
            skl_raw       = parse_extracted_items(row.get("potential_skills_optional_raw"))
            req_raw       = parse_extracted_items(row.get("potential_skill_essential_raw"))

            # 2) Combine & de-duplicate occupations (title first, then full description)
            combined_occ_raw = []
            _seen = set()
            for s in (occ_title_raw + occ_full_raw):
                u = norm_uuid(s)
                if u and u not in _seen:
                    _seen.add(u)
                    combined_occ_raw.append(u)

            # 3) Filter against valid sets
            #    (If you prefer reproducible output regardless of input order, uncomment the "sorted" version.)
            occupations = [u for u in combined_occ_raw if u in valid_occupation_uuids]
            # occupations = sorted(set(occupations))  # <- deterministic, order-agnostic alternative

            skills = sorted({u for u in (norm_uuid(x) for x in skl_raw) if u in valid_skill_uuids})
            skill_requirements = sorted({u for u in (norm_uuid(x) for x in req_raw) if u in valid_skill_uuids})

            # 4) Map UUIDs -> Labels/Descriptions (aligned to "occupations")
            occupation_labels = [str(occ_map.get(u, {}).get("PREFERREDLABEL", f"UNKNOWN_{u}")).strip() for u in occupations]
            occupation_descriptions = [str(occ_map.get(u, {}).get("DESCRIPTION", "")).strip() for u in occupations]

            skill_labels = [str(skill_map.get(u, {}).get("PREFERREDLABEL", f"UNKNOWN_{u}")).strip() for u in skills]
            skill_descriptions = [str(skill_map.get(u, {}).get("DESCRIPTION", "")).strip() for u in skills]

            skill_requirements_labels = [str(skill_map.get(u, {}).get("PREFERREDLABEL", f"UNKNOWN_{u}")).strip() for u in skill_requirements]
            skill_requirements_descriptions = [str(skill_map.get(u, {}).get("DESCRIPTION", "")).strip() for u in skill_requirements]

            # 5) Group info (by parent IDs -> labels/descriptions in deterministic order)
            occ_groups, occ_group_descs = _get_group_info(occupations, occ_map, occ_hierarchy_map, occ_group_map)
            skill_groups, skill_group_descs = _get_group_info(skills, skill_map, skill_hierarchy_map, skill_group_map)
            req_groups, req_group_descs = _get_group_info(skill_requirements, skill_map, skill_hierarchy_map, skill_group_map)

            # 6) Assemble final entry (note: no *_full occupation fields anymore)
            job_entry = {
                "opportunity_group_id": ogid,
                "opportunity_ref_id": orid,
                "opportunity_title": str(row.get("opportunity_title", "")).strip(),
                "opportunity_description": str(row.get("opportunity_description", "")).strip(),
                "opportunity_requirements": str(row.get("opportunity_requirements", "")).strip(),
                "full_details": str(row.get("full_details", "")).strip(),

                "potential_occupations_uuids": occupations,
                "potential_occupations": occupation_labels,
                "potential_occupations_descriptions": occupation_descriptions,
                "potential_occupation_groups": occ_groups,
                "potential_occupation_group_descriptions": occ_group_descs,

                "potential_optional_skills_uuids": skills,
                "potential_optional_skills": skill_labels,
                "potential_optional_skills_descriptions": skill_descriptions,
                "potential_optional_skill_groups": skill_groups,
                "potential_optional_skill_group_descriptions": skill_group_descs,

                "potential_essential_skills_uuids": skill_requirements,
                "potential_essential_skills": skill_requirements_labels,
                "potential_essential_skills_descriptions": skill_requirements_descriptions,
                "potential_essential_skill_groups": req_groups,
                "potential_essential_skill_group_descriptions": req_group_descs,
            }

            job_data_list.append(job_entry)


        except Exception as e:
            logging.exception("Failed to process a row: %s", e)
            continue

    # Combine with existing if append mode
    final_data = (existing + job_data_list) if (append and existing) else job_data_list

    # Save JSON
    try:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        logging.info("Saved %d entries -> %s", len(final_data), output_json_path)
    except Exception as e:
        logging.error("Error saving JSON: %s", e)

    return final_data

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean BERT results and enrich with taxonomy info.")
    p.add_argument("--taxonomy-dir", type=Path, required=True,
                   help="Directory containing taxonomy CSVs: skills.csv, occupations.csv, "
                        "skill_hierarchy.csv, skill_groups.csv, occupation_hierarchy.csv, occupation_groups.csv")
    p.add_argument("--input-csv", type=Path, required=True,
                   help="Path to BERT_extracted_occupations_skills_uuid.csv (or similar).")
    p.add_argument("--output-json", type=Path, required=True,
                   help="Where to write the cleaned JSON.")
    p.add_argument("--append", action="store_true",
                   help="If set, load existing JSON and skip already-processed opportunity IDs.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    tax_dir = args.taxonomy_dir
    # Resolve taxonomy file paths
    SKILLS_FULL_ENTITY_PATH = tax_dir / "skills.csv"
    OCCUPATIONS_FULL_ENTITY_PATH = tax_dir / "occupations.csv"
    SKILLHIERARCHY_PATH = tax_dir / "skill_hierarchy.csv"
    SKILLGROUP_PATH = tax_dir / "skill_groups.csv"
    OCCHIERARCHY_PATH = tax_dir / "occupation_hierarchy.csv"
    OCCGROUP_PATH = tax_dir / "occupation_groups.csv"

    # Build maps
    logging.info("[1/3] Building identifier maps...")
    skill_map = build_identifier_map(SKILLS_FULL_ENTITY_PATH)
    occ_map = build_identifier_map(OCCUPATIONS_FULL_ENTITY_PATH)
    skill_group_map = build_identifier_map(SKILLGROUP_PATH)
    occ_group_map = build_identifier_map(OCCGROUP_PATH)

    logging.info("[2/3] Building hierarchy maps...")
    skill_hierarchy_map = build_hierarchy_map(SKILLHIERARCHY_PATH)
    occ_hierarchy_map = build_hierarchy_map(OCCHIERARCHY_PATH)

    logging.info("Loaded %d skill IDs and %d occupation IDs.", len(skill_map), len(occ_map))

    # Transform
    logging.info("[3/3] Transforming CSV -> JSON...")
    transform_csv_to_job_data(
        csv_file_path=args.input_csv,
        output_json_path=args.output_json,
        occ_map=occ_map,
        skill_map=skill_map,
        occ_hierarchy_map=occ_hierarchy_map,
        skill_hierarchy_map=skill_hierarchy_map,
        occ_group_map=occ_group_map,
        skill_group_map=skill_group_map,
        append=args.append,
    )

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
