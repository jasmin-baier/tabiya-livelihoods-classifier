
# TODO IMPORTANT: Have separate variables for uuid and originUUID
# TODO Only keep final chosen occupation
# TODO Also merge back on thos jobs that had 0 occupations/skills from bert (if at least one of them was non-zero it should be in LLM files)
# TODO Make sure to merge other job variables back in
# TODO map back to uuids --> but the ones from tabiya taxonomy (in future make robust to changes in ids somehow)
# TODO bring into correct structure as per Miro
# TODO Duplicates: omg just pass the URL through as primary ID — but no even fewer unique one's here...have to check what is going on
# TODO maybe in the full RCT keep "required" and "important" skills separate
# TODO for now created_at and updated_at are just today's date. Make script more sophisticated by recognizing which opportunities we are actually adding/updating and which already existed.
# TODO consider filtering out opportunities that are skills trainings; step 7 has a first approach
# TODO: Some of the opportunities seem hallucinated, as this file cannot find uuids for them. Will probably have to do some re-runs here as well, similar to how I did them for skills.
# TODO: This file now set all active = True, but code exists and is just commented out to actually check the status of the opportunity

# NOTE: In the main function "restructure_job_data" I can set remove_null_skill_uuids = True or False, depending on whether I am running this to find jobs for my bert-rerun or to share a final database with the tech team.


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
3_2b_clean_LLM_create_opp-db.py
---------------------------------------------------------------------
Builds a consolidated opportunity database with a strict separation
between essential and optional skills.

Improvements implemented:
  1) Join keys exclude the volatile title (use stable IDs only)
  2) Flexible readers: NDJSON streaming + streaming JSON-array reader,
     tolerant to minor format issues
  3) Atomic writes with fsync and safe replace (with retries)
  4) Structured logging (no bare prints)
  5) Unmapped-skills audit CSV
  6) Normalized label mapping (trim/casefold/NFKC), but keep originals
  7) Strict disjointness: if a skill is both essential and optional,
     keep it only in essential
  8) Deterministic ordering of skills
  9) Safer active flag using posted/closing dates (UTC-aware)
 10) CLI for paths and switches (no hardcoded OneDrive paths)
 11) Provenance metadata sidecar file (schema_version, sources, generated_at)
 12) Lightweight validation before write
 16) Safer date parsing & null handling (utc=True, errors='coerce')
 17) Reusable mapping helpers (unit-testable)

  - The upstream 3_1 outputs ONLY:
      * "essential_skills"
      * "optional_skills"

  - For skills and occupations, each mapped item carries TWO UUIDs:
      * "uuid":        FIRST element of UUIDHISTORY
      * "originUUID":  LAST element of UUIDHISTORY

  - After building the main dataframe, compute TOP-LEVEL SKILL GROUPS
    for essential and optional skill buckets by walking the hierarchy
    in `skill_hierarchy.csv`, resolving group labels + both UUIDs from
    `skill_groups.csv`, and bridging skills via `skills.csv` UUIDHISTORY.

    WHAT'S NEW (2025-11-10):
  • Treat the ESCO skill hierarchy as a TRUE DAG (multiple parents).
  • For each skill:
      - Walk ALL parent paths (DFS).
      - Build ALL skill_group chains (top → … → deep) along those paths.
      - Select the requested level on each chain.
      - Deduplicate groups and attach them.
  • `--skillgroup-level` accepts:
      - Integer ≥ 1: 1 = top/root, 2 = penultimate, 3 = third-from-top, …
      - Aliases: "top" (1), "penultimate" (2), "deepest" (last)
  • The selected groups are written to:
        essential_skillgroups
        optional_skillgroups
    (These names are kept for backward-compatibility; they now mean
     "selected level(s)" rather than strictly the root/top.)
---------------------------------------------------------------------
Output:
  - Main: JSON array of opportunity records
  - Sidecar: <out>.meta.json with provenance & schema version

Notes:
  * Accepts JSON arrays or NDJSON for LLM outputs.
  * For huge JSON arrays, a streaming array reader is included.

---------------------------------------------------------------------
USAGE EXAMPLES:

python scripts/3_llm_reranker/3_2b_clean_LLM_create_opp-db.py `
  --occupations "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\data\pre_study\llm_opportunity_responses_occupations.compact.json" `
  --skills-essential "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\data\pre_study\llm_opportunity_responses_skills_essential.compact.json" `
  --skills-optional  "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\data\pre_study\llm_opportunity_responses_skills_optional.compact.json" `
  --skills-entities  "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\Tabiya South Africa v1.0.1-rc.1\skills.csv" `
  --skill-groups-entities "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\Tabiya South Africa v1.0.1-rc.1\skill_groups.csv" `
  --skill-hierarchy  "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\Tabiya South Africa v1.0.1-rc.1\skill_hierarchy.csv" `
  --occ-entities     "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\Tabiya South Africa v1.0.1-rc.1\occupations.csv" `
  --extra            "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\data\pre_study\harambee_jobs_clean_without_duplicates.csv" `
  --out              "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\data\pre_study\2026-01-03_opportunity_database.json" `
  --unmapped-audit-out "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\data\pre_study\unmapped_skills_audit.csv" `
  --skillgroup-level 2 `
  --drop-error-rows `
  --log-level INFO `
  --remove-null-skill-uuids

"""


from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Set

import pandas as pd

SCHEMA_VERSION = "2025-11-10.DAG.1"
ID_COLUMNS = ["opportunity_group_id", "opportunity_ref_id"]


# -----------------------------
# Logging
# -----------------------------
def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


# -----------------------------
# Flexible readers
# -----------------------------
def _iter_ndjson(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
                else:
                    logging.warning("NDJSON line %d is not a JSON object; skipping.", i)
            except Exception as e:
                logging.warning("NDJSON parse error on line %d: %s (skipped)", i, e)


def _iter_json_array_stream(path: Path) -> Iterator[Dict[str, Any]]:
    dec = json.JSONDecoder()
    buf = ""
    in_array = False
    idx = 0
    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(65536)
            if not chunk and not buf:
                break
            buf += chunk

            if not in_array:
                while idx < len(buf) and buf[idx].isspace():
                    idx += 1
                if idx < len(buf) and buf[idx] == "[":
                    in_array = True
                    idx += 1
                else:
                    if not chunk:
                        raise ValueError("Not a JSON array (missing '[').")
                    continue

            while True:
                while idx < len(buf) and (buf[idx].isspace() or buf[idx] == ","):
                    idx += 1
                if idx < len(buf) and buf[idx] == "]":
                    return
                if idx >= len(buf):
                    break
                try:
                    obj, end = dec.raw_decode(buf, idx)
                except json.JSONDecodeError:
                    break
                else:
                    idx = end
                    if isinstance(obj, dict):
                        yield obj
                    else:
                        logging.warning("Array item is not an object; skipping.")
            if idx > 0:
                buf = buf[idx:]
                idx = 0
            if not chunk and not buf:
                break


def _iter_records_flex(path: Path) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".ndjson", ".jsonl"}:
        yield from _iter_ndjson(path); return
    with path.open("r", encoding="utf-8") as f:
        preview = f.read(2048)
    first_non_ws = next((ch for ch in preview if not ch.isspace()), "")
    if first_non_ws == "[":
        yield from _iter_json_array_stream(path)
    else:
        yield from _iter_ndjson(path)


# -----------------------------
# Label normalization + UUID mapping
# -----------------------------
def norm_label(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return unicodedata.normalize("NFKC", str(s)).strip().casefold()


@dataclass(frozen=True)
class LabelMap:
    raw_first_map: Dict[str, str]
    norm_first_map: Dict[str, str]
    raw_last_map: Dict[str, str]
    norm_last_map: Dict[str, str]

    @classmethod
    def build_from_csv(
        cls,
        csv_path: Path,
        label_col: str = "PREFERREDLABEL",
        uuidhist_col: str = "UUIDHISTORY",
    ) -> "LabelMap":
        df = pd.read_csv(csv_path)
        if label_col not in df.columns or uuidhist_col not in df.columns:
            raise ValueError(
                f"CSV {csv_path} must include columns '{label_col}' and '{uuidhist_col}'"
            )
        df = df.dropna(subset=[label_col, uuidhist_col]).copy()
        split = df[uuidhist_col].astype(str).str.split("\n")
        first_uuids = split.str[0]
        last_uuids  = split.str[-1]
        labels = df[label_col].astype(str)

        raw_first_map = dict(zip(labels, first_uuids))
        norm_first_map = {norm_label(lbl): u for lbl, u in zip(labels, first_uuids)}
        raw_last_map  = dict(zip(labels, last_uuids))
        norm_last_map = {norm_label(lbl): u for lbl, u in zip(labels, last_uuids)}
        return cls(raw_first_map, norm_first_map, raw_last_map, norm_last_map)

    def map_label_pair(self, label: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if label is None:
            return (None, None)
        nl = norm_label(label)
        first = self.norm_first_map.get(nl, self.raw_first_map.get(label))
        last  = self.norm_last_map.get(nl,  self.raw_last_map.get(label))
        return (first, last)


def _extract_labels_from_llm_field(value: Any) -> List[str]:
    """
    Accepts:
      - list[str]
      - list[dict] with label-ish keys
      - str containing JSON list (e.g. '["SQL","Excel"]')
      - str comma/pipe/semicolon separated ('SQL, Excel' or 'SQL|Excel')
    """
    labels: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    labels.append(item.strip())
            elif isinstance(item, dict):
                lab = item.get("preferred_label") or item.get("label") or item.get("skill_label") or item.get("name")
                if isinstance(lab, str) and lab.strip():
                    labels.append(lab.strip())
        return labels

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            return _extract_labels_from_llm_field(obj)
        except Exception:
            pass
        if any(sep in s for sep in [",", "|", ";"]):
            parts = [p.strip() for p in s.replace("|", ",").replace(";", ",").split(",")]
            return [p for p in parts if p]
        return [s]
    return []


def map_labels_with_audit(
    labels: Iterable[str],
    mapper: LabelMap,
    drop_nulls: bool = False,
    audit_sink: Optional[List[Tuple[str, Optional[str]]]] = None,
) -> List[Dict[str, Optional[str]]]:
    out: List[Dict[str, Optional[str]]] = []
    for lab in labels or []:
        first_uuid, last_uuid = mapper.map_label_pair(lab)
        unmapped = (first_uuid is None) and (last_uuid is None)
        if unmapped and drop_nulls:
            if audit_sink is not None:
                audit_sink.append((lab, None))
            continue
        if unmapped and audit_sink is not None:
            audit_sink.append((lab, None))
        out.append({"preferred_label": lab, "uuid": first_uuid, "originUUID": last_uuid})
    return out


# -----------------------------
# Domain helpers
# -----------------------------
def extract_occupation(rec: Dict[str, Any]) -> Optional[Dict[str, Optional[str]]]:
    def _pick_label(d: dict) -> Optional[str]:
        for k in ("occupation", "preferred_label", "label", "name", "title"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    fc = rec.get("final_choice")
    if isinstance(fc, str) and fc.strip():
        return {"preferred_label": fc.strip()}
    if isinstance(fc, dict):
        lab = _pick_label(fc)
        if lab:
            return {"preferred_label": lab}
    if isinstance(fc, list) and fc:
        for item in fc:
            if isinstance(item, dict):
                lab = _pick_label(item)
                if lab:
                    return {"preferred_label": lab}
            elif isinstance(item, str) and item.strip():
                return {"preferred_label": item.strip()}

    for k in ("final_occupation", "occupation_final", "chosen_occupation"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return {"preferred_label": v.strip()}
        if isinstance(v, dict):
            lab = _pick_label(v)
            if lab:
                return {"preferred_label": lab}

    ranked = rec.get("ranked_occupations") or rec.get("occupations_ranked") or []
    if isinstance(ranked, list) and ranked:
        dict_items = [x for x in ranked if isinstance(x, dict)]
        if dict_items:
            best = next((o for o in dict_items if o.get("rank") == 1), None)
            if best is None:
                try:
                    best = min(dict_items, key=lambda o: o.get("rank", float("inf")))
                except Exception:
                    best = dict_items[0]
            lab = _pick_label(best)
            if lab:
                return {"preferred_label": lab}
        else:
            s0 = ranked[0]
            if isinstance(s0, str) and s0.strip():
                return {"preferred_label": s0.strip()}
    return None


def deterministic_sort(skill_list: List[Dict[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
    return sorted(
        skill_list,
        key=lambda d: (
            (d.get("preferred_label") or "").casefold(),
            "" if d.get("uuid") is None else d.get("uuid"),
        ),
    )


def disjoint_optional_from_essential(
    essential: List[Dict[str, Optional[str]]],
    optional: List[Dict[str, Optional[str]]],
) -> List[Dict[str, Optional[str]]]:
    ess_norms = {norm_label(x.get("preferred_label")) for x in essential if x}
    out: List[Dict[str, Optional[str]]] = []
    for item in optional:
        if norm_label(item.get("preferred_label")) in ess_norms:
            continue
        out.append(item)
    return out


def canon_id(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit():
            return head
        try:
            f = float(s)
            i = int(f)
            if f == i:
                return str(i)
        except Exception:
            pass
    try:
        f = float(s)
        i = int(f)
        if f == i:
            return str(i)
    except Exception:
        pass
    return s


def _now_utc_normalized() -> pd.Timestamp:
    try:
        ts = pd.Timestamp.utcnow()
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    except Exception:
        ts = pd.Timestamp.now(tz="UTC")
    return ts.normalize()


def compute_active_flags(df: pd.DataFrame) -> pd.Series:
    today_utc = _now_utc_normalized()
    date_closing = pd.to_datetime(df.get("date_closing"), utc=True, errors="coerce")
    date_posted  = pd.to_datetime(df.get("date_posted"),  utc=True, errors="coerce")
    active = pd.Series(True, index=df.index)
    has_closing = date_closing.notna()
    active.loc[has_closing] = today_utc <= date_closing[has_closing]
    mask = (~has_closing) & date_posted.notna()
    active.loc[mask] = (today_utc - date_posted[mask]) <= pd.Timedelta(days=120)
    return active


# -----------------------------
# ESCO taxonomy (DAG) & selection
# -----------------------------
@dataclass
class TaxonomyIndex:
    # DAG: child ID -> set of parent IDs (may be empty)
    parents_of: Dict[str, Set[str]]
    # any skill UUID in UUIDHISTORY -> skill ID (for bridging)
    skill_uuid_to_id: Dict[str, str]
    # group ID -> (preferred_label, first_uuid, last_uuid)
    group_id_to_meta: Dict[str, Tuple[str, str, str]]
    # memo: node ID -> set of chains (each chain is tuple of group IDs, TOP→DEEP)
    memo_chains_top: Dict[str, Set[Tuple[str, ...]]]

    def _chains_to_node_top(self, node_id: str, visited: Set[str]) -> Set[Tuple[str, ...]]:
        """
        Return ALL group chains from ROOT→...→node_id (TOP→DEEP),
        considering the DAG. Each chain is a tuple of group IDs (only group nodes).
        """
        if node_id in visited:
            logging.warning("Cycle detected involving node %s; cutting this path.", node_id)
            return set()
        if node_id in self.memo_chains_top:
            return self.memo_chains_top[node_id]

        is_group = node_id in self.group_id_to_meta
        parents = self.parents_of.get(node_id, set())
        out: Set[Tuple[str, ...]] = set()

        if not parents:
            # At a root: chain is either [node] if node is a group, else empty []
            chain = (node_id,) if is_group else tuple()
            out.add(chain)
            self.memo_chains_top[node_id] = out
            return out

        new_visited = set(visited)
        new_visited.add(node_id)

        for p in parents:
            parent_chains = self._chains_to_node_top(p, new_visited)
            if not parent_chains:
                continue
            for ch in parent_chains:
                if is_group:
                    out.add(ch + (node_id,))
                else:
                    out.add(ch)

        self.memo_chains_top[node_id] = out
        return out

    def group_chains_top(self, node_id: Optional[str]) -> List[List[str]]:
        """
        Public wrapper: returns list of non-empty TOP→DEEP group chains to node_id.
        """
        if not node_id:
            return []
        chains = self._chains_to_node_top(node_id, visited=set())
        # Keep only chains that contain at least one group
        return [list(ch) for ch in chains if len(ch) > 0]

    def groups_at_level(self, node_id: Optional[str], level_spec: str) -> Set[str]:
        """
        For the given node, return ALL group IDs that appear at the requested
        level across ALL parent paths. Level counts from TOP (1=top/root).
        """
        result: Set[str] = set()
        chains = self.group_chains_top(node_id)
        for ch in chains:
            idx = _resolve_level_index_from_top_length(len(ch), level_spec)
            result.add(ch[idx])
        return result


def build_taxonomy_index(
    skills_csv: Path,
    skill_groups_csv: Path,
    skill_hierarchy_csv: Path,
) -> TaxonomyIndex:
    # 1) child -> SET[parent] from hierarchy (DAG)
    h = pd.read_csv(skill_hierarchy_csv)
    required_cols = {"CHILDID", "PARENTID"}
    if not required_cols.issubset(set(h.columns)):
        raise ValueError(f"{skill_hierarchy_csv} must have columns {sorted(required_cols)}")
    parents_of: Dict[str, Set[str]] = {}
    for _, row in h.iterrows():
        child = str(row["CHILDID"]).strip()
        parent = str(row["PARENTID"]).strip()
        if not child or not parent:
            continue
        parents_of.setdefault(child, set()).add(parent)

    # 2) group meta by ID
    gdf = pd.read_csv(skill_groups_csv)
    for col in ("ID", "PREFERREDLABEL", "UUIDHISTORY"):
        if col not in gdf.columns:
            raise ValueError(f"{skill_groups_csv} missing column '{col}'")
    gdf = gdf.dropna(subset=["ID", "PREFERREDLABEL", "UUIDHISTORY"]).copy()
    g_split = gdf["UUIDHISTORY"].astype(str).str.split("\n")
    group_id_to_meta: Dict[str, Tuple[str, str, str]] = {}
    for gid, label, uuids in zip(gdf["ID"].astype(str), gdf["PREFERREDLABEL"].astype(str), g_split):
        first_u = uuids[0] if len(uuids) else None
        last_u  = uuids[-1] if len(uuids) else None
        group_id_to_meta[gid] = (label, first_u, last_u)

    # 3) skill uuid → ID (bridge)
    sdf = pd.read_csv(skills_csv)
    for col in ("ID", "UUIDHISTORY"):
        if col not in sdf.columns:
            raise ValueError(f"{skills_csv} missing column '{col}'")
    sdf = sdf.dropna(subset=["ID", "UUIDHISTORY"]).copy()
    skill_uuid_to_id: Dict[str, str] = {}
    for sid, uuids in zip(sdf["ID"].astype(str), sdf["UUIDHISTORY"].astype(str)):
        for u in uuids.split("\n"):
            u = u.strip()
            if not u:
                continue
            if u not in skill_uuid_to_id:
                skill_uuid_to_id[u] = sid

    return TaxonomyIndex(
        parents_of=parents_of,
        skill_uuid_to_id=skill_uuid_to_id,
        group_id_to_meta=group_id_to_meta,
        memo_chains_top={},
    )


def _resolve_level_index_from_top_length(n: int, level_spec: str) -> int:
    """
    Resolve the 0-based index for a TOP→DEEP chain of length n.

    - Numeric string "1","2","3"...: 1 = top, 2 = penultimate, …
    - Aliases: "top"->1, "penultimate"->2, "deepest"->n
    - If requested level exceeds n, clamp to n.
    """
    s = str(level_spec).strip().lower()
    if s in {"top"}:
        lvl = 1
    elif s in {"penultimate", "second"}:
        lvl = 2
    elif s in {"deepest", "bottom", "last"}:
        lvl = n
    else:
        if s.isdigit():
            lvl = max(1, int(s))
        else:
            # Default to penultimate
            lvl = 2
    idx = min(lvl - 1, n - 1)
    return idx


def _bucket_groups_selected_multi(
    skill_bucket: List[Dict[str, Optional[str]]],
    tx: TaxonomyIndex,
    level_spec: str = "2",
) -> List[Dict[str, Optional[str]]]:
    """
    For a bucket (essential/optional), return the DEDUPED set of groups
    at the requested level across ALL skills and ALL DAG paths.
    """
    group_ids: Set[str] = set()

    # 1) Resolve group IDs at requested level across ALL skills
    for sk in skill_bucket or []:
        if not isinstance(sk, dict):
            continue
        # Prefer originUUID (latest), then uuid (earliest)
        candidates = [sk.get("originUUID"), sk.get("uuid")]
        skill_id: Optional[str] = None
        for u in candidates:
            if u and u in tx.skill_uuid_to_id:
                skill_id = tx.skill_uuid_to_id[u]
                break
        if not skill_id:
            continue
        group_ids |= tx.groups_at_level(skill_id, level_spec)

    # 2) Convert to meta and dedupe by UUID pair
    out: List[Dict[str, Optional[str]]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for gid in group_ids:
        meta = tx.group_id_to_meta.get(gid)
        if not meta:
            continue
        label, first_u, last_u = meta
        key = (first_u or "", last_u or "")
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        out.append({"preferred_label": label, "uuid": first_u, "originUUID": last_u})

    # 3) Deterministic sort
    out.sort(key=lambda d: ((d.get("preferred_label") or "").casefold(), d.get("uuid") or ""))
    return out


# -----------------------------
# Atomic write + provenance
# -----------------------------
def atomic_write_text(path: Path, text: str, retries: int = 3, delay_s: float = 0.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as f:
                tmp = Path(f.name)
                f.write(text); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path); return
        except Exception as e:
            logging.warning("Atomic write attempt %d failed for %s: %s", attempt, path, e)
            if tmp and tmp.exists():
                try: tmp.unlink()
                except Exception: pass
            if attempt < retries:
                time.sleep(delay_s)
            else:
                raise


def dump_provenance_sidecar(out_path: Path, sources: Dict[str, Path]) -> None:
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            name: {
                "path": str(p),
                "exists": p.exists(),
                "size": (p.stat().st_size if p.exists() else None),
                "modified_at": (datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
                                if p.exists() else None),
            } for name, p in sources.items()
        },
    }
    atomic_write_text(out_path.with_suffix(out_path.suffix + ".meta.json"), json.dumps(meta, indent=2))


# -----------------------------
# Core pipeline
# -----------------------------
def build_frames(
    occupations_json_path: Path,
    skills_optional_path: Optional[Path],
    skills_essential_path: Optional[Path],
    extra_data_path: Path,
    skills_entity_path: Path,
    occupations_entity_path: Path,
    remove_null_skill_uuids: bool,
    unmapped_audit_csv: Optional[Path],
    drop_error_rows: bool,
    # Taxonomy inputs
    skill_groups_entity_path: Path,
    skill_hierarchy_path: Path,
    # Which group level to select (string: int≥1 or alias)
    skillgroup_level: str = "2",
) -> pd.DataFrame:
    skill_map = LabelMap.build_from_csv(skills_entity_path)
    occ_map   = LabelMap.build_from_csv(occupations_entity_path)

    # ---- Occupations --------------------------------------------------------
    occ_rows: List[Dict[str, Any]] = []
    for rec in _iter_records_flex(occupations_json_path):
        if drop_error_rows and rec.get("error"):
            continue
        best = extract_occupation(rec)
        if not best:
            occ_payload = None
        else:
            first_u, last_u = occ_map.map_label_pair(best.get("preferred_label"))
            occ_payload = {
                "preferred_label": best.get("preferred_label"),
                "uuid": first_u,
                "originUUID": last_u,
            }
        occ_rows.append({
            "opportunity_group_id": str(rec.get("opportunity_group_id")) if rec.get("opportunity_group_id") is not None else None,
            "opportunity_ref_id":   str(rec.get("opportunity_ref_id"))   if rec.get("opportunity_ref_id")   is not None else None,
            "opportunity_title": rec.get("opportunity_title"),
            "occupation": occ_payload,
        })
    occ_df = pd.DataFrame(occ_rows)
    for c in ID_COLUMNS:
        if c in occ_df.columns:
            occ_df[c] = occ_df[c].map(canon_id)

    # ---- Optional skills ----------------------------------------------------
    opt_rows: List[Dict[str, Any]] = []
    if skills_optional_path:
        for rec in _iter_records_flex(skills_optional_path):
            if drop_error_rows and rec.get("error"):
                continue
            opt_labels = _extract_labels_from_llm_field(rec.get("optional_skills") or [])
            mapped = map_labels_with_audit(opt_labels, skill_map, drop_nulls=False, audit_sink=None)
            opt_rows.append({
                "opportunity_group_id": str(rec.get("opportunity_group_id")) if rec.get("opportunity_group_id") is not None else None,
                "opportunity_ref_id":   str(rec.get("opportunity_ref_id"))   if rec.get("opportunity_ref_id")   is not None else None,
                "optional_skills": mapped,
            })
    opt_df = pd.DataFrame(opt_rows) if opt_rows else pd.DataFrame(columns=ID_COLUMNS + ["optional_skills"])
    for c in ID_COLUMNS:
        if c in opt_df.columns:
            opt_df[c] = opt_df[c].map(canon_id)

    # ---- Essential skills ---------------------------------------------------
    ess_rows: List[Dict[str, Any]] = []
    if skills_essential_path:
        for rec in _iter_records_flex(skills_essential_path):
            if drop_error_rows and rec.get("error"):
                continue
            ess_labels = _extract_labels_from_llm_field(rec.get("essential_skills") or [])
            mapped = map_labels_with_audit(ess_labels, skill_map, drop_nulls=False, audit_sink=None)
            ess_rows.append({
                "opportunity_group_id": str(rec.get("opportunity_group_id")) if rec.get("opportunity_group_id") is not None else None,
                "opportunity_ref_id":   str(rec.get("opportunity_ref_id"))   if rec.get("opportunity_ref_id")   is not None else None,
                "essential_skills": mapped,
            })
    ess_df = pd.DataFrame(ess_rows) if ess_rows else pd.DataFrame(columns=ID_COLUMNS + ["essential_skills"])
    for c in ID_COLUMNS:
        if c in ess_df.columns:
            ess_df[c] = ess_df[c].map(canon_id)

    # Merge skills
    skills_df = pd.merge(opt_df, ess_df, on=ID_COLUMNS, how="outer")
    for col in ("optional_skills","essential_skills"):
        if col in skills_df.columns:
            skills_df[col] = skills_df[col].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else x))

    def _massage_skill_buckets(row: pd.Series) -> pd.Series:
        ess = deterministic_sort(row.get("essential_skills", []) or [])
        opt = deterministic_sort(row.get("optional_skills", []) or [])
        opt = disjoint_optional_from_essential(ess, opt)
        if remove_null_skill_uuids:
            ess = [d for d in ess if (d.get("uuid") or d.get("originUUID"))]
            opt = [d for d in opt if (d.get("uuid") or d.get("originUUID"))]
        row["essential_skills"] = ess
        row["optional_skills"] = opt
        return row

    if not skills_df.empty:
        skills_df = skills_df.apply(_massage_skill_buckets, axis=1)

    # ---- Extra table & merge ------------------------------------------------
    extra_df = pd.read_csv(extra_data_path)
    for c in ID_COLUMNS:
        if c in extra_df.columns:
            extra_df[c] = extra_df[c].map(canon_id)
    extra_df["active"] = compute_active_flags(extra_df)

    for c in ID_COLUMNS:
        if c in occ_df.columns:
            occ_df[c] = occ_df[c].astype(str)
    combined = pd.merge(occ_df, skills_df, on=ID_COLUMNS, how="left")
    final_df = pd.merge(
        extra_df, combined, on=ID_COLUMNS, how="left", suffixes=("_extra", "_llm")
    )

    # Prefer extra columns
    def coalesce_cols(df, base, left_name, right_name, prefer="left"):
        left  = df[left_name]  if left_name  in df.columns else None
        right = df[right_name] if right_name in df.columns else None
        if left is not None and right is not None:
            df[base] = (left if prefer == "left" else right).combine_first(
                       right if prefer == "left" else left)
            df.drop(columns=[left_name, right_name], inplace=True)
        elif left is not None:
            df.rename(columns={left_name: base}, inplace=True)
        elif right is not None:
            df.rename(columns={right_name: base}, inplace=True)

    coalesce_cols(final_df, "opportunity_title", "opportunity_title_extra", "opportunity_title_llm", "left")
    coalesce_cols(final_df, "company_name",    "company_name_extra",    "company_name_llm",    "left")
    coalesce_cols(final_df, "full_details",    "full_details_extra",    "full_details_llm",    "left")
    coalesce_cols(final_df, "opportunity_url", "opportunity_url_extra", "opportunity_url_llm", "left")

    # Normalize list/object columns
    for col in ("essential_skills","optional_skills"):
        if col not in final_df.columns:
            final_df[col] = [[] for _ in range(len(final_df))]
        else:
            final_df[col] = final_df[col].apply(
                lambda x: x if isinstance(x, list) else ([] if (x is None or (isinstance(x, float) and pd.isna(x))) else x)
            )
    if "occupation" in final_df.columns:
        final_df["occupation"] = final_df["occupation"].apply(
            lambda x: None if (x is None or (isinstance(x, float) and pd.isna(x))) else x
        )

    # ---- ESCO taxonomy: build index & attach groups (DAG-aware) ------------
    tx = build_taxonomy_index(
        skills_csv=skills_entity_path,
        skill_groups_csv=skill_groups_entity_path,
        skill_hierarchy_csv=skill_hierarchy_path,
    )

    def _attach_groups(row: pd.Series) -> pd.Series:
        ess = row.get("essential_skills", []) or []
        opt = row.get("optional_skills", []) or []
        row["essential_skillgroups"] = _bucket_groups_selected_multi(ess, tx, level_spec=skillgroup_level)
        row["optional_skillgroups"]  = _bucket_groups_selected_multi(opt, tx, level_spec=skillgroup_level)
        return row

    if not final_df.empty:
        final_df = final_df.apply(_attach_groups, axis=1)

    # Dedupe
    before = len(final_df)
    final_df = final_df.drop_duplicates(subset=ID_COLUMNS, keep="first").copy()
    logging.info("Removed %d duplicate rows; remaining: %d", before - len(final_df), len(final_df))

    # Timestamps + row hash
    now_iso = datetime.now(timezone.utc).isoformat()
    final_df["created_at"] = now_iso
    final_df["updated_at"] = now_iso

    def _row_hash(row: pd.Series) -> str:
        key = "|".join(str(row.get(c)) for c in ID_COLUMNS) + "|" + str(row.get("updated_at"))
        return sha1(key.encode("utf-8")).hexdigest()
    final_df["row_hash"] = final_df.apply(_row_hash, axis=1)

    return final_df


def dataframe_to_json_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    import math
    def _to_json_safe(v):
        if isinstance(v, (list, tuple)):
            return [_to_json_safe(x) for x in v]
        if isinstance(v, dict):
            return {k: _to_json_safe(x) for k, x in v.items()}
        if isinstance(v, (pd.Timestamp, datetime)):
            try:
                return v.tz_convert("UTC").isoformat()
            except Exception:
                return v.isoformat()
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    records = df.to_dict(orient="records")
    return [_to_json_safe(r) for r in records]


def validate_record(rec: Dict[str, Any]) -> None:
    for k in ID_COLUMNS:
        assert k in rec and isinstance(rec[k], (str, int)), f"Missing/invalid ID field {k}"
    for bucket in ("essential_skills", "optional_skills"):
        arr = rec.get(bucket, [])
        assert isinstance(arr, list), f"{bucket} must be a list"
        for item in arr:
            assert isinstance(item, dict), f"{bucket} items must be objects"
            assert "preferred_label" in item, f"{bucket} item missing preferred_label"
            u_first = item.get("uuid", None)
            u_last  = item.get("originUUID", None)
            assert (u_first is None) or isinstance(u_first, str), f"{bucket} item uuid invalid type"
            assert (u_last  is None) or isinstance(u_last,  str), f"{bucket} item originUUID invalid type"
    for gfield in ("essential_skillgroups", "optional_skillgroups"):
        garr = rec.get(gfield, [])
        if garr is None:
            continue
        assert isinstance(garr, list), f"{gfield} must be a list"
        for item in garr:
            assert isinstance(item, dict), f"{gfield} items must be objects"
            assert "preferred_label" in item, f"{gfield} item missing preferred_label"
            if "uuid" in item:        assert (item["uuid"] is None) or isinstance(item["uuid"], str)
            if "originUUID" in item:  assert (item["originUUID"] is None) or isinstance(item["originUUID"], str)
    occ = rec.get("occupation", None)
    if occ is not None:
        assert isinstance(occ, dict), "occupation must be an object or null"
        assert "preferred_label" in occ, "occupation missing preferred_label"
        if "uuid" in occ:        assert (occ["uuid"] is None) or isinstance(occ["uuid"], str)
        if "originUUID" in occ:  assert (occ["originUUID"] is None) or isinstance(occ["originUUID"], str)


def validate_before_write(records: List[Dict[str, Any]]) -> None:
    for rec in records:
        validate_record(rec)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Create opportunity DB with essential/optional skills, plus selected-level ESCO skill groups (DAG-aware).")
    ap.add_argument("--occupations", type=Path, required=True)
    ap.add_argument("--skills-optional", type=Path, required=False)
    ap.add_argument("--skills-essential", type=Path, required=False)
    ap.add_argument("--skills-entities", type=Path, required=True)
    ap.add_argument("--skill-groups-entities", type=Path, required=True, help="Path to ESCO skill_groups.csv")
    ap.add_argument("--skill-hierarchy", type=Path, required=True, help="Path to ESCO skill_hierarchy.csv (CHILDID,PARENTID)")
    ap.add_argument("--occ-entities", type=Path, required=True)
    ap.add_argument("--extra", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--unmapped-audit-out", type=Path, required=False)
    ap.add_argument("--remove-null-skill-uuids", action="store_true")
    ap.add_argument("--drop-error-rows", action="store_true", help="Drop LLM rows that contain an 'error' field.")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument(
        "--skillgroup-level",
        default="2",
        help="Which ESCO skill_group level from the TOP to attach. "
             "Accepts integer >=1 (1=top, 2=penultimate, 3=third-from-top, ...), "
             "or aliases: top, penultimate, deepest. Default: 2.",
    )
    args = ap.parse_args(argv)

    setup_logging(args.log_level)

    sources = {
        "occupations": args.occupations,
        "skills_optional": args.skills_optional or Path(""),
        "skills_essential": args.skills_essential or Path(""),
        "skills_entities": args.skills_entities,
        "skill_groups_entities": args.skill_groups_entities,
        "skill_hierarchy": args.skill_hierarchy,
        "occupations_entities": args.occ_entities,
        "extra": args.extra,
    }

    df = build_frames(
        occupations_json_path=args.occupations,
        skills_optional_path=args.skills_optional,
        skills_essential_path=args.skills_essential,
        extra_data_path=args.extra,
        skills_entity_path=args.skills_entities,
        occupations_entity_path=args.occ_entities,
        remove_null_skill_uuids=args.remove_null_skill_uuids,
        unmapped_audit_csv=args.unmapped_audit_out,
        drop_error_rows=args.drop_error_rows,
        skill_groups_entity_path=args.skill_groups_entities,
        skill_hierarchy_path=args.skill_hierarchy,
        skillgroup_level=str(args.skillgroup_level),
    )

    records = dataframe_to_json_records(df)
    validate_before_write(records)

    text = json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False)
    atomic_write_text(args.out, text)
    dump_provenance_sidecar(args.out, sources)

    logging.info(
        "Wrote %d records to %s (skillgroup level = %s)",
        len(records), args.out, str(args.skillgroup_level)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
