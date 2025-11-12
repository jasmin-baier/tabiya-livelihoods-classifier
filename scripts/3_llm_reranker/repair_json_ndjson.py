#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
repair_json_ndjson.py
---------------------
Repairs mixed/dirty JSON or NDJSON files and writes clean NDJSON.

Fixes:
  - Non-JSON tokens: NaN, Infinity, -Infinity -> null
  - Stray lines: "", numbers, null -> skipped
  - Trailing commas before } or ] removed
  - Strips UTF-8 BOM if present
  - Accepts input as NDJSON *or* JSON array; output = NDJSON
  - Only writes JSON *objects* (dicts). Other items are skipped.

Usage:
python scripts/3_llm_reranker/repair_json_ndjson.py `
--in "C:\Users\jasmi\Downloads\llm_opportunity_responses_occupations.compact.json" `
--out "C:\Users\jasmi\Downloads\llm_opportunity_responses_occupations_final.json" `
--log-level INFO

python scripts/3_llm_reranker/repair_json_ndjson.py `
--in "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\data\pre_study\llm_opportunity_responses_skills_optional.ndjson" `
--out "C:\Users\jasmi\OneDrive - Nexus365\Documents\PhD - Oxford BSG\Paper writing projects\Ongoing\Compass\data\pre_study\llm_opportunity_responses_skills_optional_repaired.ndjson" `
--log-level INFO

"""

from __future__ import annotations
import argparse, json, logging, re
from pathlib import Path
from typing import Dict, Iterator, Any

# ---------- helpers ----------

RE_NAN = re.compile(r'(?<!")\bNaN\b(?!")')
RE_INF = re.compile(r'(?<!")\bInfinity\b(?!")')
RE_NINF = re.compile(r'(?<!")\b-Infinity\b(?!")')
RE_TRAIL_COMMA = re.compile(r',(\s*[}\]])')   # ", }" or ", ]" -> " }" or " ]"

def _clean_line(s: str) -> str:
    s = s.lstrip("\ufeff").strip()                # remove BOM + trim
    if not s:
        return ""
    # Replace bare NaN/Infinity with null (JSON standard)
    s = RE_NINF.sub("null", s)
    s = RE_INF.sub("null", s)
    s = RE_NAN.sub("null", s)
    # Remove trailing commas before } or ]
    s = RE_TRAIL_COMMA.sub(r"\1", s)
    return s

def _iter_ndjson(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            s = _clean_line(raw)
            if not s or s in ('""', '"', "'", "null", "[]"):
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                logging.warning("NDJSON parse skip line %d (%s)", i, e)
                continue
            if isinstance(obj, dict):
                yield obj
            else:
                # skip non-object items like "", 0, [], etc.
                logging.info("NDJSON non-object on line %d skipped (%s)", i, type(obj).__name__)

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
            # small cleanup in buffer too
            buf = _clean_line(buf)

            if not in_array:
                # find '['
                while idx < len(buf) and buf[idx].isspace():
                    idx += 1
                if idx < len(buf) and buf[idx] == "[":
                    in_array = True
                    idx += 1
                else:
                    if not chunk:
                        logging.error("Not a JSON array file.")
                        return
                    continue

            while True:
                # skip commas/space
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
                idx = end
                # sanitize tokens inside decoded object (very rare here)
                if isinstance(obj, dict):
                    yield obj
                else:
                    logging.info("Array non-object element skipped (%s)", type(obj).__name__)
            if idx > 0:
                buf = buf[idx:]
                idx = 0

def _guess_mode(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        preview = f.read(2048).lstrip("\ufeff")
    for ch in preview:
        if ch.isspace():
            continue
        return "array" if ch == "[" else "ndjson"
    return "ndjson"

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Repair mixed JSON/NDJSON into clean NDJSON.")
    ap.add_argument("--in",  dest="inp", type=Path, required=True, help="Input .json or .ndjson")
    ap.add_argument("--out", dest="out", type=Path, required=True, help="Output .ndjson (clean)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    mode = _guess_mode(args.inp)
    logging.info("Detected input mode: %s", mode)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8", newline="\n") as w:
        iterator = _iter_json_array_stream(args.inp) if mode == "array" else _iter_ndjson(args.inp)
        for obj in iterator:
            # write strict JSON (no NaN/Infinity)
            w.write(json.dumps(obj, ensure_ascii=False, allow_nan=False))
            w.write("\n")
            written += 1

    logging.info("Done. Wrote %d clean objects to %s", written, args.out)

if __name__ == "__main__":
    main()
