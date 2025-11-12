r"""

Convert a mixed-format JSON file (top-level array + NDJSON tail) into clean NDJSON.
This may be necessary due to having used different arguments in 3_1_LLM_pick_skills_full_details.py

python scripts/3_llm_reranker/mixed-json-to-ndjson.py `
  --input  "C:\Users\jasmi\Downloads\llm_opportunity_responses_skills_essential.json" `
  --output "C:\Users\jasmi\Downloads\llm_opportunity_responses_skills_essential.ndjson" `
  --dedupe last `
  --key opportunity_group_id

"""

#!/usr/bin/env python3
import argparse, io, json, logging, os, sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Try ijson for streaming JSON arrays; fall back if unavailable
try:
    import ijson  # type: ignore
    HAVE_IJSON = True
except Exception:
    HAVE_IJSON = False


def peek_first_non_ws(fp: io.BufferedReader) -> str:
    """Return the first non-whitespace character (or '') and reset the file pointer."""
    pos = fp.tell()
    ch = b''
    while True:
        b = fp.read(1)
        if not b:
            break
        if not b.isspace():
            ch = b
            break
    fp.seek(pos)
    return ch.decode('utf-8', errors='ignore')


def iter_json_array_items(fp: io.BufferedReader) -> Iterable[dict]:
    """
    Yield objects from a top-level JSON array.
    Prefer ijson (streaming). If unavailable, fall back to stdlib (loads entire array).
    Leaves fp positioned right after the array when ijson is used; otherwise resets to end of array.
    """
    if HAVE_IJSON:
        # Streaming parse: "[ {...}, {...}, ... ]"
        for obj in ijson.items(fp, 'item'):
            if isinstance(obj, dict):
                yield obj
            else:
                # Only yield dict-like records; skip other item types defensively
                continue
        # ijson leaves fp at end of the array; the caller can continue reading remainder as NDJSON
        return

    # Fallback: read the whole file content and raw-decode the first top-level JSON value (array)
    # This supports mixed files: we decode one array and ignore trailing text here (caller continues)
    content = fp.read().decode('utf-8', errors='ignore')
    dec = json.JSONDecoder()
    obj, end_idx = dec.raw_decode(content)  # may raise
    if not isinstance(obj, list):
        raise ValueError("Top-level JSON value is not an array.")
    for item in obj:
        if isinstance(item, dict):
            yield item
    # Move file pointer to the position right after the array for the caller to read the tail
    # (We reopen a BytesIO over the unread tail for the caller to continue reading lines.)
    fp.seek(0, io.SEEK_SET)
    # Replace the file object with a BytesIO of the unread tail by monkey-patching .readlines usage.
    # Instead, we just stash the tail on the fp object for the caller.
    fp.tail_from = end_idx  # type: ignore[attr-defined]


def iter_mixed_records(path: Path) -> Tuple[Iterable[dict], dict]:
    """
    Inspect the file and return an iterator over all JSON objects in (array part + NDJSON tail).
    Also returns a stats dict.
    """
    stats = dict(array_items=0, ndjson_lines=0, ndjson_objects=0, invalid_lines=0, format="unknown", used_ijson=HAVE_IJSON)

    # Open in binary for consistent positioning
    f = open(path, 'rb')

    first = peek_first_non_ws(f)
    if first == '[':
        stats["format"] = "array or mixed"
        # Yield array items
        def gen():
            nonlocal f, stats
            try:
                for obj in iter_json_array_items(f):
                    stats["array_items"] += 1
                    yield obj
            except Exception as e:
                logging.warning(f"Failed to parse JSON array cleanly: {e}. Continuing with NDJSON tail if any.")

            # If the fallback path set a 'tail_from' attribute, respect it
            tail_from = getattr(f, 'tail_from', None)
            if tail_from is not None:
                f.close()
                with open(path, 'rb') as f2:
                    f2.seek(tail_from)
                    for obj in _iter_ndjson(f2, stats):
                        yield obj
            else:
                # ijson path: we are already at the end of the array; read remainder as NDJSON
                for obj in _iter_ndjson(f, stats):
                    yield obj

        return gen(), stats
    else:
        stats["format"] = "ndjson"
        # Pure (or assumed) NDJSON
        def gen():
            for obj in _iter_ndjson(f, stats):
                yield obj
        return gen(), stats


def _iter_ndjson(f: io.BufferedReader, stats: dict) -> Iterable[dict]:
    """Yield JSON objects from NDJSON lines."""
    for raw in f:
        stats["ndjson_lines"] += 1
        line = raw.decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                stats["ndjson_objects"] += 1
                yield rec
        except Exception:
            stats["invalid_lines"] += 1
            continue


def write_ndjson(records: Iterable[dict], out_path: Path, key: str, dedupe: str):
    """
    Write records to NDJSON. Dedupe options:
      - 'none'  : write all
      - 'first' : first occurrence wins (streaming, low memory)
      - 'last'  : last occurrence wins (stores all in memory, then writes)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dedupe == 'none':
        with open(out_path, 'w', encoding='utf-8', newline='\n') as out:
            for rec in records:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return

    if dedupe == 'first':
        seen = set()
        kept = 0
        with open(out_path, 'w', encoding='utf-8', newline='\n') as out:
            for rec in records:
                rid = rec.get(key)
                if rid is None:
                    # Keep unkeyed records as-is
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    continue
                if rid in seen:
                    continue
                seen.add(rid)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
        logging.info(f"first-wins dedupe wrote {kept} unique records.")
        return

    if dedupe == 'last':
        # Stores all records in memory (key -> record). Newest wins.
        store: Dict[str, dict] = {}
        unkeyed = []
        total = 0
        for rec in records:
            total += 1
            rid = rec.get(key)
            if rid is None:
                unkeyed.append(rec)
            else:
                store[rid] = rec
        with open(out_path, 'w', encoding='utf-8', newline='\n') as out:
            # Preserve insertion-ish order: unkeyed first, then keyed by insertion order of dict (last wins)
            for rec in unkeyed:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for rec in store.values():
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logging.info(f"last-wins dedupe: read {total} records → wrote {len(unkeyed) + len(store)}.")
        return

    raise ValueError(f"Unknown dedupe mode: {dedupe}")


def main():
    ap = argparse.ArgumentParser(description="Convert possibly mixed JSON-array/NDJSON file into clean NDJSON.")
    ap.add_argument("--input", required=True, help="Path to the mixed or single-format input (.json or .ndjson).")
    ap.add_argument("--output", required=True, help="Path to write .ndjson.")
    ap.add_argument("--key", default="opportunity_ref_id", help="Field to use for de-duplication (default: opportunity_ref_id).")
    ap.add_argument("--dedupe", choices=["none", "first", "last"], default="last",
                   help="De-duplication strategy: none | first (streaming) | last (memory-heavy). Default: last.")
    ap.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    in_path = Path(args.input)
    out_path = Path(args.output)

    logging.info(f"Reading: {in_path}")
    records_iter, stats = iter_mixed_records(in_path)

    logging.info(f"Detected format: {stats['format']} (ijson={'yes' if stats['used_ijson'] else 'no'})")
    logging.info(f"Array items: {stats.get('array_items', 0)}; NDJSON lines: {stats.get('ndjson_lines', 0)}; "
                 f"NDJSON objects: {stats.get('ndjson_objects', 0)}; invalid lines: {stats.get('invalid_lines', 0)}")

    # Write out the clean NDJSON
    logging.info(f"Writing NDJSON to: {out_path} (dedupe={args.dedupe}, key={args.key})")
    write_ndjson(records_iter, out_path, key=args.key, dedupe=args.dedupe)
    logging.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
