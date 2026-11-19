"""Create nested 2K/10K mining files from the existing 40K mining output.

Default input files in Project/data/splits (Train) and data/negatives (Mining):
  train_2k.jsonl, train_10k.jsonl, train_40k.jsonl,
  mined_train_40k.jsonl
Output files:
  mined_train_2k.jsonl, mined_train_10k.jsonl

Does not run MiniLM, re-mine negatives, alter seeds, or overwrite changed data.
Run from the project directory: python split_mined.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def tag(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def iter_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield obj


def load_train(path: Path, expected_count: int):
    """Keep only metadata, not full paragraph text (40K can be large)."""
    ids = []
    metadata = {}
    for row in iter_jsonl(path):
        qid = str(row.get("id", ""))
        if not qid or qid in metadata:
            raise ValueError(f"Missing or repeated question ID in {path}: {qid!r}")
        docs = row.get("documents")
        if not isinstance(docs, list) or len(docs) != 10:
            raise ValueError(f"{path}: question {qid} must have exactly 10 documents")
        docids = [d["doc_id"] for d in docs]
        positives = [d["doc_id"] for d in docs if d.get("label") == 1]
        if len(set(docids)) != 10 or len(positives) != 2:
            raise ValueError(f"{path}: invalid document IDs or gold labels for {qid}")
        question = row.get("question")
        if not isinstance(question, str) or not question:
            raise ValueError(f"{path}: missing question for {qid}")
        ids.append(qid)
        metadata[qid] = (question, tuple(docids), frozenset(positives))
    if len(ids) != expected_count:
        raise ValueError(f"{path}: expected {expected_count:,} questions, found {len(ids):,}")
    return ids, metadata


def load_mined(path: Path, full_metadata: dict, expected_count: int):
    mined = {}
    for row in iter_jsonl(path):
        qid = str(row.get("id", ""))
        if qid not in full_metadata or qid in mined:
            raise ValueError(f"{path}: unknown or repeated mined question ID {qid!r}")
        question, docids, gold = full_metadata[qid]
        if row.get("question") != question:
            raise ValueError(f"{path}: question text mismatch for {qid}")
        if set(row.get("positive_doc_ids", [])) != gold:
            raise ValueError(f"{path}: positive document mismatch for {qid}")
        negatives = set(docids) - gold
        for key in ("hard_negative_doc_ids", "random_negative_doc_ids"):
            chosen = row.get(key)
            if (not isinstance(chosen, list) or len(chosen) != 2
                    or len(set(chosen)) != 2 or not set(chosen) <= negatives):
                raise ValueError(f"{path}: invalid {key} for {qid}")
        scores = row.get("document_scores")
        if (not isinstance(scores, list) or len(scores) != len(docids)
                or any(not isinstance(x, (int, float)) or not math.isfinite(x)
                       for x in scores)):
            raise ValueError(f"{path}: invalid document_scores for {qid}")
        mined[qid] = row
    if len(mined) != expected_count or set(mined) != set(full_metadata):
        raise ValueError(f"{path}: expected exactly {expected_count:,} matching mined IDs")
    return mined


def stage_output(path: Path, rows: list[dict]) -> Path | None:
    """Create a temporary file only; verify existing files before changing anything."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if path.exists():
            if path.read_bytes() != tmp.read_bytes():
                raise FileExistsError(
                    f"Output already exists with DIFFERENT content: {path}. "
                    "Back it up or delete it deliberately before rerunning."
                )
            tmp.unlink()
            return None  # Already correct; repeat runs are safe.
        return tmp
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Project data root, containing splits/ and negatives/")
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--negatives-dir", type=Path, default=None)
    parser.add_argument("--full-size", type=int, default=40000,
                        help="Size of the original mining set (default: 40000)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2000, 10000],
                        help="Subset training sizes (default: 2000 10000)")
    args = parser.parse_args()
    sizes = sorted(set(args.sizes))
    if not sizes or any(n <= 0 or n >= args.full_size for n in sizes):
        parser.error("Subset sizes must be positive and smaller than --full-size")
    splits = args.splits_dir or args.data_dir / "splits"
    negatives = args.negatives_dir or args.data_dir / "negatives"

    full_path = splits / f"train_{tag(args.full_size)}.jsonl"
    source_path = negatives / f"mined_train_{tag(args.full_size)}.jsonl"
    print(f"Checking full training set: {full_path}")
    _, full_metadata = load_train(full_path, args.full_size)
    print(f"Checking full mining set:   {source_path}")
    mined = load_mined(source_path, full_metadata, args.full_size)

    planned = []
    last_ids = None
    for n in sizes:
        train_path = splits / f"train_{tag(n)}.jsonl"
        ids, metadata = load_train(train_path, n)
        if not set(ids) <= set(full_metadata):
            raise ValueError(f"{train_path}: not a subset of the full training IDs")
        for qid in ids:
            if metadata[qid] != full_metadata[qid]:
                raise ValueError(f"{train_path}: source document/question mismatch for {qid}")
        if last_ids is not None and not last_ids <= set(ids):
            raise ValueError(f"{train_path}: training subsets are not nested")
        last_ids = set(ids)
        destination = negatives / f"mined_train_{tag(n)}.jsonl"
        rows = [mined[qid] for qid in ids]  # Follow train file order, not mining order.
        planned.append((destination, rows))
        print(f"Validated {train_path.name}: {n:,} questions, all found in 40K mining")

    staged = []
    try:
        for dest, rows in planned:
            staged.append((dest, stage_output(dest, rows), len(rows)))
        for dest, temp, count in staged:
            if temp is None:
                print(f"UNCHANGED: {dest} ({count:,} questions)")
            else:
                temp.replace(dest)
                print(f"CREATED:   {dest} ({count:,} questions)")
    finally:
        for _, temp, _ in staged:
            if temp is not None:
                temp.unlink(missing_ok=True)
    print("Done. No MiniLM inference or model training was performed.")


if __name__ == "__main__":
    main()
