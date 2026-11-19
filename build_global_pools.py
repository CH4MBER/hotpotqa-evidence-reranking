"""Build deduplicated global-negative document pools from existing HotpotQA train JSONL.

No models are loaded and no train/val/test split is changed.

Default inputs:  data/splits/train_2k.jsonl, train_10k.jsonl, train_40k.jsonl
Default outputs: data/global_pools/pool_2k.jsonl, pool_10k.jsonl, pool_40k.jsonl
                 data/global_pools/pools_manifest.json

Each unique Wikipedia page TITLE is represented by the first encountered
paragraph text from its training split. A stable page ID is derived from the
normalized title. At sampling time train.py MUST exclude all 10 titles in the
current question (and ideally check additional false-negative risks).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield obj


def canonical_title(title: str) -> str:
    """Normalization for page identity / self-question candidate exclusion."""
    return " ".join(unicodedata.normalize("NFKC", title).split()).casefold()


def page_id(key: str) -> str:
    return "page_" + hashlib.sha256(key.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_one(path: Path, expected_n: int) -> tuple[list[dict], set[str], dict]:
    """Read source training file; one record per canonical title, deterministic order."""
    seen_questions: set[str] = set()
    pages: dict[str, dict] = {}
    alternate_texts = 0
    repeated_mentions = 0
    within_question_canonical_collisions = 0
    type_counter: Counter[str] = Counter()

    for row in read_jsonl(path):
        qid = row.get("id")
        question = row.get("question")
        docs = row.get("documents")
        if not isinstance(qid, str) or not qid or qid in seen_questions:
            raise ValueError(f"Missing/duplicate question ID in {path}: {qid!r}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question text for {qid} in {path}")
        if not isinstance(docs, list) or len(docs) != 10:
            raise ValueError(f"{path}: question {qid} must have exactly 10 documents")
        seen_questions.add(qid)
        type_counter[str(row.get("type", "unknown"))] += 1
        seen_in_question: set[str] = set()
        seen_raw_titles: set[str] = set()
        positive_count = 0
        doc_ids: set[int] = set()
        for doc in docs:
            if not isinstance(doc, dict):
                raise ValueError(f"Malformed document in {qid}")
            title, body = doc.get("title"), doc.get("text")
            label, local_id = doc.get("label"), doc.get("doc_id")
            if not isinstance(title, str) or not title.strip() or not isinstance(body, str) or not body.strip():
                raise ValueError(f"Missing title/text in question {qid}")
            if label not in (0, 1) or not isinstance(local_id, int) or local_id not in range(10):
                raise ValueError(f"Invalid label/doc_id for question {qid}")
            if local_id in doc_ids:
                raise ValueError(f"Duplicate local doc_id in question {qid}")
            doc_ids.add(local_id)
            positive_count += int(label == 1)
            # The official split validator checks uniqueness of ORIGINAL titles.
            # Canonical-title collisions (case/spacing/Unicode differences) are
            # expected when building a deduplicated global pool and should NOT
            # invalidate a previously verified training question.
            if title in seen_raw_titles:
                raise ValueError(f"Duplicate original candidate title in question {qid}: {title!r}")
            seen_raw_titles.add(title)
            key = canonical_title(title)
            if not key:
                raise ValueError(f"Empty normalized candidate title in question {qid}: {title!r}")
            if key in seen_in_question:
                within_question_canonical_collisions += 1
            else:
                seen_in_question.add(key)
            if key in pages:
                repeated_mentions += 1
                if pages[key]["text"] != body:
                    alternate_texts += 1
                continue
            pages[key] = {
                "doc_id": page_id(key),
                "title": title,
                "text": body,
                "source_question_id": qid,
                "source_doc_id": local_id,
            }
        if positive_count != 2:
            raise ValueError(f"{path}: question {qid} must have exactly two positive documents")
    if len(seen_questions) != expected_n:
        raise ValueError(f"{path}: expected {expected_n:,} questions, got {len(seen_questions):,}")
    stats = {
        "question_count": len(seen_questions),
        "candidate_mentions": 10 * len(seen_questions),
        "unique_pages": len(pages),
        "repeated_title_mentions": repeated_mentions,
        "within_question_canonical_collisions": within_question_canonical_collisions,
        "same_title_different_text_mentions": alternate_texts,
        "question_type_distribution": dict(sorted(type_counter.items())),
    }
    return list(pages.values()), seen_questions, stats


def tag(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--splits-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--sizes", type=int, nargs="+", default=[2000, 10000, 40000])
    args = p.parse_args()
    sizes = sorted(set(args.sizes))
    if not sizes or any(n <= 0 for n in sizes):
        p.error("--sizes must contain positive integers")
    output_dir = args.output_dir or (args.data_dir / "global_pools")
    splits_dir = args.splits_dir or args.data_dir / "splits"

    # Compute in a temporary location and refuse to replace any differing outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build_pools_", dir=output_dir) as temporary:
        stage = Path(temporary)
        metadata = {}
        staged_files = {}
        previous_questions: set[str] = set()
        previous_pages: dict[str, tuple[str, str]] = {}
        for size in sizes:
            label = tag(size)
            source = splits_dir / f"train_{label}.jsonl"
            pages, questions, stats = build_one(source, size)
            if not previous_questions.issubset(questions):
                raise ValueError(f"Train {label} is not nested relative to previous size")
            page_map = {canonical_title(p["title"]): (p["doc_id"], p["text"]) for p in pages}
            for title_key, old in previous_pages.items():
                if page_map.get(title_key) != old:
                    raise ValueError(f"Pool {label} lost/changed a page from the smaller pool")
            previous_questions = questions
            previous_pages = page_map

            name = f"pool_{label}.jsonl"
            target = stage / name
            with target.open("w", encoding="utf-8", newline="\n") as stream:
                for page in pages:
                    stream.write(json.dumps(page, ensure_ascii=False, allow_nan=False) + "\n")
            metadata[label] = {
                "train_file": str(source.resolve()),
                "train_sha256": file_hash(source),
                "pool_file": str((output_dir / name).resolve()),
                "pool_sha256": file_hash(target),
                **stats,
            }
            staged_files[name] = target
            print(f"{label:>4} | {size:>6,} questions | {len(pages):>7,} unique pages | "
                  f"{stats['same_title_different_text_mentions']:,} text variants | "
                  f"{stats['within_question_canonical_collisions']:,} in-question normalized-title collisions")

        manifest = {
            "version": 1,
            "deduplication": "NFKC + whitespace normalization + casefold(title)",
            "representative_text": "first occurrence in corresponding training file",
            "source": "ONLY the matching training split (never Val/Test)",
            "sampling_warning": "Exclude every title in the current question; other false negatives remain possible.",
            "sizes": metadata,
        }
        manifest_path = stage / "pools_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged_files[manifest_path.name] = manifest_path

        # Validate ALL existing outputs before moving ANY files.
        for name, staged in staged_files.items():
            final = output_dir / name
            if final.exists() and file_hash(final) != file_hash(staged):
                raise FileExistsError(f"Existing output differs: {final}. Move/back it up explicitly.")
        for name, staged in staged_files.items():
            final = output_dir / name
            if final.exists():
                print(f"UNCHANGED: {final}")
            else:
                os.replace(staged, final)
                print(f"CREATED:   {final}")
    print("Done. No model training, Test reading, or Negative Mining was performed.")


if __name__ == "__main__":
    main()
