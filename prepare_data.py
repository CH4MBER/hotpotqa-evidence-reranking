"""Clean, reproducible HotpotQA Distractor Train / Val / Test splits.

Default outputs under Project/data/splits:
  train_2k.jsonl / train_10k.jsonl / train_40k.jsonl (NESTED)
  val.jsonl                                         (3,000, official Train)
  test.jsonl                                        (all eligible official Validation)
  data_splits_manifest.json                         (statistics and hashes)

This is the fresh 2K/10K/40K nested split and fixed 3K Val.
The original legacy files are not used or modified.

Eligibility: 10 DISTINCT candidate titles, exactly 2 DISTINCT gold titles.
Invalid sentence IDs are recorded as warnings but not excluded because our
labels are at DOCUMENT level. The official Validation is held out for Test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "datasets" / "raw" / "hotpotqa_distractor"
DEFAULT_OUT = ROOT / "data" / "splits"


def check_raw(example: dict[str, Any]) -> tuple[str | None, tuple[str, ...]]:
    """Match eligibility rules from the project's audit script."""
    ctx = example.get("context")
    facts = example.get("supporting_facts")
    if not isinstance(ctx, dict) or not isinstance(facts, dict):
        return "missing_or_malformed_context_or_facts", ()
    titles = ctx.get("title")
    paragraphs = ctx.get("sentences")
    support_titles = facts.get("title")
    sent_ids = facts.get("sent_id")
    if not isinstance(titles, (list, tuple)) or not isinstance(paragraphs, (list, tuple)):
        return "malformed_context_lists", ()
    if not isinstance(support_titles, (list, tuple)):
        return "malformed_supporting_titles", ()
    if len(titles) != len(paragraphs):
        return "context_title_sentence_length_mismatch", ()
    if len(titles) != 10:
        return "candidate_count_not_10", ()
    if any(not isinstance(t, str) or not t.strip() for t in titles):
        return "invalid_candidate_title", ()
    if len(set(titles)) != 10:
        return "duplicate_candidate_titles", ()
    if any(not isinstance(t, str) or not t.strip() for t in support_titles):
        return "invalid_gold_title", ()
    positive_titles = set(support_titles)
    if len(positive_titles) != 2:
        return "distinct_positive_count_not_2", ()
    if not positive_titles.issubset(set(titles)):
        return "gold_title_missing_from_candidates", ()

    warnings = []
    if not isinstance(sent_ids, (list, tuple)) or len(support_titles) != len(sent_ids):
        warnings.append("supporting_fact_sentence_id_count_mismatch")
    else:
        by_title = dict(zip(titles, paragraphs))
        for title, sent_id in zip(support_titles, sent_ids):
            sentences = by_title[title]
            if (not isinstance(sentences, (list, tuple))
                    or not isinstance(sent_id, int)
                    or not (0 <= sent_id < len(sentences))):
                warnings.append("invalid_supporting_sentence_reference")
                break
    if any(not isinstance(p, (list, tuple)) or not any(isinstance(s, str) and s.strip() for s in p)
           for p in paragraphs):
        warnings.append("empty_or_malformed_paragraph")
    return None, tuple(warnings)


def normalize(example: dict[str, Any]) -> dict[str, Any]:
    """Keep document schema / text construction identical to original prepare_data.py."""
    titles = example["context"]["title"]
    positive = set(example["supporting_facts"]["title"])
    documents = []
    for i, (title, sentences) in enumerate(zip(titles, example["context"]["sentences"])):
        # A malformed paragraph is an audit warning, not a false gold label.
        paragraph = " ".join(str(s).strip() for s in sentences).strip() if isinstance(sentences, (list, tuple)) else ""
        documents.append({
            "doc_id": i,
            "title": title,
            "text": f"Title: {title}\n{paragraph}",
            "label": int(title in positive),
        })
    return {
        "id": example["id"],
        "question": example["question"],
        "type": example["type"],
        "documents": documents,
    }


def check_prepared(rows: list[dict], description: str) -> set[str]:
    ids: set[str] = set()
    for i, row in enumerate(rows):
        qid = row.get("id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"{description}[{i}]: missing question ID")
        if qid in ids:
            raise ValueError(f"{description}: duplicate ID {qid}")
        ids.add(qid)
        docs = row.get("documents")
        if not isinstance(docs, list) or len(docs) != 10:
            raise ValueError(f"{description}[{qid}]: expected 10 documents")
        titles = [d.get("title") for d in docs]
        if len(set(titles)) != 10:
            raise ValueError(f"{description}[{qid}]: candidate titles not unique")
        if [d.get("doc_id") for d in docs] != list(range(10)):
            raise ValueError(f"{description}[{qid}]: document indices not 0..9")
        if sum(d.get("label") == 1 for d in docs) != 2:
            raise ValueError(f"{description}[{qid}]: expected exactly 2 gold documents")
        if any(d.get("label") not in (0, 1) for d in docs):
            raise ValueError(f"{description}[{qid}]: invalid binary label")
    return ids


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON: {path}:{line_no}") from e
    return rows


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_ids(rows: list[dict]) -> str:
    ordered = "\n".join(str(r["id"]) for r in rows) + "\n"
    return hashlib.sha256(ordered.encode("utf-8")).hexdigest()


def type_counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(str(r.get("type", "unknown")) for r in rows).items()))


def short_name(n: int) -> str:
    return f"{n//1000}k" if n % 1000 == 0 else str(n)


def allocate(n: int, capacities: dict[str, int]) -> dict[str, int]:
    """Near-proportional type-stratified allocation, with exact total n."""
    total = sum(capacities.values())
    if n < 0 or n > total:
        raise ValueError(f"Requested {n:,} rows but only {total:,} available")
    fractions = {k: n * v / total for k, v in capacities.items()}
    assigned = {k: min(v, int(fractions[k])) for k, v in capacities.items()}
    for _ in range(n - sum(assigned.values())):
        available = [k for k in capacities if assigned[k] < capacities[k]]
        best = max(available, key=lambda k: (fractions[k] - assigned[k], k))
        assigned[best] += 1
    return assigned


def draw_stratified(pools: dict[str, list[int]], n: int, rng) -> list[int]:
    quota = allocate(n, {key: len(items) for key, items in pools.items()})
    selected = []
    for key in sorted(pools):
        take = quota[key]
        if take:
            selected.extend(pools[key][-take:])
            del pools[key][-take:]
    rng.shuffle(selected)
    return selected


def prepare(raw: Any, train_sizes: list[int], val_count: int, seed: int,
            min_test: int = 7000) -> tuple[dict[str, list[dict]], dict]:
    """Build fresh, non-overlapping splits in memory. No filesystem writes."""
    import random
    if not {"train", "validation"}.issubset(set(raw.keys())):
        raise ValueError("DatasetDict must contain official train and validation")
    if train_sizes != sorted(set(train_sizes)) or not train_sizes or train_sizes[0] <= 0:
        raise ValueError("Train sizes must be positive, sorted, and unique")
    if val_count <= 0 or min_test <= 0:
        raise ValueError("Val/Test requirements must be positive")

    rng = random.Random(seed)
    all_ids: set[str] = set()
    rejection: dict[str, Counter] = {}
    annotation_warnings: dict[str, Counter] = {}
    eligible_indices: dict[str, dict[str, list[int]]] = {}

    for split_name in ("train", "validation"):
        rows = raw[split_name]
        pools: dict[str, list[int]] = {}
        skipped: Counter[str] = Counter()
        warnings: Counter[str] = Counter()
        print(f"Auditing official {split_name}: {len(rows):,} rows", flush=True)
        for index, example in enumerate(rows):
            qid = example.get("id")
            if not isinstance(qid, str) or not qid:
                skipped["missing_question_id"] += 1
                continue
            if qid in all_ids:
                raise ValueError(f"DATA LEAKAGE / duplicate ID in official splits: {qid}")
            all_ids.add(qid)
            reason, ws = check_raw(example)
            if reason is not None:
                skipped[reason] += 1
                continue
            warnings.update(ws)
            qtype = str(example.get("type", "unknown"))
            pools.setdefault(qtype, []).append(index)
        rejection[split_name] = skipped
        annotation_warnings[split_name] = warnings
        eligible_indices[split_name] = pools

    train_pool = eligible_indices["train"]
    num_train_eligible = sum(len(v) for v in train_pool.values())
    test_indices = sorted(i for indices in eligible_indices["validation"].values() for i in indices)
    if num_train_eligible < train_sizes[-1] + val_count:
        raise ValueError(f"Official Train has only {num_train_eligible:,} eligible questions; "
                         f"needs {train_sizes[-1] + val_count:,}")
    if len(test_indices) < min_test:
        raise ValueError(f"Official Validation has only {len(test_indices):,} eligible questions; "
                         f"needs {min_test:,}")

    for indices in train_pool.values():
        rng.shuffle(indices)
    # Val is chosen independently from ALL official Train questions, before training.
    val_indices = draw_stratified(train_pool, val_count, rng)
    train_indices: list[int] = []
    last_size = 0
    # Independent type-stratified increments guarantee exact nesting.
    for size in train_sizes:
        train_indices.extend(draw_stratified(train_pool, size - last_size, rng))
        last_size = size

    train = [normalize(raw["train"][i]) for i in train_indices]
    val = [normalize(raw["train"][i]) for i in val_indices]
    test = [normalize(raw["validation"][i]) for i in test_indices]

    train_ids = check_prepared(train, "train")
    val_ids = check_prepared(val, "val")
    test_ids = check_prepared(test, "test")
    if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
        raise ValueError("DATA LEAKAGE: overlapping train/val/test IDs")
    files = {f"train_{short_name(n)}.jsonl": train[:n] for n in train_sizes}
    files.update({"val.jsonl": val, "test.jsonl": test})
    manifest = {
        "dataset": "hotpotqa/hotpot_qa", "config": "distractor", "seed": seed,
        "fresh_split": True, "nested_train": True, "train_sizes": train_sizes,
        "val_count": len(val), "test_count": len(test),
        "test_source": "all eligible official validation, HELD OUT",
        "val_source": "official train, disjoint from train groups",
        "sampling": "random, seeded, question-type stratified",
        "eligibility": "10 DISTINCT candidates and 2 DISTINCT gold document titles",
        "annotation_warning_policy": "report sentence-index anomalies; document labels remain valid",
        "official_counts": {
            "train_total": len(raw["train"]), "train_eligible": num_train_eligible,
            "validation_total": len(raw["validation"]), "validation_eligible": len(test),
        },
        "skipped": {k: dict(sorted(v.items())) for k, v in rejection.items()},
        "annotation_warnings": {k: dict(sorted(v.items())) for k, v in annotation_warnings.items()},
        "files": {name: {
            "count": len(rows), "ordered_question_ids_sha256": digest_ids(rows),
            "type_distribution": type_counts(rows),
        } for name, rows in files.items()},
        "all_split_id_overlap_counts": {"train_val": 0, "train_test": 0, "val_test": 0},
        "legacy_files_changed": False,
        "warning": "Do NOT use test.jsonl for model selection or hyperparameter tuning",
    }
    return files, manifest


def save_safely(out: Path, files: dict[str, list[dict]], manifest: dict) -> None:
    """Stage, compare, then write; never overwrite different existing outputs."""
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hotpot_prepare_", dir=str(out)) as tmpname:
        tmp = Path(tmpname)
        filenames = {}
        for name, rows in files.items():
            staged = tmp / name
            with staged.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            hash_value = digest_file(staged)
            manifest["files"][name]["sha256"] = hash_value
            filenames[name] = staged

        manifest_file = tmp / "data_splits_manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        filenames[manifest_file.name] = manifest_file

        # Check ALL collisions before moving even the first output.
        for name, source in filenames.items():
            target = out / name
            if target.exists() and digest_file(target) != digest_file(source):
                raise FileExistsError(
                    f"Refusing to change existing file {target}. Back it up / move it first if you intended a new split."
                )

        for name, source in filenames.items():
            target = out / name
            if target.exists():
                print(f"  unchanged: {name}")
            else:
                os.replace(source, target)
                n = manifest["files"].get(name, {}).get("count", "")
                print(f"  created:   {name} {f'({n:,} questions)' if isinstance(n, int) else ''}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--train-sizes", type=int, nargs="+", default=[2000, 10000, 40000])
    p.add_argument("--val-count", type=int, default=3000)
    p.add_argument("--min-test", type=int, default=7000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    sizes = sorted(set(args.train_sizes))
    if any(n <= 0 for n in sizes) or args.val_count <= 0 or args.min_test <= 0:
        p.error("Train sizes, Val size and minimum Test size must all be positive")
    if not args.dataset_dir.is_dir():
        p.error(f"Local raw dataset missing: {args.dataset_dir}")
    from datasets import load_from_disk
    raw = load_from_disk(str(args.dataset_dir))
    print(f"New nested Train: {sizes}; Val: {args.val_count:,}; Test: all eligible official Validation")
    files, manifest = prepare(raw, sizes, args.val_count, args.seed, args.min_test)
    print("Writing verified splits (never overwrites conflicting existing files)...")
    save_safely(args.out_dir, files, manifest)
    print("\nFINAL DATA SPLITS:")
    for name, rows in files.items():
        print(f"  {name:<24} {len(rows):>7,}  {type_counts(rows)}")
    print("  Train / Val / Test question-ID overlap: 0 (PASS)")
    print("  Manifest:", args.out_dir / "data_splits_manifest.json")
    print("  Test is held out: do not use it before final model selection.")


if __name__ == "__main__":
    main()
