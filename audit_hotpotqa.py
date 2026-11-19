"""Audit HotpotQA Distractor for a 10-document / 2-positive reranking study.

READ-ONLY for the raw dataset and existing train/val/test files.

Default experiment targets:
    nested Train = 2,000 / 10,000 / 40,000 questions
    Val = 3,000 questions, held out from the OFFICIAL train split
    Test = 7,000 questions from the OFFICIAL validation split

Usage (from your project's root, with the project .venv active):
    python audit_hotpotqa.py
    python audit_hotpotqa.py --export-eligible-ids

This audits eligibility and counts; it does not create the experiment splits.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "datasets" / "raw" / "hotpotqa_distractor"
DEFAULT_REPORT = ROOT / "outputs" / "data_audit" / "hotpotqa_eligibility_report.json"


def positive_title_count(row: dict[str, Any]) -> int | None:
    try:
        titles = row["supporting_facts"]["title"]
        if not isinstance(titles, (list, tuple)):
            return None
        return len(set(titles))
    except (KeyError, TypeError, ValueError):
        return None


def inspect_row(row: dict[str, Any]) -> tuple[str | None, tuple[str, ...]]:
    """Return (exclusion_reason, annotation_warnings).

    Eligibility matches the *intent* of the project's title-based labeling:
    10 DISTINCT candidate paragraphs and exactly 2 DISTINCT gold paragraphs.
    Do not require two supporting sentences: several sentences can refer to
    the same paragraph.
    """
    context = row.get("context")
    facts = row.get("supporting_facts")
    if not isinstance(context, dict) or not isinstance(facts, dict):
        return "missing_or_malformed_context_or_facts", ()

    titles = context.get("title")
    paragraphs = context.get("sentences")
    support_titles = facts.get("title")
    support_sentence_ids = facts.get("sent_id")

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
    gold_titles = set(support_titles)
    if len(gold_titles) != 2:
        return "distinct_positive_count_not_2", ()
    if not gold_titles.issubset(set(titles)):
        return "gold_title_missing_from_candidates", ()

    warnings: list[str] = []
    if not isinstance(support_sentence_ids, (list, tuple)) or len(support_titles) != len(support_sentence_ids):
        warnings.append("supporting_fact_sentence_id_count_mismatch")
    else:
        title_to_paragraph = dict(zip(titles, paragraphs))
        for title, sent_id in zip(support_titles, support_sentence_ids):
            sentence_list = title_to_paragraph[title]
            if (not isinstance(sentence_list, (list, tuple))
                    or not isinstance(sent_id, int)
                    or not (0 <= sent_id < len(sentence_list))):
                warnings.append("invalid_supporting_sentence_reference")
                break
    if any(not isinstance(p, (list, tuple)) or not any(isinstance(s, str) and s.strip() for s in p)
           for p in paragraphs):
        warnings.append("empty_or_malformed_paragraph")

    return None, tuple(warnings)


def audit_split(split: Any, split_name: str) -> dict[str, Any]:
    """Single-pass scan, retaining IDs (not document texts) for cross-split checks."""
    rejected: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    docs_hist: Counter[str] = Counter()
    gold_hist: Counter[str] = Counter()
    type_hist: Counter[str] = Counter()
    level_hist: Counter[str] = Counter()
    seen_ids: set[str] = set()
    valid_ids: list[str] = []
    duplicate_id_count = 0

    total = len(split)
    print(f"\nScanning official {split_name}: {total:,} rows", flush=True)
    for index, row in enumerate(split):
        qid = row.get("id")
        if not isinstance(qid, str) or not qid.strip():
            rejected["missing_question_id"] += 1
            continue
        if qid in seen_ids:
            duplicate_id_count += 1
            rejected["duplicate_question_id"] += 1
            continue
        seen_ids.add(qid)

        context = row.get("context")
        titles = context.get("title") if isinstance(context, dict) else None
        docs_hist[str(len(titles)) if isinstance(titles, (list, tuple)) else "unknown"] += 1
        ngold = positive_title_count(row)
        gold_hist[str(ngold) if ngold is not None else "unknown"] += 1

        reason, row_warnings = inspect_row(row)
        if reason is not None:
            rejected[reason] += 1
        else:
            valid_ids.append(qid)
            type_hist[str(row.get("type", "unknown"))] += 1
            level_hist[str(row.get("level", "unknown"))] += 1
            warnings.update(row_warnings)
        if (index + 1) % 20000 == 0:
            print(f"  {index + 1:,}/{total:,} scanned; {len(valid_ids):,} eligible", flush=True)

    return {
        "split": split_name,
        "total_rows": total,
        "unique_question_ids": len(seen_ids),
        "duplicate_id_rows": duplicate_id_count,
        "eligible_count": len(valid_ids),
        "ineligible_count": total - len(valid_ids),
        "eligible_ratio": round(len(valid_ids) / total, 6) if total else 0,
        "rejected_by_first_reason": dict(sorted(rejected.items())),
        "annotation_warnings_on_eligible": dict(sorted(warnings.items())),
        "candidate_document_count_distribution": dict(sorted(docs_hist.items(), key=lambda x: x[0])),
        "distinct_supporting_title_count_distribution": dict(sorted(gold_hist.items(), key=lambda x: x[0])),
        "eligible_question_type_distribution": dict(sorted(type_hist.items())),
        "eligible_question_level_distribution": dict(sorted(level_hist.items())),
        "_all_ids": seen_ids,
        "_eligible_ids": valid_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                        help="Project-local DatasetDict created by download_dataset.py")
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[2000, 10000, 40000],
                        help="Nested train sizes; only the LARGEST counts toward capacity")
    parser.add_argument("--val-size", type=int, default=3000,
                        help="Val examples reserved from the official TRAIN split")
    parser.add_argument("--test-size", type=int, default=7000,
                        help="Required held-out examples from the official VALIDATION split")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--export-eligible-ids", action="store_true",
                        help="Also save eligible question IDs in separate text files")
    args = parser.parse_args()
    if not args.train_sizes or any(n <= 0 for n in args.train_sizes):
        parser.error("--train-sizes must be positive integers")
    if args.val_size <= 0 or args.test_size <= 0:
        parser.error("--val-size and --test-size must be positive")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        parser.error(f"Raw dataset not found: {dataset_dir}\nRun python download_dataset.py first.")

    try:
        from datasets import load_from_disk
    except ImportError:
        parser.error("Missing dependency 'datasets'. Activate your project .venv and install requirements.txt")

    raw = load_from_disk(str(dataset_dir))
    if "train" not in raw or "validation" not in raw:
        parser.error("Expected a DatasetDict with both 'train' and 'validation' splits")

    train = audit_split(raw["train"], "train")
    test = audit_split(raw["validation"], "validation")
    shared_ids = train["_all_ids"] & test["_all_ids"]

    largest_train = max(args.train_sizes)
    required_official_train = largest_train + args.val_size
    enough_train = train["eligible_count"] >= required_official_train
    enough_test = test["eligible_count"] >= args.test_size
    no_leakage = len(shared_ids) == 0
    no_duplicate_ids = train["duplicate_id_rows"] == 0 and test["duplicate_id_rows"] == 0
    feasible = enough_train and enough_test and no_leakage and no_duplicate_ids

    capacity = {
        "nested_train_sizes": sorted(set(args.train_sizes)),
        "val_size_from_official_train": args.val_size,
        "test_size_from_official_validation": args.test_size,
        "required_official_train_eligible": required_official_train,
        "available_official_train_eligible": train["eligible_count"],
        "available_official_validation_eligible": test["eligible_count"],
        "can_make_nested_train_and_val": enough_train,
        "can_make_test": enough_test,
        "cross_official_split_duplicate_id_count": len(shared_ids),
        "no_cross_split_question_id_overlap": no_leakage,
        "no_duplicate_ids_within_official_splits": no_duplicate_ids,
        "can_use_entire_official_validation": test["eligible_count"] == test["total_rows"],
        "overall_feasible": feasible,
    }

    print("\n" + "=" * 68)
    print("HOTPotQA ELIGIBILITY / CAPACITY AUDIT")
    print("Filter: 10 DISTINCT candidate titles + 2 DISTINCT gold titles")
    print("=" * 68)
    for entry, label in ((train, "Official TRAIN"), (test, "Official VALIDATION")):
        print(f"{label}: {entry['eligible_count']:,}/{entry['total_rows']:,} eligible "
              f"({entry['eligible_ratio'] * 100:.2f}%)")
        print(f"  Rejections: {entry['rejected_by_first_reason']}")
        print(f"  Question types (eligible): {entry['eligible_question_type_distribution']}")
        if entry["annotation_warnings_on_eligible"]:
            print(f"  Annotation warnings: {entry['annotation_warnings_on_eligible']}")

    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print("\nRequested experiment:")
    print(f"  Nested TRAIN sizes: {', '.join(f'{n:,}' for n in sorted(set(args.train_sizes)))}")
    print(f"  VAL (from official TRAIN): {args.val_size:,}")
    print(f"  TEST (from official VALIDATION): {args.test_size:,}")
    print(f"  TRAIN+VAL requires {required_official_train:,} DISTINCT eligible official-train questions: "
          f"{verdict(enough_train)}")
    print(f"  TEST requires {args.test_size:,} eligible official-validation questions: "
          f"{verdict(enough_test)}")
    print(f"  No overlap between official split question IDs: {verdict(no_leakage)}")
    print(f"  No duplicate question IDs within either split: {verdict(no_duplicate_ids)}")
    print(f"  Entire official Validation usable without filtering: "
          f"{verdict(capacity['can_use_entire_official_validation'])}")
    print(f"\nOVERALL FEASIBILITY: {verdict(feasible)}")
    print("Note: This proves COUNT feasibility, not balanced sampling or downstream model accuracy.")

    # Only JSON-serializable summaries are saved. Existing data and results are never overwritten.
    for result in (train, test):
        del result["_all_ids"]
    train_eligible_ids = train.pop("_eligible_ids")
    test_eligible_ids = test.pop("_eligible_ids")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "raw_dataset_dir": str(dataset_dir),
        "filter_definition": (
            "Exactly 10 context titles, 10 unique titles, matching sentences lists, "
            "exactly 2 distinct supporting-fact titles, and both gold titles in context. "
            "Supporting-fact sentence validity and paragraph emptiness are warnings only."
        ),
        "train_split": train,
        "official_validation_split": test,
        "capacity": capacity,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary report: {args.report.resolve()}")
    if args.export_eligible_ids:
        for label, ids in (("official_train", train_eligible_ids),
                           ("official_validation", test_eligible_ids)):
            path = args.report.parent / f"eligible_ids_{label}.txt"
            path.write_text("\n".join(ids) + "\n", encoding="utf-8")
            print(f"Eligible IDs:    {path.resolve()}")
    return 0 if feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
