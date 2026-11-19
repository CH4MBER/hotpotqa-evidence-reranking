"""Reusable paragraph-ranking evaluation for the HotpotQA distractor project.

Expected gold JSONL format is produced by prepare_data.py:
{"id": ..., "question": ..., "documents":
    [{"doc_id": 0, "title": ..., "text": ..., "label": 0 or 1}, ...]}

Prediction JSONL format: {"id": ..., "scores": [score_for_doc0, ...]}
Scores MUST correspond to the same document order as in the gold JSONL.

All metrics are question-level macro-averages over examples. For this
project each question has exactly two gold supporting paragraphs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent
METRIC_NAMES = ("recall_at_2", "paragraph_em_at_2", "ndcg_at_5")


def load_jsonl(path: Path) -> list[dict]:
    """Read JSON Lines and report invalid lines with their source location."""
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find input file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def validate_example(example: dict) -> None:
    """Enforce the exact experimental setup used during preparation."""
    if "id" not in example or "question" not in example or "documents" not in example:
        raise ValueError("Example missing id/question/documents")
    docs = example["documents"]
    if len(docs) != 10:
        raise ValueError(f"Example {example['id']}: expected 10 documents, got {len(docs)}")
    ids = [doc["doc_id"] for doc in docs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Example {example['id']}: duplicate doc_id")
    labels = [doc["label"] for doc in docs]
    if any(label not in (0, 1) for label in labels) or sum(labels) != 2:
        raise ValueError(f"Example {example['id']}: expected exactly 2 positive labels")
    if not isinstance(example["question"], str):
        raise ValueError(f"Example {example['id']}: invalid question")
    if any(not isinstance(doc.get("text"), str) for doc in docs):
        raise ValueError(f"Example {example['id']}: a document is missing text")


def evaluate_ranking(example: dict, scores: list[float]) -> dict:
    """Evaluate one question; higher score means a higher ranking.

    Ties are deterministically broken by the original document position.
    """
    validate_example(example)
    docs = example["documents"]
    if len(scores) != len(docs):
        raise ValueError(f"Example {example['id']}: {len(scores)} scores for {len(docs)} documents")
    converted = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in converted):
        raise ValueError(f"Example {example['id']}: scores must be finite")

    ranked_positions = sorted(range(len(docs)), key=lambda pos: (-converted[pos], pos))
    ranked_doc_ids = [docs[pos]["doc_id"] for pos in ranked_positions]
    positive_doc_ids = [doc["doc_id"] for doc in docs if doc["label"] == 1]

    top_2 = set(ranked_doc_ids[:2])
    positives = set(positive_doc_ids)
    recall_at_2 = len(top_2 & positives) / len(positives)
    paragraph_em_at_2 = int(top_2 == positives)

    dcg_5 = sum(
        docs[pos]["label"] / math.log2(rank + 2)
        for rank, pos in enumerate(ranked_positions[:5])
    )
    ideal_dcg_5 = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(min(5, len(positives)))
    )
    ndcg_at_5 = dcg_5 / ideal_dcg_5

    return {
        "id": example["id"],
        "type": example.get("type"),
        "scores": converted,
        "ranked_doc_ids": ranked_doc_ids,
        "positive_doc_ids": positive_doc_ids,
        "recall_at_2": recall_at_2,
        "paragraph_em_at_2": paragraph_em_at_2,
        "ndcg_at_5": ndcg_at_5,
    }


def summarize(results: list[dict]) -> dict:
    """Macro-average every metric (not a corpus-pooled calculation)."""
    if not results:
        raise ValueError("Cannot summarize empty evaluation results")
    return {"num_questions": len(results), **{
        metric: mean(float(row[metric]) for row in results)
        for metric in METRIC_NAMES
    }}


def evaluate_dataset(examples: list[dict], scores_by_id: dict[str, list[float]]) -> tuple[list[dict], dict]:
    gold_ids = [row["id"] for row in examples]
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Duplicate gold example IDs")
    missing = set(gold_ids) - set(scores_by_id)
    extra = set(scores_by_id) - set(gold_ids)
    if missing or extra:
        raise ValueError(f"Prediction ID mismatch: {len(missing)} missing, {len(extra)} extra")
    results = [evaluate_ranking(row, scores_by_id[row["id"]]) for row in examples]
    return results, summarize(results)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate precomputed 10-document ranking scores")
    parser.add_argument("--gold", type=Path, default=PROJECT_ROOT / "data" / "splits" / "val.jsonl")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="JSONL rows with id and scores (ordered like the gold documents)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to save the JSON summary")
    args = parser.parse_args()

    gold = load_jsonl(args.gold)
    predictions = load_jsonl(args.predictions)
    scores_by_id = {}
    for row in predictions:
        if "id" not in row or "scores" not in row:
            raise ValueError("Each prediction must contain id and scores")
        if row["id"] in scores_by_id:
            raise ValueError(f"Duplicate prediction id: {row['id']}")
        scores_by_id[row["id"]] = row["scores"]

    _results, summary = evaluate_dataset(gold, scores_by_id)
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("Saved summary:", args.output)


if __name__ == "__main__":
    main()
