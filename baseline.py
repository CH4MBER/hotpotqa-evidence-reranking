"""Evaluate BM25 and frozen pretrained MiniLM on the SAME HotpotQA validation split.

Default usage from the project root:
    python baseline.py

Only inference is performed. No model parameters are updated.
Outputs go to outputs/baselines/ and can be re-evaluated with evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
# Match mine_negatives.py's project-local Hugging Face cache location.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".cache" / "sentence_transformers"))

from evaluate import evaluate_ranking, load_jsonl, summarize, validate_example, write_jsonl

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    """Simple deterministic word tokens for the BM25 lexical baseline."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def score_bm25(examples: list[dict]) -> list[list[float]]:
    """BM25 reranks ONLY the ten provided candidates for each question."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise SystemExit("Install BM25 dependency: python -m pip install rank-bm25") from exc

    scores = []
    for row in examples:
        # The prepared doc['text'] already includes its title; do not prepend again.
        corpus = [tokenize(doc["text"]) for doc in row["documents"]]
        query = tokenize(row["question"])
        bm25 = BM25Okapi(corpus)
        scores.append([float(value) for value in bm25.get_scores(query)])
    return scores


def score_minilm(examples: list[dict], model_name: str, device: str,
                 batch_size: int, max_length: int) -> list[list[float]]:
    import torch
    from sentence_transformers import CrossEncoder

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    print(f"Loading CrossEncoder: {model_name} on {device}")
    print(f"max_length={max_length}, batch_size={batch_size}")
    model = CrossEncoder(model_name, device=device, max_length=max_length)

    # Identical question-document construction to mine_negatives.py.
    pairs = [
        (row["question"], doc["text"])
        for row in examples
        for doc in row["documents"]
    ]
    # CrossEncoder.predict is inference only; it does not train or backpropagate.
    with torch.inference_mode():
        flat = np.asarray(model.predict(pairs, batch_size=batch_size,
                                        show_progress_bar=True)).reshape(-1)
    if len(flat) != len(pairs):
        raise ValueError(f"Expected {len(pairs)} model scores, got {len(flat)}")

    scores = []
    offset = 0
    for row in examples:
        size = len(row["documents"])
        scores.append([float(value) for value in flat[offset:offset + size]])
        offset += size
    return scores


def save_experiment(key: str, examples: list[dict], scores: list[list[float]],
                    output_dir: Path, dataset_stem: str) -> dict:
    if len(examples) != len(scores):
        raise ValueError("Number of examples and score groups must match")
    results = [evaluate_ranking(row, prediction)
               for row, prediction in zip(examples, scores)]
    summary = summarize(results)
    # Keep score arrays in exactly the order of the gold input documents.
    predictions = [{"id": row["id"], "scores": row["scores"]} for row in results]
    prediction_path = output_dir / f"{key}_{dataset_stem}_predictions.jsonl"
    details_path = output_dir / f"{key}_{dataset_stem}_details.jsonl"
    write_jsonl(prediction_path, predictions)
    write_jsonl(details_path, results)
    print(f"  {key}: Recall@2={summary['recall_at_2']:.2%}, "
          f"EM@2={summary['paragraph_em_at_2']:.2%}, "
          f"nDCG@5={summary['ndcg_at_5']:.2%}")
    print("    Predictions:", prediction_path)
    print("    Details:    ", details_path)
    return {**summary,
            "predictions_file": str(prediction_path),
            "details_file": str(details_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 / original MiniLM validation-set baselines")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "splits" / "val.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "baselines_val3k")
    parser.add_argument("--model", choices=("both", "bm25", "minilm"), default="both")
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
                        help="Pretrained CrossEncoder Hugging Face model ID")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("batch-size and max-length must be positive")
    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")

    examples = load_jsonl(args.input)
    for row in examples:
        validate_example(row)
    print(f"Loaded {len(examples)} questions from {args.input}")
    print("Each question is ranked against all 10 candidate documents.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    if args.model in ("both", "bm25"):
        print("\n[1] BM25 (lexical reranking baseline)")
        bm25_scores = score_bm25(examples)
        summaries["bm25"] = save_experiment("bm25", examples, bm25_scores,
                                            args.output_dir, args.input.stem)
    if args.model in ("both", "minilm"):
        print("\n[2] CrossEncoder inference (NO TRAINING)")
        minilm_scores = score_minilm(examples, args.model_name,
                                    args.device, args.batch_size, args.max_length)
        summaries["pretrained_minilm"] = save_experiment(
            "pretrained_minilm", examples, minilm_scores,
            args.output_dir, args.input.stem)

    # Preserve the other baseline if BM25 and MiniLM are run separately.
    # Running --model both always creates a fresh, complete summary.
    summary_path = args.output_dir / f"summary_{args.input.stem}.json"
    previous = {}
    if args.model != "both" and summary_path.is_file():
        candidate = json.loads(summary_path.read_text(encoding="utf-8"))
        if candidate.get("dataset") == str(args.input):
            previous = candidate
            summaries = {**previous.get("results", {}), **summaries}

    final = {
        "dataset": str(args.input),
        "num_questions": len(examples),
        "bm25_tokenizer": "lowercase + Unicode word tokens",
        "minilm_model": args.model_name if args.model in ("minilm", "both")
                        else previous.get("minilm_model"),
        "minilm_max_length": args.max_length if args.model in ("minilm", "both")
                             else previous.get("minilm_max_length"),
        "results": summaries,
    }
    summary_path.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print("\nSaved summary:", summary_path)
    print("\nModel                      Recall@2    EM@2     nDCG@5")
    print("-" * 61)
    for key, result in summaries.items():
        print(f"{key:<26} {result['recall_at_2']:>7.2%}  "
              f"{result['paragraph_em_at_2']:>7.2%}  {result['ndcg_at_5']:>7.2%}")
    print("\nAll metrics are macro-averaged over questions.")
    print("No optimizer or model training is used in this script.")


if __name__ == "__main__":
    main()
