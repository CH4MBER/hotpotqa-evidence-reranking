"""Final HELD-OUT Test evaluation for COMS 4705 HotpotQA evidence reranking.

Evaluate each frozen best validation checkpoint ONCE on the same Test split.
Never trains, picks checkpoints using Test, or changes training hyperparameters.

Quick start (from the project root, after all 27 training runs are complete):
  python test_experiments.py --dry-run
  python test_experiments.py
  python test_experiments.py --resume

Requires existing evaluate.py and baseline.py beside this file. Output:
  outputs/test_evaluation/models/<size>/<strategy>/seed_<seed>/
      metrics.json, predictions.jsonl, details.jsonl
  outputs/test_evaluation/baselines/<name>/...
  outputs/test_evaluation/results_by_run.csv
  outputs/test_evaluation/results_mean_sd.csv
  outputs/test_evaluation/results_by_type.csv
  outputs/test_evaluation/FINAL_TEST_RESULTS.md

Before anything is evaluated, the script verifies all selected trained runs,
including the epoch budget, data count, val count, non-debug flag, and best_model.
For reproducibility, completed results are bound to dataset/checkpoint hashes.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ROOT / ".cache" / "sentence_transformers"))

from evaluate import METRIC_NAMES, evaluate_ranking, load_jsonl, summarize, validate_example

SIZES = {"2k": 2_000, "10k": 10_000, "40k": 40_000}
STRATEGIES = ("candidate_random", "hard", "global_random")
DEFAULT_SEEDS = (42, 123, 2026)
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")


@dataclass(frozen=True)
class Run:
    size: str
    strategy: str
    seed: int
    attempt: Path
    checkpoint: Path
    summary_path: Path
    summary: dict[str, Any]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_weight_files(checkpoint: Path) -> list[Path]:
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {checkpoint / 'config.json'}")
    available = [checkpoint / name for name in WEIGHT_FILES if (checkpoint / name).is_file()]
    if not available:
        raise FileNotFoundError(f"Missing saved model weights in {checkpoint}")
    # Index files point to weight shards; include them, so hash changes when weights change.
    if any(p.name.endswith(".index.json") for p in available):
        for index in [p for p in available if p.name.endswith(".index.json")]:
            index_data = read_json(index)
            for filename in sorted(set(index_data.get("weight_map", {}).values())):
                shard = checkpoint / filename
                if not shard.is_file():
                    raise FileNotFoundError(f"Missing model shard: {shard}")
                available.append(shard)
    return sorted(set(available))


def discover_run(model_root: Path, size: str, strategy: str, seed: int,
                 expected_epochs: int, expected_val_questions: int) -> Run:
    run_dir = model_root / size / strategy / f"seed_{seed}"
    marker = run_dir / "completed.json"
    if not marker.is_file():
        raise FileNotFoundError(
            f"No completed.json at {marker}. Only verified finished training runs can be tested."
        )
    completed = read_json(marker)
    for field, expected in (("size", size), ("strategy", strategy), ("seed", seed)):
        if completed.get(field) != expected:
            raise ValueError(f"{marker}: {field}={completed.get(field)!r}, expected {expected!r}")
    attempt_name = completed.get("attempt_dir", "")
    if not isinstance(attempt_name, str) or not attempt_name.startswith("attempt_") or Path(attempt_name).name != attempt_name:
        raise ValueError(f"Invalid attempt_dir in {marker}: {attempt_name!r}")
    attempt = run_dir / attempt_name
    summary_path = attempt / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing training summary: {summary_path}")
    summary = read_json(summary_path)
    fields = {
        "size": size, "strategy": strategy, "seed": seed, "epochs": expected_epochs,
        "train_questions": SIZES[size], "val_questions": expected_val_questions,
        "debug_subset": False,
    }
    for field, expected in fields.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"Training metadata mismatch at {summary_path}: {field}={summary.get(field)!r}; "
                f"expected {expected!r}. Check 3-epoch/pilot/old attempts."
            )
    best_epoch = summary.get("best_epoch")
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= expected_epochs:
        raise ValueError(f"Bad best_epoch in {summary_path}: {best_epoch!r}")
    if not isinstance(summary.get("best_val_metrics"), dict):
        raise ValueError(f"Missing best_val_metrics in {summary_path}")
    checkpoint = attempt / "best_model"
    model_weight_files(checkpoint)
    if not (checkpoint / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Saved tokenizer not found in {checkpoint}")
    return Run(size, strategy, seed, attempt, checkpoint, summary_path, summary)


def discover_all(args: argparse.Namespace) -> list[Run]:
    results = []
    failures = []
    for size in args.sizes:
        for strategy in args.strategies:
            for seed in args.seeds:
                try:
                    results.append(discover_run(args.model_root, size, strategy, seed,
                                                args.expected_epochs, args.expected_val_questions))
                except (ValueError, OSError) as exc:
                    failures.append(f"  {size}/{strategy}/seed_{seed}: {exc}")
    if failures:
        raise RuntimeError(
            f"Preflight FAILED for {len(failures)} of {len(failures) + len(results)} checkpoints; "
            "TEST HAS NOT BEEN OPENED.\n" + "\n".join(failures)
        )
    if not results:
        raise RuntimeError("No experiments selected")
    max_lengths = {r.summary.get("max_length") for r in results}
    base_models = {r.summary.get("model_name") for r in results}
    if len(max_lengths) != 1 or not next(iter(max_lengths)) or len(base_models) != 1:
        raise ValueError("Inconsistent max_length or base model across completed training runs")
    return results


def verify_test_examples(test_file: Path, val_file: Path, train_file: Path | None,
                         expected_count: int) -> list[dict]:
    if not test_file.is_file():
        raise FileNotFoundError(f"Test dataset not found: {test_file}")
    test = load_jsonl(test_file)
    if len(test) != expected_count:
        raise ValueError(f"Expected {expected_count:,} test questions, found {len(test):,}: {test_file}")
    ids = [row["id"] for row in test]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs within Test")
    for row in test:
        validate_example(row)
    test_ids = set(ids)
    for split_name, path in (("Val", val_file), ("Train", train_file)):
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Required {split_name} overlap-check file is missing: {path}")
        seen_ids: set[str] = set()
        # Stream rows rather than loading a potentially large 40K training split into memory.
        with path.open("r", encoding="utf-8") as stream:
            for line_num, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                qid = item["id"]
                if qid in seen_ids:
                    raise ValueError(f"Duplicate {split_name} ID at {path}:{line_num}: {qid}")
                seen_ids.add(qid)
                if qid in test_ids:
                    raise ValueError(f"Test/{split_name} overlap: question {qid!r}; stop before evaluating")
    return test


def select_device_precision(device_requested: str, precision_requested: str) -> tuple[str, bool]:
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_requested == "auto" else device_requested
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    available = device == "cuda" and torch.cuda.is_bf16_supported()
    if precision_requested == "bf16" and not available:
        raise RuntimeError("BF16 requested but this CUDA device does not support BF16")
    bf16 = precision_requested == "bf16" or (precision_requested == "auto" and available)
    return device, bf16


def score_checkpoint(examples: list[dict], checkpoint: str | Path, batch_size: int,
                     max_length: int, device: str, use_bf16: bool,
                     progress_every: int) -> list[list[float]]:
    """Exact train.py validation pair construction and raw-logit scoring semantics."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
    if model.config.num_labels != 1:
        raise ValueError(f"Expected single-logit CrossEncoder, got num_labels={model.config.num_labels}")
    model.to(device)
    model.eval()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    pairs = [(row["question"], doc["text"]) for row in examples for doc in row["documents"]]
    flat_scores: list[float] = []
    total_batches = math.ceil(len(pairs) / batch_size)
    try:
        with torch.inference_mode():
            for batch_idx, offset in enumerate(range(0, len(pairs), batch_size), 1):
                current = pairs[offset:offset + batch_size]
                questions, passages = zip(*current)
                tokens = tokenizer(list(questions), list(passages), padding=True,
                                   truncation=True, max_length=max_length, return_tensors="pt")
                tokens = {key: value.to(device) for key, value in tokens.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(**tokens).logits.reshape(-1)
                flat_scores.extend(logits.float().cpu().tolist())
                if progress_every > 0 and (batch_idx % progress_every == 0 or batch_idx == total_batches):
                    print(f"    inference: {batch_idx}/{total_batches} batches ({batch_idx/total_batches:.0%})", flush=True)
        if len(flat_scores) != len(pairs):
            raise RuntimeError(f"Expected {len(pairs):,} logits; got {len(flat_scores):,}")
        scores: list[list[float]] = []
        pos = 0
        for row in examples:
            count = len(row["documents"])
            scores.append(flat_scores[pos:pos + count])
            pos += count
        return scores
    finally:
        del model, tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def make_details(examples: list[dict], scores: list[list[float]]) -> tuple[list[dict], dict, dict]:
    if len(examples) != len(scores):
        raise ValueError(f"Score group mismatch: {len(examples)} examples, {len(scores)} score groups")
    details = [evaluate_ranking(row, row_scores) for row, row_scores in zip(examples, scores)]
    metrics = summarize(details)
    grouped: dict[str, list[dict]] = {}
    for detail in details:
        kind = detail.get("type") or "unknown"
        grouped.setdefault(str(kind), []).append(detail)
    by_type = {kind: summarize(values) for kind, values in sorted(grouped.items())}
    return details, metrics, by_type


def signature_for_run(run: Run, test_hash: str, max_length: int,
                      device: str, use_bf16: bool) -> str:
    paths = [run.summary_path, run.checkpoint / "config.json", *model_weight_files(run.checkpoint)]
    # The tokenizer is saved with best_model; hash its defining files as well.
    paths += [p for p in run.checkpoint.iterdir() if p.is_file() and p.name in
              {"tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"}]
    payload = {
        "source_type": "finetuned", "size": run.size, "strategy": run.strategy, "seed": run.seed,
        "test_sha256": test_hash, "checkpoint": str(run.checkpoint.resolve()),
        "max_length": max_length, "device": device, "bf16": use_bf16,
        "files_sha256": {p.name: sha256(p) for p in sorted(set(paths))},
        "evaluator_version": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def signature_for_baseline(name: str, test_hash: str, max_length: int,
                           device: str, use_bf16: bool, model_name: str) -> str:
    payload = {"source_type": "baseline", "name": name, "test_sha256": test_hash,
               "max_length": max_length, "device": device, "bf16": use_bf16,
               "model_name": model_name, "evaluator_version": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def output_for_run(output_root: Path, run: Run) -> Path:
    return output_root / "models" / run.size / run.strategy / f"seed_{run.seed}"


def store_evaluation(output: Path, examples: list[dict], details: list[dict],
                     result: dict[str, Any]) -> None:
    # Write metrics LAST; it is the completion marker for --resume.
    predictions = [{"id": d["id"], "scores": d["scores"]} for d in details]
    atomic_jsonl(output / "predictions.jsonl", predictions)
    atomic_jsonl(output / "details.jsonl", details)
    atomic_json(output / "metrics.json", result)


def read_completed(output: Path, fingerprint: str, num_questions: int,
                   resume: bool) -> dict | None:
    summary = output / "metrics.json"
    if not summary.exists():
        return None
    old = read_json(summary)
    if old.get("fingerprint") != fingerprint or old.get("metrics", {}).get("num_questions") != num_questions:
        raise RuntimeError(f"Existing Test results have a different configuration: {summary}. "
                           "Do not overwrite; inspect or choose a new --output-root.")
    if not (output / "predictions.jsonl").is_file() or not (output / "details.jsonl").is_file():
        raise RuntimeError(f"Incomplete result artifacts at {output}; inspect before proceeding")
    if not resume:
        raise RuntimeError(f"Test result already exists: {summary}. Pass --resume to reuse it.")
    return old


def evaluate_model(run: Run, examples: list[dict], test_hash: str,
                   args: argparse.Namespace, device: str, use_bf16: bool) -> dict:
    max_length = int(run.summary["max_length"])
    output = output_for_run(args.output_root, run)
    signature = signature_for_run(run, test_hash, max_length, device, use_bf16)
    cached = read_completed(output, signature, len(examples), args.resume)
    if cached is not None:
        print(f"  SKIP completed Test: {run.size}/{run.strategy}/seed_{run.seed}", flush=True)
        return cached
    started = time.monotonic()
    print(f"  TEST {run.size}/{run.strategy}/seed_{run.seed} | best epoch {run.summary['best_epoch']}", flush=True)
    scored = score_checkpoint(examples, run.checkpoint, args.batch_size, max_length,
                              device, use_bf16, args.progress_every)
    details, metrics, by_type = make_details(examples, scored)
    result = {
        "kind": "finetuned", "size": run.size, "strategy": run.strategy, "seed": run.seed,
        "best_epoch": run.summary["best_epoch"],
        "val_metrics": run.summary["best_val_metrics"],
        "model_checkpoint": str(run.checkpoint.resolve()),
        "training_summary": str(run.summary_path.resolve()),
        "test_file": str(args.test_file.resolve()), "test_sha256": test_hash,
        "max_length": max_length, "device": device, "precision": "bf16" if use_bf16 else "fp32",
        "batch_size": args.batch_size, "fingerprint": signature,
        "metrics": metrics, "metrics_by_type": by_type,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    store_evaluation(output, examples, details, result)
    print(f"    Test Recall@2={metrics['recall_at_2']:.2%} "
          f"EM@2={metrics['paragraph_em_at_2']:.2%} nDCG@5={metrics['ndcg_at_5']:.2%}", flush=True)
    return result


def evaluate_baseline(name: str, examples: list[dict], test_hash: str, args: argparse.Namespace,
                      device: str, use_bf16: bool, model_name: str, max_length: int) -> dict:
    output = args.output_root / "baselines" / name
    signature = signature_for_baseline(name, test_hash, max_length, device, use_bf16, model_name)
    cached = read_completed(output, signature, len(examples), args.resume)
    if cached is not None:
        print(f"  SKIP completed baseline: {name}", flush=True)
        return cached
    print(f"  BASELINE {name}", flush=True)
    started = time.monotonic()
    if name == "bm25":
        from baseline import score_bm25
        scores = score_bm25(examples)
    elif name == "pretrained_minilm":
        scores = score_checkpoint(examples, model_name, args.batch_size, max_length,
                                  device, use_bf16, args.progress_every)
    else:
        raise ValueError(f"Unknown baseline: {name}")
    details, metrics, by_type = make_details(examples, scores)
    result = {
        "kind": "baseline", "name": name, "test_file": str(args.test_file.resolve()),
        "test_sha256": test_hash, "model_name": model_name if name == "pretrained_minilm" else None,
        "max_length": max_length if name == "pretrained_minilm" else None,
        "device": device if name == "pretrained_minilm" else "cpu",
        "precision": "bf16" if use_bf16 and name == "pretrained_minilm" else "fp32",
        "fingerprint": signature, "metrics": metrics, "metrics_by_type": by_type,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    store_evaluation(output, examples, details, result)
    print(f"    Test Recall@2={metrics['recall_at_2']:.2%} "
          f"EM@2={metrics['paragraph_em_at_2']:.2%} nDCG@5={metrics['ndcg_at_5']:.2%}", flush=True)
    return result


def generate_reports(results: list[dict], baselines: list[dict], output_root: Path,
                     test_file: Path) -> None:
    def pct(v: float) -> float:
        return round(100 * v, 4)

    run_rows = []
    type_rows = []
    for entry in results:
        m = entry["metrics"]
        val = entry["val_metrics"]
        row = {"size": entry["size"], "strategy": entry["strategy"], "seed": entry["seed"],
               "best_epoch": entry["best_epoch"], "test_questions": m["num_questions"]}
        for metric in METRIC_NAMES:
            row[f"test_{metric}_pct"] = pct(m[metric])
            row[f"val_{metric}_pct"] = pct(val[metric])
            row[f"test_minus_val_{metric}_pp"] = pct(m[metric] - val[metric])
        run_rows.append(row)
        for kind, group in entry["metrics_by_type"].items():
            type_row = {"size": entry["size"], "strategy": entry["strategy"],
                        "seed": entry["seed"], "type": kind, "questions": group["num_questions"]}
            type_row.update({f"{m}_pct": pct(group[m]) for m in METRIC_NAMES})
            type_rows.append(type_row)
    run_headers = ["size", "strategy", "seed", "best_epoch", "test_questions"] + [
        field for metric in METRIC_NAMES for field in
        (f"test_{metric}_pct", f"val_{metric}_pct", f"test_minus_val_{metric}_pp")]
    type_headers = ["size", "strategy", "seed", "type", "questions"] + [f"{m}_pct" for m in METRIC_NAMES]
    atomic_csv(output_root / "results_by_run.csv", run_headers, run_rows)
    atomic_csv(output_root / "results_by_type.csv", type_headers, type_rows)
    aggregate_rows = []
    by_group: dict[tuple[str, str], list[dict]] = {}
    for item in results:
        by_group.setdefault((item["size"], item["strategy"]), []).append(item)
    for (size, strategy), items in by_group.items():
        row = {"size": size, "strategy": strategy, "n_seeds": len(items),
               "seeds": ",".join(str(e["seed"]) for e in items)}
        for metric in METRIC_NAMES:
            vals = [float(item["metrics"][metric]) * 100 for item in items]
            row[f"{metric}_mean_pct"] = round(statistics.mean(vals), 4)
            row[f"{metric}_sd_pp"] = round(statistics.stdev(vals), 4) if len(vals) >= 2 else 0.0
        aggregate_rows.append(row)
    aggregate_headers = ["size", "strategy", "n_seeds", "seeds"] + [
        field for metric in METRIC_NAMES for field in (f"{metric}_mean_pct", f"{metric}_sd_pp")]
    atomic_csv(output_root / "results_mean_sd.csv", aggregate_headers, aggregate_rows)
    lines = [
        "# Final Held-out Test Results — HotpotQA Evidence Reranking", "",
        f"Test: `{test_file}` ({results[0]['metrics']['num_questions']:,} questions)",
        "", "**All checkpoints were frozen based on Validation Recall@2 before Test scoring.**", "",
        "Metrics are macro-averages. For fine-tuned models: three seeds, mean ± sample SD in percentage points.", "",
        "| Size | Strategy | Recall@2 | Paragraph EM@2 | nDCG@5 |",
        "|---|---|---:|---:|---:|",
    ]
    for b in baselines:
        m = b["metrics"]
        lines.append(f"| — | {b['name']} | {m['recall_at_2']*100:.2f}% | "
                     f"{m['paragraph_em_at_2']*100:.2f}% | {m['ndcg_at_5']*100:.2f}% |")
    for r in aggregate_rows:
        def fmt(metric: str) -> str:
            return f"{r[f'{metric}_mean_pct']:.2f} ± {r[f'{metric}_sd_pp']:.2f}%"
        lines.append(f"| {r['size']} | {r['strategy']} | {fmt('recall_at_2')} | "
                     f"{fmt('paragraph_em_at_2')} | {fmt('ndcg_at_5')} |")
    lines += ["", "Detailed results: `results_by_run.csv` and `results_by_type.csv`.",
              "The Test split must NOT be used to revise training or checkpoint-selection rules.", ""]
    report = output_root / "FINAL_TEST_RESULTS.md"
    temp = report.with_name(report.name + ".tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temp, report)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-root", type=Path, default=ROOT / "outputs" / "experiments")
    p.add_argument("--test-file", type=Path, default=ROOT / "data" / "splits" / "test.jsonl")
    p.add_argument("--val-file", type=Path, default=ROOT / "data" / "splits" / "val.jsonl")
    p.add_argument("--train-file", type=Path, default=ROOT / "data" / "splits" / "train_40k.jsonl",
                   help="Largest nested Train split; used ONLY for Test/Train ID overlap check")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "test_evaluation")
    p.add_argument("--sizes", choices=tuple(SIZES), nargs="+", default=list(SIZES))
    p.add_argument("--strategies", choices=STRATEGIES, nargs="+", default=list(STRATEGIES))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--expected-epochs", type=int, default=10)
    p.add_argument("--expected-val-questions", type=int, default=3_000)
    p.add_argument("--expected-test-questions", type=int, default=7_345)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    p.add_argument("--progress-every", type=int, default=300)
    p.add_argument("--no-baselines", action="store_true", help="Skip BM25 and pretrained MiniLM Test evaluation")
    p.add_argument("--dry-run", action="store_true", help="Validate frozen checkpoints; do NOT open Test or write files")
    p.add_argument("--resume", action="store_true", help="Reuse only verified results; never overwrite completed Test runs")
    return p


def main(argv: list[str] | None = None) -> None:
    p = make_parser()
    args = p.parse_args(argv)
    for name in ("model_root", "test_file", "val_file", "train_file", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    for field in ("batch_size", "expected_epochs", "expected_val_questions", "expected_test_questions"):
        if getattr(args, field) <= 0:
            p.error(f"{field} must be positive")
    if args.progress_every < 0:
        p.error("--progress-every cannot be negative")
    for name in ("sizes", "strategies", "seeds"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            p.error(f"Duplicate {name} values")

    # ABSOLUTE RULE: discover ALL checkpoints first, before opening the Test file.
    runs = discover_all(args)
    print(f"Preflight OK: {len(runs)} frozen best checkpoints; expected {args.expected_epochs} epochs.")
    for run in runs:
        print(f"  {run.size:>3} / {run.strategy:<17} seed={run.seed:<4} "
              f"best_epoch={run.summary['best_epoch']:2d}  {run.attempt.name}")
    if args.dry_run:
        print("\nDRY RUN ONLY: Test file not opened; no predictions or files written.")
        return

    examples = verify_test_examples(args.test_file, args.val_file, args.train_file,
                                    args.expected_test_questions)
    test_hash = sha256(args.test_file)
    device, use_bf16 = select_device_precision(args.device, args.precision)
    max_length = int(runs[0].summary["max_length"])
    model_name = str(runs[0].summary.get("model_name") or DEFAULT_MODEL)
    print(f"\nTEST UNLOCKED: {len(examples):,} questions / {10*len(examples):,} candidate pairs")
    print(f"Device={device}, precision={'bf16' if use_bf16 else 'fp32'}, "
          f"batch={args.batch_size}, max_length={max_length}")
    print("NO TRAINING / NO TEST-BASED MODEL SELECTION.\n")

    all_results = []
    for index, run in enumerate(runs, 1):
        print(f"[{index:02d}/{len(runs):02d}]", end="", flush=True)
        all_results.append(evaluate_model(run, examples, test_hash, args, device, use_bf16))

    baseline_results = []
    if not args.no_baselines:
        for name in ("bm25", "pretrained_minilm"):
            baseline_results.append(evaluate_baseline(name, examples, test_hash, args, device, use_bf16,
                                                      model_name, max_length))
    generate_reports(all_results, baseline_results, args.output_root, args.test_file)
    print("\nALL TEST EVALUATIONS COMPLETE")
    print("Summary table:", args.output_root / "FINAL_TEST_RESULTS.md")
    print("Individual runs:", args.output_root / "results_by_run.csv")
    print("Mean ± SD:", args.output_root / "results_mean_sd.csv")
    print("Bridge/Comparison breakdown:", args.output_root / "results_by_type.csv")
    print("Remember: these are final held-out results; do not tune training using them.")


if __name__ == "__main__":
    main()
