"""Fine-tune the HotpotQA evidence reranker with three negative-sampling strategies.

Compatible with the existing COMS 4705 run_experiments.py command contract.

Examples (run from the Project directory):
  python train.py --size 2k --strategy candidate_random --seed 42 \
      --epochs 1 --limit-train-questions 32 --limit-val-questions 50 \
      --output-dir outputs/smoke/manual_2k_candidate_seed42

  python run_experiments.py --sizes 2k --strategies candidate_random hard \
      --seeds 42 --epochs 1 --output-root outputs/smoke_experiments

This trainer NEVER reads the held-out data/splits/test.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".cache" / "sentence_transformers"))

from evaluate import evaluate_ranking, summarize, validate_example  # same metrics as baseline.py

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
SIZES = {"2k": 2000, "10k": 10000, "40k": 40000}
STRATEGIES = ("global_random", "candidate_random", "hard")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_title(title: str) -> str:
    """Match build_global_pools.py exactly for page identity/exclusion."""
    return " ".join(unicodedata.normalize("NFKC", title).split()).casefold()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    result: list[dict] = []
    # Iteration on a file handle is essential: str.splitlines() splits U+2028
    # occurring INSIDE some Wikipedia JSON strings.
    with path.open("r", encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_num}")
            result.append(value)
    if not result:
        raise ValueError(f"Input file is empty: {path}")
    return result


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def validate_train_and_mining(train: list[dict], mined: list[dict]) -> dict[str, dict]:
    train_ids: set[str] = set()
    for row in train:
        validate_example(row)
        qid = row["id"]
        if not isinstance(qid, str) or not qid or qid in train_ids:
            raise ValueError(f"Empty/duplicate train question ID: {qid!r}")
        train_ids.add(qid)
        if not all(isinstance(doc["title"], str) and doc["title"].strip()
                   and isinstance(doc["text"], str) and doc["text"].strip()
                   for doc in row["documents"]):
            raise ValueError(f"Question {qid}: missing document title/text")

    index: dict[str, dict] = {}
    for row in mined:
        qid = row.get("id")
        if not isinstance(qid, str) or not qid or qid in index:
            raise ValueError(f"Empty/duplicate mining question ID: {qid!r}")
        index[qid] = row
    if train_ids != set(index):
        missing = train_ids - set(index)
        extra = set(index) - train_ids
        raise ValueError(f"Mining IDs mismatch: {len(missing)} missing and {len(extra)} extra; "
                         f"examples: {list(sorted(missing))[:2]}, {list(sorted(extra))[:2]}")
    for row in train:
        qid = row["id"]
        mined_row = index[qid]
        if mined_row.get("question") != row["question"]:
            raise ValueError(f"Question text mismatch in mining for {qid}")
        doc_map = {doc["doc_id"]: doc for doc in row["documents"]}
        if len(doc_map) != 10:
            raise ValueError(f"Question {qid} has invalid document IDs")
        positives = {doc_id for doc_id, doc in doc_map.items() if doc["label"] == 1}
        if set(mined_row.get("positive_doc_ids", [])) != positives:
            raise ValueError(f"Positive document mismatch in mining for {qid}")
        hard = mined_row.get("hard_negative_doc_ids")
        if not isinstance(hard, list) or len(hard) != 2 or len(set(hard)) != 2:
            raise ValueError(f"Question {qid}: expected exactly 2 different hard negatives")
        if any(doc_id not in doc_map or doc_map[doc_id]["label"] != 0 for doc_id in hard):
            raise ValueError(f"Question {qid}: hard negatives must refer to label-0 documents")
        if "document_scores" in mined_row:
            scores = mined_row["document_scores"]
            if not isinstance(scores, list) or len(scores) != 10:
                raise ValueError(f"Question {qid}: expected 10 mined document scores")
            ordered = [doc["doc_id"] for doc in row["documents"]]
            negative_positions = [i for i, doc_id in enumerate(ordered) if doc_map[doc_id]["label"] == 0]
            expected = [ordered[i] for i in sorted(negative_positions, key=lambda i: float(scores[i]), reverse=True)[:2]]
            if hard != expected:
                raise ValueError(f"Question {qid}: hard-negative IDs disagree with saved scores")
    return index


def validate_val(val: list[dict], train_ids: set[str]) -> None:
    val_ids: set[str] = set()
    for row in val:
        validate_example(row)
        qid = row["id"]
        if not isinstance(qid, str) or qid in val_ids:
            raise ValueError(f"Invalid/duplicate validation question ID: {qid!r}")
        if qid in train_ids:
            raise ValueError(f"DATA LEAKAGE: question {qid} occurs in both Train and Val")
        val_ids.add(qid)


def validate_pool(pool: list[dict], train_ids: set[str]) -> list[dict]:
    pool_by_title: set[str] = set()
    for doc in pool:
        if not isinstance(doc.get("title"), str) or not doc["title"].strip():
            raise ValueError("Global pool document without title")
        if not isinstance(doc.get("text"), str) or not doc["text"].strip():
            raise ValueError("Global pool document without text")
        key = canonical_title(doc["title"])
        if not key or key in pool_by_title:
            raise ValueError(f"Duplicate/empty normalized title in global pool: {doc['title']!r}")
        pool_by_title.add(key)
        source = doc.get("source_question_id")
        if source is not None and source not in train_ids:
            raise ValueError(f"Global pool includes a non-train source question: {source}")
    if len(pool) < 12:
        raise ValueError("Global negative pool is unexpectedly small")
    return pool


def sample_global(pool: list[dict], forbidden_titles: set[str], seed: int, qid: str) -> list[dict]:
    """Uniformly draw two distinct pages conditional on exclusion by title.

    Rejection sampling is unbiased over the remaining pool, and avoids scanning a
    274K document pool once per question. A bounded fallback covers extreme cases.
    """
    rng = random.Random(f"global:{seed}:{qid}")
    selected: list[dict] = []
    selected_titles: set[str] = set()
    tries = 0
    while len(selected) < 2 and tries < 1000:
        tries += 1
        document = pool[rng.randrange(len(pool))]
        key = canonical_title(document["title"])
        if key in forbidden_titles or key in selected_titles:
            continue
        selected.append(document)
        selected_titles.add(key)
    if len(selected) != 2:
        raise ValueError(f"Could not draw two valid global negatives for question {qid}")
    return selected


def construct_training_data(train: list[dict], mined_index: dict[str, dict],
                            strategy: str, seed: int, pool: list[dict] | None):
    pairs: list[tuple[str, str, float]] = []
    selection: list[dict] = []
    for row in train:
        qid = row["id"]
        docs = row["documents"]
        by_doc_id = {doc["doc_id"]: doc for doc in docs}
        positives = [doc for doc in docs if doc["label"] == 1]
        negatives = [doc for doc in docs if doc["label"] == 0]
        if len(positives) != 2 or len(negatives) != 8:
            raise ValueError(f"Question {qid} must have two positives and eight negatives")

        if strategy == "hard":
            selected = [by_doc_id[doc_id] for doc_id in mined_index[qid]["hard_negative_doc_ids"]]
            selected_ids: list[int | str] = [doc["doc_id"] for doc in selected]
        elif strategy == "candidate_random":
            rng = random.Random(f"{seed}:{qid}")  # matches mine_negatives.py for seed 42
            selected = rng.sample(negatives, 2)
            selected_ids = [doc["doc_id"] for doc in selected]
        elif strategy == "global_random":
            assert pool is not None
            excluded = {canonical_title(doc["title"]) for doc in docs}
            selected = sample_global(pool, excluded, seed, qid)
            selected_ids = [doc["doc_id"] for doc in selected]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if len(selected) != 2:
            raise AssertionError("Two negatives required per question")
        for doc in positives:
            pairs.append((row["question"], doc["text"], 1.0))
        for doc in selected:
            pairs.append((row["question"], doc["text"], 0.0))
        selection.append({
            "id": qid,
            "positive_doc_ids": [doc["doc_id"] for doc in positives],
            "negative_doc_ids": selected_ids,
            "negative_titles": [doc["title"] for doc in selected],
            "strategy": strategy,
        })
    return pairs, selection


def checkpoint_save(model, tokenizer, directory: Path) -> None:
    """Save to a staging directory, then replace the previous best directory."""
    staging = directory.with_name(directory.name + ".staging")
    previous = directory.with_name(directory.name + ".previous")
    if staging.exists() or previous.exists():
        raise FileExistsError(f"Stale checkpoint staging/backup exists beside {directory}. Inspect before retry.")
    staging.mkdir(parents=True)
    try:
        model.save_pretrained(str(staging), safe_serialization=True)
        tokenizer.save_pretrained(str(staging))
        if not (staging / "config.json").exists() or not (
            (staging / "model.safetensors").exists() or (staging / "pytorch_model.bin").exists()
        ):
            raise RuntimeError("Saved checkpoint is missing config or model weights")
        if directory.exists():
            os.replace(directory, previous)
        try:
            os.replace(staging, directory)
        except Exception:
            if previous.exists() and not directory.exists():
                os.replace(previous, directory)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def set_seed(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=tuple(SIZES), default="2k")
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--mined-file", type=Path)
    parser.add_argument("--val-file", type=Path)
    parser.add_argument("--pool-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Unique directory. Refuses to overwrite an existing training summary/checkpoint.")
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--limit-train-questions", type=int, default=None,
                        help="DEBUG ONLY: restrict train questions, not for official experiments.")
    parser.add_argument("--limit-val-questions", type=int, default=None,
                        help="DEBUG ONLY: restrict validation questions, not for official experiments.")
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    for key in ("epochs", "batch_size", "max_length", "log_every"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_','-')} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("Invalid optimizer/scheduler parameters")
    for key in ("limit_train_questions", "limit_val_questions"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be positive")
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.eval_batch_size <= 0:
        raise ValueError("eval-batch-size must be positive")

    data = PROJECT_ROOT / "data"
    train_file = args.train_file or data / "splits" / f"train_{args.size}.jsonl"
    mined_file = args.mined_file or data / "negatives" / f"mined_train_{args.size}.jsonl"
    val_file = args.val_file or data / "splits" / "val.jsonl"
    pool_file = args.pool_file or data / "global_pools" / f"pool_{args.size}.jsonl"
    output = args.output_dir.resolve()
    if (output / "training_summary.json").exists() or (output / "best_model").exists():
        raise FileExistsError(f"Refusing to overwrite completed or partial trainer output: {output}")
    if args.strategy == "global_random" and not pool_file.is_file():
        raise FileNotFoundError(f"Global negative pool missing: {pool_file}")

    train = load_jsonl(train_file)
    mined = load_jsonl(mined_file)
    val = load_jsonl(val_file)
    expected_size = SIZES[args.size]
    if len(train) != expected_size:
        raise ValueError(f"{train_file} has {len(train)} rows, expected {expected_size}")
    mined_index = validate_train_and_mining(train, mined)
    validate_val(val, set(mined_index))
    full_train_n, full_val_n = len(train), len(val)
    if args.limit_train_questions is not None:
        train = train[:args.limit_train_questions]
    if args.limit_val_questions is not None:
        val = val[:args.limit_val_questions]
    debug_run = (len(train) != full_train_n or len(val) != full_val_n)
    if debug_run:
        print("WARNING: DEBUG SUBSET TRAINING. Results are NOT comparable to full runs.", flush=True)

    pool = validate_pool(load_jsonl(pool_file), set(mined_index)) if args.strategy == "global_random" else None
    training_pairs, sampling = construct_training_data(train, mined_index, args.strategy, args.seed, pool)
    if len(training_pairs) != 4 * len(train):
        raise AssertionError("Training pairs must equal four times the question count")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    bf16_supported = device == "cuda" and torch.cuda.is_bf16_supported()
    if args.precision == "bf16" and not bf16_supported:
        raise RuntimeError("BF16 requested but CUDA BF16 support is unavailable")
    use_bf16 = args.precision == "bf16" or (args.precision == "auto" and bf16_supported)
    if device == "cpu" and not debug_run:
        print("WARNING: Full training on CPU can be very slow.", flush=True)

    set_seed(args.seed, torch)
    print(f"Train {args.size} / {args.strategy} / seed {args.seed} | {device} | "
          f"precision={'bf16' if use_bf16 else 'fp32'}", flush=True)
    print(f"Train questions={len(train):,}, pairs={len(training_pairs):,}; Val questions={len(val):,}", flush=True)
    print(f"Loading pretrained checkpoint: {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    if model.config.num_labels != 1:
        raise ValueError(f"Expected num_labels=1 for pointwise BCE, got {model.config.num_labels}")
    model.to(device)
    model.train()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    class PairDataset(Dataset):
        def __init__(self, triples):
            self.triples = triples

        def __len__(self):
            return len(self.triples)

        def __getitem__(self, idx):
            return self.triples[idx]

    def train_collate(batch):
        questions, passages, labels = zip(*batch)
        tokens = tokenizer(list(questions), list(passages), padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt")
        return tokens, torch.tensor(labels, dtype=torch.float32)

    def val_collate(batch):
        questions, passages = zip(*batch)
        return tokenizer(list(questions), list(passages), padding=True, truncation=True,
                         max_length=args.max_length, return_tensors="pt")

    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(PairDataset(training_pairs), batch_size=args.batch_size,
                              shuffle=True, collate_fn=train_collate, num_workers=0,
                              pin_memory=(device == "cuda"), generator=shuffle_generator)
    val_pairs = [(row["question"], doc["text"]) for row in val for doc in row["documents"]]
    val_loader = DataLoader(PairDataset(val_pairs), batch_size=args.eval_batch_size,
                            shuffle=False, collate_fn=val_collate, num_workers=0,
                            pin_memory=(device == "cuda"))

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay_params if name.endswith("bias") or "LayerNorm.weight" in name
         else decay_params).append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate)
    updates_per_epoch = len(train_loader)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_multiplier(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) /
                   float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    criterion = nn.BCEWithLogitsLoss()
    output.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output / "sampling_selections.jsonl", sampling)
    configuration = {
        "size": args.size, "strategy": args.strategy, "seed": args.seed,
        "train_file": str(train_file.resolve()), "mined_file": str(mined_file.resolve()),
        "pool_file": str(pool_file.resolve()) if args.strategy == "global_random" else None,
        "val_file": str(val_file.resolve()), "model_name": args.model_name,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps, "total_optimizer_steps": total_steps,
        "max_length": args.max_length, "device": device,
        "precision": "bf16" if use_bf16 else "fp32",
        "train_questions": len(train), "val_questions": len(val),
        "debug_subset": debug_run, "started_utc": utc_now(),
        "checkpoint_selection": "maximize Val Recall@2; ties: EM@2, then nDCG@5; earliest epoch wins",
        "negative_sampling": "fixed per run; resampled by seed across runs",
    }
    atomic_json(output / "run_config.json", configuration)
    epoch_history: list[dict] = []
    best_key = (-1.0, -1.0, -1.0)
    best_epoch = 0
    best_metrics: dict = {}
    started = time.perf_counter()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        model.train()
        cumulative_loss = 0.0
        seen_pairs = 0
        epoch_start = time.perf_counter()
        for batch_number, (encoded, labels) in enumerate(train_loader, 1):
            encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**encoded).logits.reshape(-1)
                loss = criterion(logits.float(), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            batch_size = labels.numel()
            cumulative_loss += float(loss.detach().item()) * batch_size
            seen_pairs += batch_size
            if batch_number % args.log_every == 0 or batch_number == len(train_loader):
                print(f"  step {batch_number}/{len(train_loader)} | "
                      f"loss={cumulative_loss/seen_pairs:.5f} | "
                      f"lr={scheduler.get_last_lr()[0]:.3g}", flush=True)
        train_loss = cumulative_loss / seen_pairs

        # Score all ten validation candidate paragraphs for every question,
        # exactly matching baseline.py's question/document construction.
        model.eval()
        validation_scores: list[float] = []
        with torch.inference_mode():
            for batch in val_loader:
                encoded = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    output_logits = model(**encoded).logits.reshape(-1)
                validation_scores.extend(output_logits.float().cpu().tolist())
        if len(validation_scores) != len(val_pairs):
            raise RuntimeError("Validation scoring returned wrong number of logits")
        offset = 0
        results: list[dict] = []
        for row in val:
            count = len(row["documents"])
            results.append(evaluate_ranking(row, validation_scores[offset:offset+count]))
            offset += count
        metrics = summarize(results)
        epoch_record = {
            "epoch": epoch,
            "train_bce_loss": train_loss,
            **metrics,
            "seconds": round(time.perf_counter() - epoch_start, 2),
            "optimizer_step": global_step,
        }
        epoch_history.append(epoch_record)
        atomic_json(output / "metrics_history.json", epoch_history)
        print(f"  Val Recall@2={metrics['recall_at_2']:.2%} "
              f"EM@2={metrics['paragraph_em_at_2']:.2%} "
              f"nDCG@5={metrics['ndcg_at_5']:.2%} | "
              f"train BCE={train_loss:.4f}", flush=True)
        key = (metrics["recall_at_2"], metrics["paragraph_em_at_2"], metrics["ndcg_at_5"])
        if key > best_key:
            checkpoint_save(model, tokenizer, output / "best_model")
            best_key, best_epoch, best_metrics = key, epoch, metrics
            best_predictions = [{"id": row["id"], "scores": record["scores"]}
                                for row, record in zip(val, results)]
            atomic_jsonl(output / "best_val_predictions.jsonl", best_predictions)
            atomic_jsonl(output / "best_val_details.jsonl", results)
            print(f"  NEW BEST model saved (epoch {epoch})", flush=True)

    summary = {
        **configuration,
        "finished_utc": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "train_pairs_per_epoch": len(training_pairs),
        "num_positive_pairs_per_epoch": len(train) * 2,
        "num_negative_pairs_per_epoch": len(train) * 2,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "history": epoch_history,
        "best_model_dir": str((output / "best_model").resolve()),
        "best_val_predictions": str((output / "best_val_predictions.jsonl").resolve()),
        "sampling_selections": str((output / "sampling_selections.jsonl").resolve()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "torch_version": torch.__version__,
    }
    # Write this LAST: run_experiments.py interprets summary + valid best_model
    # checkpoint as a completed training attempt.
    atomic_json(output / "training_summary.json", summary)
    print("\nTRAINING COMPLETE", flush=True)
    print(f"Best epoch: {best_epoch} | Recall@2={best_metrics['recall_at_2']:.2%} | "
          f"EM@2={best_metrics['paragraph_em_at_2']:.2%} | "
          f"nDCG@5={best_metrics['ndcg_at_5']:.2%}", flush=True)
    print(f"Summary: {output / 'training_summary.json'}", flush=True)
    print(f"Best model: {output / 'best_model'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted. The runner will mark this attempt as incomplete.", file=sys.stderr)
        raise
