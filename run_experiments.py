"""Launch the HotpotQA 3 x 3 x 3 fine-tuning matrix, one run at a time.

IMPORTANT: train.py does NOT yet exist. This script provides a stable CLI
contract for the upcoming train.py. Use --dry-run until train.py is ready.

Planned trainer CLI:
  python train.py --size 2k --strategy candidate_random --seed 42 \
    --train-file data/splits/train_2k.jsonl --mined-file data/negatives/mined_train_2k.jsonl \
    --val-file data/splits/val.jsonl --epochs 3 --batch-size 32 \
    --learning-rate 2e-5 --max-length 256 --output-dir <attempt_dir>

For global_random only, adds --pool-file data/global_pools/pool_2k.jsonl.

Trainer output contract (required after exit code zero):
  <attempt_dir>/training_summary.json     (JSON with matching size/strategy/seed)
  <attempt_dir>/best_model/config.json
  <attempt_dir>/best_model/model.safetensors OR pytorch_model.bin

Output per attempt is immutable: interrupted/failed runs are never overwritten.
Use --resume to skip verified completed runs and retry unfinished ones.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALLOWED_SIZES = ("2k", "10k", "40k")
ALLOWED_STRATEGIES = ("candidate_random", "hard", "global_random")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temp, path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def artifact_check(attempt: Path, size: str, strategy: str, seed: int) -> None:
    """Refuse false success if trainer returned 0 without saving actual artifacts."""
    summary_path = attempt / "training_summary.json"
    model_dir = attempt / "best_model"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing trainer summary: {summary_path}")
    summary = read_json(summary_path)
    expected = {"size": size, "strategy": strategy, "seed": seed}
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"{summary_path}: expected {key}={value!r}, got {summary.get(key)!r}")
    if not (model_dir / "config.json").is_file():
        raise RuntimeError(f"Missing best model config: {model_dir / 'config.json'}")
    weight_names = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
                    "pytorch_model.bin.index.json")
    if not any((model_dir / candidate).is_file() for candidate in weight_names):
        raise RuntimeError(f"Missing best model weight file in {model_dir}")


def next_attempt_directory(run_dir: Path) -> Path:
    attempt_number = 1
    while (run_dir / f"attempt_{attempt_number:03d}").exists():
        attempt_number += 1
    return run_dir / f"attempt_{attempt_number:03d}"


def make_command(args, size: str, strategy: str, seed: int, attempt: Path) -> list[str]:
    data = args.data_dir
    cmd = [
        sys.executable, str(args.train_script),
        "--size", size, "--strategy", strategy, "--seed", str(seed),
        "--train-file", str(data / "splits" / f"train_{size}.jsonl"),
        "--mined-file", str(data / "negatives" / f"mined_train_{size}.jsonl"),
        "--val-file", str(data / "splits" / "val.jsonl"),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--max-length", str(args.max_length),
        "--output-dir", str(attempt),
    ]
    if strategy == "global_random":
        cmd += ["--pool-file", str(data / "global_pools" / f"pool_{size}.jsonl")]
    return cmd


def required_files(args, size: str, strategy: str) -> list[Path]:
    data = args.data_dir
    files = [data / "splits" / f"train_{size}.jsonl",
             data / "negatives" / f"mined_train_{size}.jsonl",
             data / "splits" / "val.jsonl"]
    if strategy == "global_random":
        files.append(data / "global_pools" / f"pool_{size}.jsonl")
    return files


def fingerprint(args, size: str, strategy: str, seed: int,
                file_hash_cache: dict[Path, str]) -> str:
    required = required_files(args, size, strategy) + [args.train_script]
    for path in required:
        if path not in file_hash_cache:
            file_hash_cache[path] = sha256(path)
    spec = {
        "size": size, "strategy": strategy, "seed": seed,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "max_length": args.max_length,
        "files": {str(p.resolve()): file_hash_cache[p] for p in required},
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()


def command_display(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def launch(args, size: str, strategy: str, seed: int, run_dir: Path,
           signature: str) -> bool:
    """Return True only for a zero-exit run that saved all required artifacts."""
    attempt = next_attempt_directory(run_dir)
    attempt.mkdir(parents=True, exist_ok=False)
    cmd = make_command(args, size, strategy, seed, attempt)
    log_path = attempt / "train.log"
    state_path = attempt / "runner_status.json"
    state = {
        "size": size, "strategy": strategy, "seed": seed,
        "fingerprint": signature, "status": "running", "started_utc": now(),
        "command": cmd, "log": str(log_path),
    }
    write_json(state_path, state)
    print(f"\nSTART {size}/{strategy}/seed_{seed} | {attempt.name}", flush=True)
    print(command_display(cmd), flush=True)

    code = -1
    failure_reason = None
    process = None
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write("Command: " + command_display(cmd) + "\n\n")
            log.flush()
            process = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            assert process.stdout is not None
            for output in process.stdout:
                print(output, end="", flush=True)
                log.write(output)
                log.flush()
            code = process.wait()
        if code != 0:
            raise RuntimeError(f"train.py exited with code {code}")
        artifact_check(attempt, size, strategy, seed)
    except KeyboardInterrupt:
        failure_reason = "Interrupted by user"
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        state.update({"status": "interrupted", "finished_utc": now(),
                      "return_code": process.returncode if process else None,
                      "error": failure_reason})
        write_json(state_path, state)
        raise
    except Exception as exc:
        failure_reason = str(exc)
    if failure_reason is None:
        state.update({"status": "completed", "finished_utc": now(), "return_code": 0})
        write_json(state_path, state)
        write_json(run_dir / "completed.json", {
            "size": size, "strategy": strategy, "seed": seed,
            "fingerprint": signature,
            "attempt_dir": attempt.name, "completed_utc": now(),
        })
        print(f"SUCCESS {size}/{strategy}/seed_{seed}", flush=True)
        return True
    state.update({"status": "failed", "finished_utc": now(),
                  "return_code": code, "error": failure_reason})
    write_json(state_path, state)
    print(f"FAILED {size}/{strategy}/seed_{seed}: {failure_reason}", flush=True)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--train-script", type=Path, default=ROOT / "train.py")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "experiments")
    p.add_argument("--sizes", choices=ALLOWED_SIZES, nargs="+", default=list(ALLOWED_SIZES))
    p.add_argument("--strategies", choices=ALLOWED_STRATEGIES, nargs="+", default=list(ALLOWED_STRATEGIES))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--dry-run", action="store_true", help="Preview the commands; train.py is not required")
    p.add_argument("--resume", action="store_true", help="Skip verified successes, create fresh attempts for failures")
    p.add_argument("--continue-on-error", action="store_true", help="Keep launching after a failed run")
    args = p.parse_args()
    # Interpret user-supplied relative paths from their current shell directory,
    # not from the subprocess working directory.
    args.data_dir = args.data_dir.resolve()
    args.train_script = args.train_script.resolve()
    args.output_root = args.output_root.resolve()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or args.max_length <= 0:
        p.error("Training hyperparameters must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        p.error("Provide non-repeated seeds")
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.strategies)) != len(args.strategies):
        p.error("Duplicate sizes/strategies are not allowed")

    sizes = [s for s in ALLOWED_SIZES if s in args.sizes]
    strategies = [s for s in ALLOWED_STRATEGIES if s in args.strategies]
    runs = list(itertools.product(sizes, strategies, args.seeds))
    print(f"Plan: {len(runs)} runs = {len(sizes)} sizes x {len(strategies)} strategies x {len(args.seeds)} seeds")
    print(f"Epochs={args.epochs}, batch size={args.batch_size}, lr={args.learning_rate:g}, max length={args.max_length}")
    print("All runs use the same data/splits/val.jsonl. Test is NEVER used.")

    missing = sorted({str(file) for size, strategy, _ in runs
                      for file in required_files(args, size, strategy) if not file.is_file()})
    if missing:
        p.error("Missing required data files:\n  " + "\n  ".join(missing))
    if not args.dry_run and not args.train_script.is_file():
        p.error(f"Missing trainer: {args.train_script}. Implement train.py first; use --dry-run meanwhile.")

    hash_cache: dict[Path, str] = {}
    completed, skipped, failed = 0, 0, 0
    for number, (size, strategy, seed) in enumerate(runs, 1):
        run_dir = args.output_root / size / strategy / f"seed_{seed}"
        # Dry run is completely read-only: do not hash huge inputs or create directories.
        if args.dry_run:
            proposed = run_dir / "attempt_001"
            print(f"[{number:02d}/{len(runs):02d}] {command_display(make_command(args, size, strategy, seed, proposed))}")
            continue

        signature = fingerprint(args, size, strategy, seed, hash_cache)
        completed_path = run_dir / "completed.json"
        if completed_path.is_file():
            marker = read_json(completed_path)
            attempt_dir = run_dir / str(marker.get("attempt_dir", ""))
            if marker.get("fingerprint") != signature:
                p.error(f"Run already completed with DIFFERENT inputs/settings: {run_dir}. Move it aside before rerunning.")
            try:
                artifact_check(attempt_dir, size, strategy, seed)
            except Exception as exc:
                p.error(f"Invalid completed run marker at {run_dir}: {exc}")
            if not args.resume:
                p.error(f"Completed run already exists: {run_dir}; pass --resume to skip it.")
            skipped += 1
            print(f"[{number:02d}/{len(runs):02d}] SKIP completed {size}/{strategy}/seed_{seed}")
            continue

        success = launch(args, size, strategy, seed, run_dir, signature)
        if success:
            completed += 1
        else:
            failed += 1
            if not args.continue_on_error:
                break

    if args.dry_run:
        print("\nDRY RUN ONLY: no training was performed and nothing was written.")
    else:
        print(f"\nFinished: {completed} newly completed, {skipped} skipped, {failed} failed.")
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
