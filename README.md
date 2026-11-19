# Hard Negative Mining for Evidence Reranking in Multi-Hop Question Answering

**Columbia University · COMS 4705 Natural Language Processing · Fall 2026**

This project studies whether **static hard-negative mining** improves evidence-paragraph reranking for multi-hop questions, compared with sampling negatives randomly from the same candidate set or from a broader training-document pool. We use the **HotpotQA Distractor** setting and fine-tune the pretrained [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) cross-encoder.

The model **ranks evidence paragraphs; it does not generate answers**. For each question, the reranker scores the ten supplied candidate Wikipedia paragraphs and aims to place both gold supporting paragraphs at the top.

## Research design

| Component | Configuration |
| --- | --- |
| Dataset | HotpotQA Distractor (ten candidate paragraphs per question) |
| Training subsets | **2,000 / 10,000 / 40,000** nested questions |
| Internal validation | **3,000** questions from the official training split |
| Held-out test | **7,345** eligible questions from the official validation split |
| Model | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Negative strategies | `global_random`, `candidate_random`, `hard` |
| Per-question training examples | 2 gold-positive pairs + 2 sampled-negative pairs |
| Seeds | `42`, `123`, `2026` |
| Training | 10 epochs maximum, batch size 32, learning rate `2e-5`, maximum sequence length 256 |
| Objective | Pointwise binary cross-entropy with logits (BCEWithLogitsLoss) |
| Checkpoint selection | Highest validation Recall@2; ties broken by Paragraph EM@2, then nDCG@5 |
| Metrics | Recall@2, Paragraph EM@2, nDCG@5 |

All three strategies use the **same base checkpoint**, train-question subsets, number of positive/negative pairs, optimization settings, and validation questions. Each of the 27 fine-tuning runs is independent (3 sizes × 3 strategies × 3 seeds).

### Negative sampling strategies

- **Candidate Random:** sample two negatives uniformly from the eight non-supporting documents among the question's ten candidates.
- **Static Hard Negative:** use the *frozen, original pretrained reranker* to score a question's non-supporting candidates and select the two highest-scoring negatives. Mining is performed before fine-tuning, not refreshed every epoch.
- **Global Random:** sample two unique pages from a training-only, title-deduplicated Wikipedia-page pool for the matching training size, excluding **all ten candidate page titles for the current question**. Unlabeled relevant pages may still remain (a possible false-negative limitation).

All negative selections are fixed for the duration of each run and reused across epochs. Random sampling is determined by the configured seed and question ID.

## Repository layout

```text
Project/
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-local-cu128.txt
│
├── train.py                       # Fine-tune one model
├── run_experiments.py             # Launch the full experiment matrix
├── evaluate.py                    # Ranking metrics and predictions evaluation
├── baseline.py                    # BM25 and frozen pretrained MiniLM
├── test_experiments.py            # Frozen-checkpoint held-out evaluation
│
├── scripts/
│   ├── data/
│   │   ├── download_dataset.py
│   │   ├── audit_hotpotqa.py
│   │   ├── prepare_data.py
│   │   ├── mine_negatives.py
│   │   ├── split_mined.py
│   │   └── build_global_pools.py
│   ├── visualization/
│   │   └── visualize_training.py
│   └── utils/                     # Optional maintenance utilities
│       ├── check_gpu.py
│       └── migrate_data_layout.py
│
├── datasets/raw/hotpotqa_distractor/  # Local HF dataset; not tracked
├── data/
│   ├── splits/                    # train_{2k,10k,40k}, val, test
│   ├── negatives/                 # Frozen mining selections
│   └── global_pools/             # Train-only global-negative pages
│
├── outputs/
│   ├── experiments/              # Model checkpoints and epoch histories
│   ├── test_evaluation/          # Per-model test predictions and aggregates
│   └── training_visualizations/  # Plots, CSV summaries, TensorBoard logs
└── reports/
    ├── milestone/
    └── final/
```

`data/`, `datasets/`, `outputs/`, model weights and local caches are deliberately Git-ignored. Keep these artifacts locally or in separately managed storage. **Curated, lightweight tables/figures can be copied into `reports/` for version control.**

> **Important after moving scripts:** Some existing data/visualization utilities calculate defaults with `Path(__file__).resolve().parent`, so changing a script's folder changes its default input/output locations. The commands below **pass explicit project-root paths** for those moved utilities. Always run them from the `Project/` root. `train.py`, `evaluate.py`, `baseline.py`, `run_experiments.py`, and `test_experiments.py` stay at the root because they import/reference one another.

## Environment

Use Python 3.11 and a GPU-enabled PyTorch build. These example commands are for **Windows PowerShell** from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-local-cu128.txt
python -m pip install tensorboard
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements-local-cu128.txt` is for the Windows RTX 50-series / CUDA 12.8 setup. For other machines, first install a PyTorch build compatible with the available hardware, then install the shared `requirements.txt`. Hugging Face will cache downloaded pretrained files locally on first use.

## Data preparation (reproduction only)

**These steps are already complete for the reported runs. Do not regenerate or overwrite the existing splits, mined negatives, or pools merely to rerun evaluation.** The data-preparation scripts are retained for reproducibility.

First download the HotpotQA Distractor dataset using `scripts/data/download_dataset.py` (see that script's local CLI options), and optionally audit it with `scripts/data/audit_hotpotqa.py`. With the raw dataset present at `datasets/raw/hotpotqa_distractor/`, the rest of the pipeline is:

```powershell
# Create deterministic nested train subsets, disjoint internal Val, and held-out Test
python scripts/data/prepare_data.py --dataset-dir datasets/raw/hotpotqa_distractor --out-dir data/splits

# Mine static hard negatives using the *frozen original* MiniLM on the 40K set
python scripts/data/mine_negatives.py --input data/splits/train_40k.jsonl --output data/negatives/mined_train_40k.jsonl --seed 42

# Derive the 2K and 10K subsets from the original 40K mined selections
python scripts/data/split_mined.py --data-dir data

# Build separate training-only global document pools for 2K / 10K / 40K
python scripts/data/build_global_pools.py --data-dir data
```

The official HotpotQA **training** split yields an eligible pool (ten distinct candidates and exactly two distinct gold pages). The nested training subsets and internal 3K validation set are sampled from it. The official HotpotQA **validation** split supplies our own held-out 7,345-question Test set. **This is not the benchmark's hidden official Test split.** No held-out Test examples are used for mining, training, or checkpoint selection.

## Training

```powershell
# Inspect the intended runs without writing models
python run_experiments.py --sizes 2k 10k 40k --epochs 10 --dry-run

# Run all 27 configurations (only if reproducing the experiments)
python run_experiments.py --sizes 2k 10k 40k --epochs 10

# Re-run the same configuration after an interruption;
# skips verified completed attempts, but does not resume an interrupted epoch
python run_experiments.py --sizes 2k 10k 40k --epochs 10 --resume
```

Each run uses `outputs/experiments/<size>/<strategy>/seed_<seed>/attempt_<NNN>/`. The runner tracks `completed.json`; the trainer saves `training_summary.json`, `metrics_history.json`, `sampling_selections.jsonl`, and `best_model/`. The best checkpoint is selected using **validation** metrics, not the last epoch or Test score.

The launcher defaults to **3 epochs** if `--epochs` is omitted, so pass **`--epochs 10` explicitly** to reproduce the reported results.

## Evaluation

### Baselines on validation

```powershell
python baseline.py --input data/splits/val.jsonl --model both --output-dir outputs/baselines_val3k
```

### Held-out Test (only after the experiment configuration is frozen)

```powershell
# Verifies all 27 finished 10-epoch models; does not read Test
python test_experiments.py --dry-run

# Scores 27 frozen best checkpoints plus BM25 and pretrained MiniLM on Test
python test_experiments.py

# After interruption, reuse completed results that pass fingerprint checks
python test_experiments.py --resume
```

Test outputs are stored under `outputs/test_evaluation/`, including `FINAL_TEST_RESULTS.md`, `results_mean_sd.csv`, `results_by_run.csv`, and `results_by_type.csv`. The script also writes per-question predictions and per-type (Bridge / Comparison) metrics. **Do not tune any training decision using the Test outcomes.**

### Final held-out Test results

Values for fine-tuned models are **mean ± sample standard deviation** across three seeds. All values are percentages.

| Train size | Strategy | Recall@2 ↑ | Paragraph EM@2 ↑ | nDCG@5 ↑ |
| --- | --- | ---: | ---: | ---: |
| — | BM25 | 59.59 | 28.59 | 75.51 |
| — | Pretrained MiniLM | 71.33 | 45.64 | 84.56 |
| 2K | Global Random | 73.16 ± 0.35 | 50.10 ± 0.63 | 86.36 ± 0.20 |
| 2K | Static Hard Negative | 85.11 ± 0.46 | 71.88 ± 0.77 | 94.16 ± 0.22 |
| 2K | **Candidate Random** | **86.37 ± 0.26** | **74.08 ± 0.45** | **94.87 ± 0.14** |
| 10K | Global Random | 75.50 ± 0.11 | 54.19 ± 0.08 | 87.91 ± 0.18 |
| 10K | Static Hard Negative | 89.49 ± 0.27 | 79.96 ± 0.49 | 96.12 ± 0.11 |
| 10K | **Candidate Random** | **89.94 ± 0.17** | **80.77 ± 0.33** | **96.43 ± 0.07** |
| 40K | Global Random | 78.59 ± 0.62 | 59.61 ± 1.14 | 89.93 ± 0.36 |
| 40K | Static Hard Negative | 91.56 ± 0.44 | 83.76 ± 0.81 | 97.07 ± 0.15 |
| 40K | **Candidate Random** | **92.39 ± 0.15** | **85.35 ± 0.24** | **97.47 ± 0.02** |

**Observed trend:** Within-candidate random negatives consistently outperform global random negatives and narrowly outperform static hard mining at all three training sizes. These are observed differences, **not claims of statistical significance**. All three strategies improve with additional training examples.

## Visualizations and TensorBoard

The visualization script aggregates three random seeds **per epoch**. By default, it plots only the **2K and 10K** experiments, matching the milestone presentation.

```powershell
python scripts/visualization/visualize_training.py --model-root outputs/experiments --output-dir outputs/training_visualizations --sizes 2k 10k

# Cleaner lines, without ±1 SD shading
python scripts/visualization/visualize_training.py --model-root outputs/experiments --output-dir outputs/training_visualizations --sizes 2k 10k --no-std-shading

# Browse mean-only curves locally
tensorboard --logdir outputs/training_visualizations/tensorboard_means --port 6006
```

Visit <http://localhost:6006>. Exported paper figures include `validation_metrics_grid.pdf/.png` (2 rows × 3 metrics), standalone validation-metric curves, and training BCE loss curves. To include 40K for the final report, use `--sizes 2k 10k 40k` (3 rows × 3 metrics).

## Git, data management, and reproducibility

- The provided `.gitignore` excludes `.venv`, Hugging Face / Torch caches, generated JSONL datasets, model checkpoints, experiment outputs, TensorBoard event logs, and secrets. **Ignored files remain on disk**; Git ignores them only for new commits.
- Put selected figures, report source files, lightweight metric summaries, and methodology notes under `reports/` if they should appear in the repository. For example:

```powershell
New-Item -ItemType Directory -Force reports/final | Out-Null
Copy-Item outputs/test_evaluation/FINAL_TEST_RESULTS.md reports/final/
Copy-Item outputs/training_visualizations/validation_metrics_grid.pdf reports/final/
```

- `.gitignore` does **not** automatically untrack files that were already committed. To remove a previously tracked large directory **from the Git index while preserving the local files**, run e.g. `git rm -r --cached -- data/ datasets/ outputs/` **only if those paths are currently tracked**, then commit the change. Check `git status` before committing. Do **not** run `git rm` without `--cached` unless you intend to delete the files locally.
- Keep experiment seeds, input hashes, code versions, exact split manifests, and `best_model/` provenance together for reproducibility. The experiments are CPU/GPU hardware-dependent and exact bitwise numerical reproducibility across machines is not guaranteed.

## Limitations and next analyses

1. Static hard negatives come from the original frozen reranker; online or iterative mining has not been tested.
2. Candidate labels are document-level and may omit other genuinely relevant Wikipedia pages, particularly in the global pool.
3. The experiments evaluate reranking over an **already provided set of ten documents**, not retrieval from the full Wikipedia corpus and not final answer generation.
4. Further analysis can include paired bootstrap confidence intervals, Bridge vs Comparison breakdowns, and qualitative failure cases. Keep all post-hoc analyses separate from training/model selection.

## References

- Yang et al. (2018), *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering* — <https://hotpotqa.github.io/>
- Pretrained MiniLM cross-encoder — <https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2>
- Sentence Transformers — <https://www.sbert.net/>
