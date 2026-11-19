"""Download HotpotQA once, then save the original splits inside this project.

Reuses an existing standard Hugging Face download cache when possible. The final
`datasets/raw/hotpotqa_distractor/` folder is self-contained and can be loaded
without the network by `prepare_data.py`.
"""
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATASET_DIR = PROJECT_ROOT / 'datasets' / 'raw' / 'hotpotqa_distractor'


def main():
    parser = argparse.ArgumentParser(description='Store original HotpotQA splits inside this project.')
    parser.add_argument('--out-dir', type=Path, default=RAW_DATASET_DIR,
                        help='Destination for locally saved Hugging Face DatasetDict.')
    args = parser.parse_args()
    destination = args.out_dir.resolve()

    from datasets import load_dataset, load_from_disk

    if destination.exists():
        try:
            dataset = load_from_disk(str(destination))
        except Exception as exc:
            parser.error(f'{destination} exists but is not a valid saved dataset: {exc}. '
                         'Please inspect or rename it before retrying.')
        if 'train' not in dataset or 'validation' not in dataset:
            parser.error(f'Incomplete dataset in {destination}; expected train and validation.')
        print('Already downloaded; reusing project-local dataset at:', destination)
    else:
        print('Loading official hotpotqa/hotpot_qa distractor dataset...')
        print('Hugging Face can reuse your existing download cache, if present.')
        dataset = load_dataset('hotpotqa/hotpot_qa', 'distractor')
        if 'train' not in dataset or 'validation' not in dataset:
            raise RuntimeError('Unexpected dataset splits: need train and validation')
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(destination))
        print('Saved complete raw dataset inside project:', destination)

    for split in ('train', 'validation'):
        print(f'{split}: {len(dataset[split]):,} questions')
    print('The raw dataset is separate from data/ and is ignored by Git.')
    print('Official validation is stored only; prepare_data.py will use TRAIN exclusively.')


if __name__ == '__main__':
    main()
