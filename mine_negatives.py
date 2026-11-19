"""Select hard vs within-candidate random negatives with frozen pretrained reranker.

Writes selection IDs and logits to data/negatives/.
Supports alternative Train inputs through --input.
"""
import argparse
import os
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# Keep Hugging Face model downloads in the same project by default.
os.environ.setdefault('HF_HOME', str(PROJECT_ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', str(PROJECT_ROOT / '.cache' / 'sentence_transformers'))
import numpy as np


def mine_one(row, scores, k=2, seed=42):
    if len(row['documents']) != len(scores):
        raise ValueError('Document and score counts differ')
    negative_indices = [i for i, doc in enumerate(row['documents']) if doc['label'] == 0]
    if len(negative_indices) < k:
        raise ValueError('Not enough negative candidates')
    hard_indices = sorted(negative_indices, key=lambda i: float(scores[i]), reverse=True)[:k]
    rng = random.Random(f'{seed}:{row["id"]}')
    random_indices = rng.sample(negative_indices, k)
    return {
        'id': row['id'],
        'question': row['question'],
        'positive_doc_ids': [doc['doc_id'] for doc in row['documents'] if doc['label'] == 1],
        'hard_negative_doc_ids': [row['documents'][i]['doc_id'] for i in hard_indices],
        'random_negative_doc_ids': [row['documents'][i]['doc_id'] for i in random_indices],
        'document_scores': [float(s) for s in scores],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=PROJECT_ROOT / 'data' / 'splits' / 'train_40k.jsonl')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data' / 'negatives' / 'mined_train_40k.jsonl')
    parser.add_argument('--model', default='cross-encoder/ms-marco-MiniLM-L6-v2')
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    # IMPORTANT: iterate the file, NOT str.splitlines(). HotpotQA paragraphs may
    # contain Unicode separators (e.g. U+2028) that splitlines() mistakes for
    # actual JSONL record boundaries, causing "Unterminated string" errors.
    rows = []
    with args.input.open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'Invalid JSONL record at {args.input}:{line_number}: {exc}'
                ) from exc
    if not rows:
        raise ValueError(f'No questions found in {args.input}')
    print(f'Loaded {len(rows):,} questions from {args.input}')

    import torch
    from sentence_transformers import CrossEncoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Loading pretrained CrossEncoder on', device)
    model = CrossEncoder(args.model, max_length=args.max_length, device=device)
    pairs = [(row['question'], doc['text']) for row in rows for doc in row['documents']]
    scores = np.asarray(model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)).reshape(-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        offset = 0
        for row in rows:
            n = len(row['documents'])
            mined = mine_one(row, scores[offset:offset + n], k=args.k, seed=args.seed)
            f.write(json.dumps(mined, ensure_ascii=False) + '\n')
            offset += n
    print('Saved', len(rows), 'questions to', args.output)
    if rows:
        with args.output.open('r', encoding='utf-8') as stream:
            first = json.loads(next(stream))
        print('First example:', json.dumps(first, indent=2)[:1300])
    print('IMPORTANT: These are raw relevance scores (logits), not calibrated probabilities.')


if __name__ == '__main__':
    main()
