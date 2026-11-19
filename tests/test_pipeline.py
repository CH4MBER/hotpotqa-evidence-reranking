import unittest
from prepare_data import normalize_example
from mine_negatives import mine_one


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            'id': 'q1', 'question': 'Which city?', 'type': 'bridge',
            'context': {'title': [f'Doc{i}' for i in range(10)],
                        'sentences': [[f'Sentence {i}.'] for i in range(10)]},
            'supporting_facts': {'title': ['Doc2', 'Doc7'], 'sent_id': [0, 0]},
        }

    def test_labels(self):
        row = normalize_example(self.sample)
        self.assertEqual(len(row['documents']), 10)
        self.assertEqual([d['doc_id'] for d in row['documents'] if d['label']], [2, 7])
        self.assertIn('Title: Doc2', row['documents'][2]['text'])

    def test_mining(self):
        row = normalize_example(self.sample)
        scores = [0.0, 0.9, 1.0, -2.0, 0.1, 0.8, 0.2, 2.0, 0.3, -0.4]
        mined = mine_one(row, scores, k=2)
        self.assertEqual(mined['hard_negative_doc_ids'], [1, 5])
        self.assertTrue(set(mined['random_negative_doc_ids']).isdisjoint({2, 7}))
        self.assertEqual(mine_one(row, scores, k=2)['random_negative_doc_ids'], mined['random_negative_doc_ids'])


if __name__ == '__main__':
    unittest.main()
