"""Fast checks for the portable package's data and integrity boundaries."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run


class PortableDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((run.DEMO / 'manifest.json').read_text(encoding='utf-8'))
        cls.rows = run.read_rows(run.verified_asset(cls.manifest['test']))
        cls.reference = run.read_rows(run.verified_asset(cls.manifest['models']['baseline']['predictions']))

    def test_every_distributed_asset_is_intact(self):
        self.assertFalse(self.manifest['final_unseen_included'])
        self.assertEqual(len(self.manifest['models']), 8)
        for spec in self.manifest['models'].values():
            run.verified_asset(spec['checkpoint'])
            reference = run.read_rows(run.verified_asset(spec['predictions']))
            rows = run.ordered_test_rows(self.rows, reference)
            self.assertEqual(len(rows), 436)
            self.assertEqual(sum(row['label'] == row['prediction'] for row in reference), spec['expected_correct'])
        run.verified_asset(self.manifest['low_rank_linear'])

    def test_input_order_does_not_change_example_alignment(self):
        ordered = run.ordered_test_rows(list(reversed(self.rows)), self.reference)
        self.assertEqual([f"validation:{row['idx']}" for row in ordered],
                         [row['test_id'] for row in self.reference])

    def test_duplicate_examples_are_rejected(self):
        invalid = [self.rows[0], *self.rows[:-1]]
        with self.assertRaisesRegex(ValueError, 'do not align'):
            run.ordered_test_rows(invalid, self.reference)

    def test_changed_test_label_is_rejected(self):
        invalid = [dict(row) for row in self.rows]
        invalid[0]['label'] = 1 - invalid[0]['label']
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            run.ordered_test_rows(invalid, self.reference)

    def test_truncated_test_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly 436'):
            run.ordered_test_rows(self.rows[:-1], self.reference)

    def test_corrupt_asset_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'demo'
            root.mkdir()
            (root / 'asset').write_bytes(b'abc')
            (root.parent / 'outside').write_bytes(b'abc')
            record = {'path': 'asset', 'size_bytes': 3, 'sha256': sha256(b'abc').hexdigest()}
            with patch.object(run, 'DEMO', root):
                self.assertEqual(run.verified_asset(record), root / 'asset')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    run.verified_asset({**record, 'path': '../outside'})
                (root / 'asset').write_bytes(b'abd')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    run.verified_asset(record)


if __name__ == '__main__':
    unittest.main()
