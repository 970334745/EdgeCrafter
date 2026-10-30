"""CPU regressions for checkpoint replacement and torch.save compatibility."""

import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.misc import dist_utils


class AtomicCheckpointTests(unittest.TestCase):
    def test_path_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pth'
            dist_utils.save_on_master({'step': 7}, path)
            self.assertEqual(torch.load(path, weights_only=True), {'step': 7})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_write_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pth'
            torch.save({'step': 7}, path)
            original_bytes = path.read_bytes()

            def incomplete_write(obj, destination, **kwargs):
                Path(destination).write_bytes(b'incomplete checkpoint')
                raise OSError('simulated storage failure')

            with mock.patch.object(dist_utils.torch, 'save', side_effect=incomplete_write):
                with self.assertRaises(OSError):
                    dist_utils.save_on_master({'step': 8}, path)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(torch.load(path, weights_only=True), {'step': 7})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_first_write_leaves_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pth'

            def incomplete_write(obj, destination, **kwargs):
                Path(destination).write_bytes(b'incomplete checkpoint')
                raise OSError('simulated storage failure')

            with mock.patch.object(dist_utils.torch, 'save', side_effect=incomplete_write):
                with self.assertRaises(OSError):
                    dist_utils.save_on_master({'step': 1}, path)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_keyword_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pth'
            dist_utils.save_on_master(obj={'step': 4}, f=path)
            self.assertEqual(torch.load(path, weights_only=True), {'step': 4})

    def test_file_objects_remain_supported(self):
        stream = io.BytesIO()
        dist_utils.save_on_master({'step': 3}, stream)
        stream.seek(0)
        self.assertEqual(torch.load(stream, weights_only=True), {'step': 3})

    def test_serialization_options_are_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'last.pth')
            dist_utils.save_on_master({'step': 5}, path,
                                      _use_new_zipfile_serialization=False,
                                      pickle_protocol=2)
            self.assertEqual(torch.load(path, weights_only=True), {'step': 5})

    def test_non_master_does_not_write(self):
        with mock.patch.object(dist_utils, 'is_main_process', return_value=False):
            with mock.patch.object(dist_utils.torch, 'save') as save:
                dist_utils.save_on_master({'step': 2}, io.BytesIO())
        save.assert_not_called()


if __name__ == '__main__':
    unittest.main()
