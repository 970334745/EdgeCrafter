"""Check real ECSolver.fit checkpoint flow without datasets or model weights."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.solver import ec_solver


class CheckpointFlowTests(unittest.TestCase):
    def make_solver(self, directory, epochs=3, stop_epoch=1):
        solver = object.__new__(ec_solver.ECSolver)
        solver.cfg = SimpleNamespace(
            epochs=epochs, lrsheduler=None, checkpoint_freq=1,
            clip_max_norm=0, print_freq=100)
        solver.model = torch.nn.Linear(1, 1)
        solver.criterion = torch.nn.MSELoss()
        solver.optimizer = torch.optim.SGD(solver.model.parameters(), lr=0.1)
        solver.lr_scheduler = mock.Mock()
        solver.lr_warmup_scheduler = None
        solver.ema = solver.scaler = solver.writer = None
        solver.device = torch.device('cpu')
        solver.output_dir = Path(directory)
        solver.last_epoch = -1
        solver.train_dataloader = SimpleNamespace(
            dataset=SimpleNamespace(_transforms=SimpleNamespace(stop_epoch=stop_epoch)),
            set_epoch=mock.Mock())
        solver.val_dataloader = solver.evaluator = solver.postprocessor = None
        solver.train = mock.Mock()
        solver.state_dict = lambda: {'last_epoch': solver.last_epoch}
        solver.loaded_checkpoints = []

        def load_resume_state(path):
            state = torch.load(path, weights_only=True)
            solver.last_epoch = state['last_epoch']
            solver.loaded_checkpoints.append(Path(path).name)

        solver.load_resume_state = load_resume_state
        return solver

    def run_fit(self, solver, ap):
        with mock.patch.object(ec_solver, 'stats', return_value=(2, 'tiny CPU test')):
            with mock.patch.object(ec_solver, 'train_one_epoch', return_value={'loss': 1.0}):
                with mock.patch.object(ec_solver, 'evaluate',
                                       return_value=({'coco_eval_bbox': [ap]}, None)):
                    # Accelerator cache management is unrelated to checkpoint flow.
                    with mock.patch.object(ec_solver.dist_utils, 'current_device',
                                           return_value='_checkpoint_test_no_accelerator'):
                        solver.fit()

    def test_last_checkpoint_reaches_final_no_augmentation_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            solver = self.make_solver(directory)
            self.run_fit(solver, ap=0.5)
            last = torch.load(Path(directory) / 'last.pth', weights_only=True)
            self.assertEqual(last['last_epoch'], 2)
            self.assertEqual(solver.loaded_checkpoints, ['best.pth'])

    def test_periodic_checkpoints_include_no_augmentation_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            solver = self.make_solver(directory)
            self.run_fit(solver, ap=0.5)
            for epoch in range(3):
                state = torch.load(Path(directory) / f'checkpoint{epoch:04}.pth',
                                   weights_only=True)
                self.assertEqual(state['last_epoch'], epoch)

    def test_zero_ap_best_checkpoint_can_start_no_augmentation_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            solver = self.make_solver(directory, epochs=2)
            self.run_fit(solver, ap=0.0)
            best = torch.load(Path(directory) / 'best.pth', weights_only=True)
            last = torch.load(Path(directory) / 'last.pth', weights_only=True)
            self.assertEqual(best['last_epoch'], 0)
            self.assertEqual(last['last_epoch'], 1)
            self.assertEqual(solver.loaded_checkpoints, ['best.pth'])


if __name__ == '__main__':
    unittest.main()
