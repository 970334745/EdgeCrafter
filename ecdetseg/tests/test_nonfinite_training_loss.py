"""Exercise the real CPU training loop with finite and non-finite losses."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.solver import ec_engine


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, samples, targets=None):
        return {'pred_boxes': (self.weight * samples).reshape(1, 1, 4)}


class BoxLoss(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, outputs, targets, **kwargs):
        return {'loss_box': outputs['pred_boxes'].square().mean() * self.factor}


class NonfiniteLossTests(unittest.TestCase):
    def run_epoch(self, model, optimizer, factor, scaler=None, ema=None, scheduler=None):
        batches = [(torch.ones(1, 4), [{'labels': torch.tensor([0])}])]
        return ec_engine.train_one_epoch(
            scheduler is not None, scheduler, model, BoxLoss(factor), batches,
            optimizer, torch.device('cpu'), epoch=0, print_freq=100,
            scaler=scaler, ema=ema)

    def assert_rejected_without_update(self, value, scaler=None):
        model = TinyDetector()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        before = model.weight.detach().clone()
        ema, scheduler = mock.Mock(), mock.Mock()
        with self.assertRaises((FloatingPointError, SystemExit)):
            self.run_epoch(model, optimizer, value, scaler, ema, scheduler)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        self.assertEqual(len(optimizer.state), 0)
        self.assertIsNone(model.weight.grad)
        ema.update.assert_not_called()
        scheduler.step.assert_not_called()

    def test_nan_loss_does_not_update(self):
        self.assert_rejected_without_update(float('nan'))

    def test_infinite_loss_does_not_update(self):
        self.assert_rejected_without_update(float('inf'))

    def test_nan_loss_with_scaler_does_not_update(self):
        self.assert_rejected_without_update(
            float('nan'), torch.amp.GradScaler('cpu', enabled=False))

    def test_infinite_loss_with_scaler_does_not_update(self):
        self.assert_rejected_without_update(
            float('inf'), torch.amp.GradScaler('cpu', enabled=False))

    def test_nonfinite_reduced_loss_stops_a_locally_finite_rank(self):
        with mock.patch.object(ec_engine.dist_utils, 'reduce_dict',
                               return_value={'loss_box': torch.tensor(float('nan'))}):
            self.assert_rejected_without_update(1.0)

    def test_finite_loss_keeps_the_expected_sgd_update(self):
        for use_scaler in (False, True):
            with self.subTest(scaler=use_scaler):
                model = TinyDetector()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                scaler = torch.amp.GradScaler('cpu', enabled=False) if use_scaler else None
                ema = mock.Mock()
                self.run_epoch(model, optimizer, 1.0, scaler=scaler, ema=ema)
                torch.testing.assert_close(model.weight.detach(), torch.tensor(0.8))
                ema.update.assert_called_once_with(model)


if __name__ == '__main__':
    unittest.main()
