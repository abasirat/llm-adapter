"""Offline checks of rates actually used by training and exact schedule resume."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import math
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


REPO = Path(os.environ.get('LLM_ADAPTER_REPO', Path(__file__).resolve().parents[1])).resolve()
if str(REPO / 'tests') not in sys.path:
    sys.path.insert(0, str(REPO / 'tests'))
# Import the module, rather than its TestCase class, to avoid duplicate discovery.
import test_training_resume as resume_checks


TRAIN = resume_checks.TRAIN


class TrainingScheduleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = resume_checks.TrainingResumeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def make_scheduler(self, *, warmup=2, total=6, base=0.01):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=base)
        scheduler = TRAIN.WarmUpCosineDecayScheduler(optimizer, warmup, total, base)
        return parameter, optimizer, scheduler

    def test_warmup_and_cosine_rates_are_prepared_before_each_update(self):
        parameter, optimizer, scheduler = self.make_scheduler()
        # Two warmup updates reach the peak once; five subsequent transitions
        # reach zero after all six successful updates, without a final no-op.
        expected = [0.005, 0.01, 0.009045084971874737, 0.006545084971874737,
                    0.003454915028125263, 0.0009549150281252633]
        actual = []
        for _ in range(6):
            actual.append(optimizer.param_groups[0]['lr'])
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=14)
        self.assertEqual(scheduler.step_count, 6)
        self.assertEqual(scheduler.get_lr(), 0.0)

    def test_no_warmup_starts_at_base_and_reaches_zero_after_last_update(self):
        _, optimizer, scheduler = self.make_scheduler(warmup=0, total=4)
        expected = [0.01, 0.008535533905932738, 0.005, 0.0014644660940672626]
        actual = []
        for _ in range(4):
            actual.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=14)
        self.assertEqual(scheduler.get_lr(), 0.0)

    def test_single_update_without_warmup_uses_base_rate(self):
        _, optimizer, scheduler = self.make_scheduler(warmup=0, total=1)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        scheduler.step()
        self.assertEqual(scheduler.get_lr(), 0.0)

    def test_scheduler_state_restores_the_upcoming_rate(self):
        _, _, scheduler = self.make_scheduler()
        for _ in range(3):
            scheduler.step()
        saved = copy.deepcopy(scheduler.state_dict())
        _, optimizer, restored = self.make_scheduler()
        optimizer.param_groups[0]['lr'] = 99.0
        restored.load_state_dict(saved)
        self.assertEqual(restored.step_count, 3)
        self.assertEqual(restored.get_lr(), scheduler.get_lr())
        self.assertEqual(restored.state_dict(), scheduler.state_dict())
        for _ in range(3):
            restored.step()
            scheduler.step()
            self.assertEqual(restored.get_lr(), scheduler.get_lr())

    def test_scheduler_state_rejects_changed_configuration(self):
        _, _, scheduler = self.make_scheduler()
        saved = scheduler.state_dict()
        for change in ({'warmup': 1}, {'total': 7}, {'base': 0.02}):
            with self.subTest(change=change):
                _, _, changed = self.make_scheduler(**change)
                with self.assertRaises(ValueError):
                    changed.load_state_dict(saved)

    def run_cpu(self, name, *, resume_state=None, model_state=None, interrupt=None, epochs=4, batch_size=3):
        actual_rates = []
        base_optimizer = TRAIN.AdamW

        class RecordingAdamW(base_optimizer):
            def step(self, *args, **kwargs):
                actual_rates.append(self.param_groups[0]['lr'])
                return super().step(*args, **kwargs)

        with mock.patch.object(TRAIN, 'AdamW', RecordingAdamW):
            result = self.fixture.run_training(
                name, self.fixture.make_dataset(), epochs=epochs, batch_size=batch_size,
                resume_state=resume_state, model_state=model_state, interrupt=interrupt,
            )
        return result, actual_rates

    def test_actual_train_logs_the_rate_used_before_during_and_after_warmup(self):
        result, actual = self.run_cpu('cpu')
        logged = [row[1] for row in result.logs]
        self.assertEqual(len(actual), 20)
        self.assertEqual(actual, logged)
        self.assertEqual(actual[:2], [0.005, 0.01])
        self.assertLess(actual[2], actual[1])
        self.assertGreater(actual[-1], 0.0)
        self.assertEqual(result.status['scheduler_step_count'], 20)
        self.assertEqual(result.status['_state']['scheduler_state_dict']['step_count'], 20)

    def test_actual_single_update_train_logs_base_rate(self):
        result, actual = self.run_cpu('single', epochs=1, batch_size=13)
        self.assertEqual(actual, [0.01])
        self.assertEqual([row[1] for row in result.logs], actual)

    def assert_equivalent(self, full, first, resumed, full_rates, first_rates, resumed_rates):
        self.assertEqual(full_rates, first_rates + resumed_rates)
        self.assertEqual(full.logs, first.logs + resumed.logs)
        self.assertEqual(full.model.seen_batches, first.model.seen_batches + resumed.model.seen_batches)
        self.assertEqual(full.model.losses, first.model.losses + resumed.model.losses)
        self.fixture.assert_tree_equal(full.weights, resumed.weights, 'weights')
        self.fixture.assert_tree_equal(full.rng, resumed.rng, 'rng')
        for key in ('optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict'):
            self.fixture.assert_tree_equal(full.status['_state'][key], resumed.status['_state'][key], key)

    def test_resume_after_warmup_matches_uninterrupted_optimizer_rates_and_weights(self):
        full, full_rates = self.run_cpu('full')
        first, first_rates = self.run_cpu('partial', interrupt=lambda state: state['epoch'] == 0 and state['batch_index'] == 3)
        resumed, resumed_rates = self.run_cpu('resumed', resume_state=first.status, model_state=first.weights)
        self.assert_equivalent(full, first, resumed, full_rates, first_rates, resumed_rates)

    def test_resume_rejects_checkpoints_with_prior_schedule_semantics(self):
        first, _ = self.run_cpu('partial', interrupt=lambda state: state['epoch'] == 0 and state['batch_index'] == 3)
        old = copy.deepcopy(first.status)
        old['_state']['resume_config']['lr_schedule_version'] = 1
        with self.assertRaisesRegex(ValueError, 'lr_schedule_version|configuration|schedule'):
            self.run_cpu('old', resume_state=old, model_state=first.weights)

    def test_resume_rejects_checkpoints_missing_schedule_version(self):
        first, _ = self.run_cpu('partial', interrupt=lambda state: state['epoch'] == 0 and state['batch_index'] == 3)
        old = copy.deepcopy(first.status)
        old['_state']['resume_config'].pop('lr_schedule_version', None)
        with self.assertRaisesRegex(ValueError, 'lr_schedule_version|configuration|schedule'):
            self.run_cpu('old_missing_version', resume_state=old, model_state=first.weights)

    def run_mock_amp(self, name, *, resume_state=None, model_state=None, interrupt=None):
        """Run the actual AMP branch on CPU, with deterministic GradScaler skips."""
        attempted_rates = []
        successful_rates = []
        scalers = []
        base_optimizer = TRAIN.AdamW
        original_tensor_to = torch.Tensor.to
        real_train = TRAIN.train

        class RecordingAdamW(base_optimizer):
            def step(self, *args, **kwargs):
                successful_rates.append(self.param_groups[0]['lr'])
                return super().step(*args, **kwargs)

        class ControlledGradScaler:
            def __init__(self, *args, **kwargs):
                self.attempt = 0
                self.scale_value = 128.0
                self.skipped = False
                scalers.append(self)

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def get_scale(self):
                return self.scale_value

            def step(self, optimizer):
                self.attempt += 1
                attempted_rates.append(optimizer.param_groups[0]['lr'])
                self.skipped = self.attempt in (2, 7)
                if not self.skipped:
                    return optimizer.step()

            def update(self):
                if self.skipped:
                    self.scale_value *= 0.5

            def state_dict(self):
                return {'attempt': self.attempt, 'scale': self.scale_value}

            def load_state_dict(self, state):
                self.attempt = state['attempt']
                self.scale_value = state['scale']

        def to_cpu_when_mocking_cuda(tensor, *args, **kwargs):
            args = list(args)
            if args and isinstance(args[0], (str, torch.device)) and torch.device(args[0]).type == 'cuda':
                args[0] = torch.device('cpu')
            if isinstance(kwargs.get('device'), (str, torch.device)) and torch.device(kwargs['device']).type == 'cuda':
                kwargs['device'] = torch.device('cpu')
            return original_tensor_to(tensor, *args, **kwargs)

        def route_amp(*args, **kwargs):
            kwargs['device'] = torch.device('cuda')
            return real_train(*args, **kwargs)

        with mock.patch.object(TRAIN, 'AdamW', RecordingAdamW), \
             mock.patch.object(TRAIN, 'train', side_effect=route_amp), \
             mock.patch.object(torch.amp, 'GradScaler', ControlledGradScaler), \
             mock.patch.object(torch, 'autocast', side_effect=lambda **kwargs: nullcontext()), \
             mock.patch.object(torch.Tensor, 'to', to_cpu_when_mocking_cuda):
            result = self.fixture.run_training(
                name, self.fixture.make_dataset(), epochs=4,
                resume_state=resume_state, model_state=model_state, interrupt=interrupt,
            )
        self.assertEqual(len(scalers), 1)
        return result, attempted_rates, successful_rates

    def test_mock_amp_skip_does_not_advance_learning_rate_schedule(self):
        result, attempted, successful = self.run_mock_amp('amp')
        self.assertEqual(len(attempted), 20)
        self.assertEqual(len(successful), 18)
        self.assertEqual(attempted[:3], [0.005, 0.01, 0.01])
        self.assertEqual(attempted[6], attempted[7])
        self.assertEqual([row[1] for row in result.logs], attempted)
        self.assertEqual(result.status['step'], 20)
        self.assertEqual(result.status['scheduler_step_count'], 18)
        self.assertEqual(result.status['_state']['scheduler_state_dict']['step_count'], 18)

    def test_resume_immediately_after_amp_skip_matches_uninterrupted_training(self):
        full, full_attempts, full_successful = self.run_mock_amp('full_amp')
        first, first_attempts, first_successful = self.run_mock_amp(
            'partial_amp', interrupt=lambda state: state['epoch'] == 0 and state['batch_index'] == 2)
        self.assertEqual(first.status['step'], 2)
        self.assertEqual(first.status['scheduler_step_count'], 1)
        resumed, resumed_attempts, resumed_successful = self.run_mock_amp(
            'resumed_amp', resume_state=first.status, model_state=first.weights)
        self.assertEqual(full_attempts, first_attempts + resumed_attempts)
        self.assert_equivalent(full, first, resumed, full_successful, first_successful, resumed_successful)


if __name__ == '__main__':
    unittest.main()
