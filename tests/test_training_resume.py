"""Offline end-to-end regression checks for consumed-batch training resume."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
from functools import partial
import importlib.util
import io
import multiprocessing
import os
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO = Path(os.environ.get('LLM_ADAPTER_REPO', Path(__file__).resolve().parents[1])).resolve()
SOURCE = Path(os.environ.get('LLM_ADAPTER_TRAIN_SOURCE', REPO / 'experiments/legal_adaptation/scripts/train.py')).resolve()


def load_training_module():
    wandb = ModuleType('wandb')
    for name in ('init', 'log', 'save', 'finish', 'Table'):
        setattr(wandb, name, mock.Mock())
    wandb.plot = SimpleNamespace(bar=mock.Mock())
    adapter = ModuleType('llm_adapter')
    adapter.setup_model = mock.Mock()
    adapter.save_model = mock.Mock()
    adapter.load_model = mock.Mock()
    peft = ModuleType('peft')
    peft.LoraConfig = mock.Mock()
    peft.PromptEncoderConfig = mock.Mock()
    peft.TaskType = SimpleNamespace(CAUSAL_LM='CAUSAL_LM')
    spec = importlib.util.spec_from_file_location('resume_training_under_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with mock.patch.dict(sys.modules, {'wandb': wandb, 'llm_adapter': adapter, 'peft': peft}):
        spec.loader.exec_module(module)
    return module


TRAIN = load_training_module()


class QuietProgress:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, *args):
        pass

    def set_description(self, *args):
        pass

    def close(self):
        pass


class TrainingInterrupted(RuntimeError):
    pass


class CountingTokenBinDataset(TRAIN.TokenBinDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_indices = []

    def __getitem__(self, index):
        self.read_indices.append(int(index))
        return super().__getitem__(index)


class TinySteeringEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)
        self.dropout = torch.nn.Dropout(0.25)
        self.mean_projection = torch.nn.Linear(4, 4)
        self.logvar_projection = torch.nn.Linear(4, 4)
        self.temperatures = []
        self.layer_attention_metrics = {
            'entropy_of_layer_attention': torch.tensor(0.69),
            'avg_attention_to_each_layer': torch.tensor([0.4, 0.6]),
        }

    def set_attention_temperature(self, temperature):
        self.temperatures.append(temperature)

    def forward(self, x):
        deterministic = self.dropout(torch.tanh(self.projection(x)))
        self.mu = self.mean_projection(deterministic)
        self.logvar = self.logvar_projection(deterministic).clamp(-3, 3)
        self.std = torch.exp(0.5 * self.logvar)
        self.delta = self.mu + self.std * torch.randn_like(self.std)
        # Exercise all three saved process RNGs, independently of data sampling.
        self.delta = self.delta + (np.random.random() + random.random()) * 1e-3
        self.kl = 0.5 * (self.mu.square() + self.logvar.exp() - 1 - self.logvar).mean()
        return self.delta

    def get_delta_loss(self):
        return self.delta.square().mean()

    def get_kl_loss(self):
        return self.kl

    def get_variational_stats(self):
        return self.mu, self.std


class TinyStochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.encoder = TinySteeringEncoder()
        self.config = SimpleNamespace(n_layer=2)
        self.seen_batches = []
        self.losses = []

    def forward(self, input_ids, **kwargs):
        self.seen_batches.append(tuple(input_ids[:, 0].detach().cpu().tolist()))
        x = input_ids.float() / 100
        delta = self.transformer.encoder(x)
        loss = (delta - 0.25 * x).square().mean()
        self.losses.append(loss.detach().item())
        return SimpleNamespace(loss=loss)


class ConstantValidationModel(TinyStochasticModel):
    def forward(self, input_ids, **kwargs):
        if not self.training:
            return SimpleNamespace(loss=torch.tensor(1.0))
        return super().forward(input_ids, **kwargs)


def compile_identity(model, **kwargs):
    # Setup work must not perturb the restored model RNG on a resumed run.
    torch.rand(3)
    return model


def save_tiny_model(model, adapter_type, adapter_config, save_path):
    torch.save(model.state_dict(), save_path)


def rng_snapshot():
    return {'torch': torch.get_rng_state().clone(), 'numpy': np.random.get_state(), 'python': random.getstate()}


class TrainingResumeTests(unittest.TestCase):
    seed = 73

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = []
        for name, first, count in [('first.bin', 1, 13), ('second.bin', 101, 9)]:
            path = self.root / name
            np.repeat(np.arange(first, first + count, dtype=np.uint16), 4).tofile(path)
            self.paths.append(str(path))

    def make_dataset(self, mixed=False, changed=False):
        paths = list(self.paths)
        if changed:
            path = self.root / 'changed.bin'
            np.repeat(np.arange(1, 15, dtype=np.uint16), 4).tofile(path)
            paths[0] = str(path)
        with redirect_stdout(io.StringIO()):
            datasets = [CountingTokenBinDataset(path, context_size=4) for path in paths[:2 if mixed else 1]]
        return TRAIN.MixedDataset(datasets, probabilities=[0.65, 0.35]) if mixed else datasets[0]

    def make_loader(self, dataset, *, batch_size=3, workers=0):
        options = dict(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            collate_fn=partial(TRAIN.token_bin_collate, context_size=4),
            num_workers=workers, persistent_workers=workers > 0,
        )
        if workers:
            options.update(prefetch_factor=3, multiprocessing_context='fork')
        return DataLoader(**options)

    def run_training(self, name, dataset, *, resume_state=None, model_state=None, interrupt=None,
                     batch_size=3, workers=0, learning_rate=0.01, seed=None, epochs=2,
                     progress_interval=1, val_interval=1, patience=0, validation_loader=None,
                     model_class=TinyStochasticModel):
        output = self.root / name
        output.mkdir(exist_ok=True)
        TRAIN.set_seed(self.seed)
        model = model_class()
        if model_state is not None:
            model.load_state_dict(model_state)
        loader = self.make_loader(dataset, batch_size=batch_size, workers=workers)
        logs = []
        statuses = []
        real_save = TRAIN.save_training_status

        def save_and_maybe_interrupt(*args, **kwargs):
            real_save(*args, **kwargs)
            statuses.append({key: value for key, value in kwargs.items() if key in ('epoch', 'step', 'batch_index')})
            if interrupt is not None and interrupt(kwargs):
                raise TrainingInterrupted('interrupted immediately after durable checkpoint')

        def log(values, step=None, **kwargs):
            # Exercise checkpoint placement around callbacks that consume RNG.
            torch.rand(1)
            np.random.random()
            random.random()
            if 'train/learning_rate' in values:
                logs.append((step, values['train/learning_rate'], values.get('variational/kl_weight'),
                             values.get('layer_attn/temperature'), values.get('train/total_loss'),
                             values.get('train/running_loss'), values.get('train/lm_loss')))

        try:
            with mock.patch.object(torch, 'compile', side_effect=compile_identity), \
                 mock.patch.object(TRAIN, 'save_model', side_effect=save_tiny_model), \
                 mock.patch.object(TRAIN, 'save_training_status', side_effect=save_and_maybe_interrupt), \
                 mock.patch.object(TRAIN.wandb, 'log', side_effect=log), \
                 mock.patch.object(TRAIN, 'tqdm', QuietProgress), \
                 redirect_stdout(io.StringIO()):
                TRAIN.train(
                    model=model, train_dataloader=loader, device=torch.device('cpu'),
                    model_path=str(output / 'checkpoint'), learning_rate=learning_rate,
                    adapter_type='layer_adapter', adapter_config={}, variational_modeling=True,
                    num_epochs=epochs, early_stopping_patience=patience, progress_interval=progress_interval,
                    val_interval=val_interval, val_dataloader=validation_loader,
                    kl_loss_weight=0.025, kl_warmup_fraction=0.5,
                    layer_adapter_max_temperature=2.0, layer_adapter_min_temperature=0.8,
                    aggregation_strategy='attention', resume_state=resume_state,
                    seed=self.seed if seed is None else seed,
                )
        except TrainingInterrupted:
            self.assertIsNotNone(interrupt)
            stopped = True
        else:
            stopped = False
        finally:
            iterator = getattr(loader, '_iterator', None)
            if iterator is not None:
                iterator._shutdown_workers()
        snapshot = rng_snapshot()
        status_path = output / 'checkpoint.training_status.json'
        if status_path.exists():
            status = TRAIN.load_training_status(str(status_path))
            weights = torch.load(status['checkpoint_path'], map_location='cpu', weights_only=True)
        else:
            self.assertIsNotNone(resume_state, 'a fresh training run must write its status')
            status = resume_state
            weights = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        return SimpleNamespace(model=model, logs=logs, statuses=statuses, status=status, weights=weights, rng=snapshot, stopped=stopped)

    def assert_tree_equal(self, expected, actual, path='root'):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual), path)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual, err_msg=path)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual), path)
            for key in expected:
                self.assert_tree_equal(expected[key], actual[key], f'{path}.{key}')
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual), path)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_tree_equal(left, right, f'{path}[{index}]')
        else:
            self.assertEqual(expected, actual, path)

    def assert_resume_equivalent(self, *, mixed=False, workers=0, interruption='middle', progress_interval=1, val_interval=1):
        full_dataset = self.make_dataset(mixed=mixed)
        full = self.run_training('full', full_dataset, workers=workers, progress_interval=progress_interval, val_interval=val_interval)
        partial_dataset = self.make_dataset(mixed=mixed)
        if interruption == 'boundary':
            stop = lambda status: status['epoch'] == 1 and status['batch_index'] == 0
        elif interruption == 'last_batch':
            expected_batches = len(self.make_loader(partial_dataset))
            stop = lambda status: status['epoch'] == 0 and status['batch_index'] == expected_batches
        else:
            stop_batch = 2 if val_interval == 2 else 3
            stop = lambda status: status['epoch'] == 0 and status['batch_index'] == stop_batch
        first = self.run_training('interrupted', partial_dataset, workers=workers, interrupt=stop, progress_interval=progress_interval, val_interval=val_interval)
        self.assertTrue(first.stopped)
        resumed_dataset = self.make_dataset(mixed=mixed)
        resumed = self.run_training('resumed', resumed_dataset, workers=workers, resume_state=first.status, model_state=first.weights, progress_interval=progress_interval, val_interval=val_interval)
        self.assertFalse(resumed.stopped)
        self.assertEqual(full.model.seen_batches, first.model.seen_batches + resumed.model.seen_batches)
        self.assertEqual(full.model.losses, first.model.losses + resumed.model.losses)
        self.assertEqual(full.logs, first.logs + resumed.logs)
        self.assertEqual(full.model.transformer.encoder.temperatures, first.model.transformer.encoder.temperatures + resumed.model.transformer.encoder.temperatures)
        self.assert_tree_equal(full.weights, resumed.weights, 'final_weights')
        self.assert_tree_equal(full.rng, resumed.rng, 'final_rng')
        self.assert_tree_equal(full.status['_state']['optimizer_state_dict'], resumed.status['_state']['optimizer_state_dict'], 'optimizer')
        for key in ('epoch', 'step', 'batch_index', 'scheduler_step_count', 'kl_scheduler_step_count', 'temp_scheduler_step_count'):
            self.assertEqual(full.status[key], resumed.status[key], key)
        if not mixed and not workers:
            self.assertEqual(full_dataset.read_indices, partial_dataset.read_indices + resumed_dataset.read_indices, 'resuming must not replay processed __getitem__ calls')
        return first, resumed

    def test_mid_epoch_resume_matches_uninterrupted_stochastic_training(self):
        self.assert_resume_equivalent()

    def test_mixed_dataset_resume_matches_uninterrupted_stochastic_training(self):
        self.assert_resume_equivalent(mixed=True)

    def test_last_batch_checkpoint_transitions_to_next_epoch_exactly(self):
        self.assert_resume_equivalent(interruption='last_batch')

    def test_epoch_boundary_checkpoint_transitions_to_next_epoch_exactly(self):
        self.assert_resume_equivalent(interruption='boundary')

    def test_checkpoint_between_logging_steps_preserves_accumulated_metrics(self):
        self.assert_resume_equivalent(progress_interval=3, val_interval=2)

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'fork workers unavailable')
    def test_prefetch_workers_resume_from_consumed_batches(self):
        self.assert_resume_equivalent(workers=2)

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'fork workers unavailable')
    def test_mixed_dataset_prefetch_workers_are_reproducible(self):
        self.assert_resume_equivalent(mixed=True, workers=2)

    def interrupted(self):
        return self.run_training('interrupted', self.make_dataset(), interrupt=lambda status: status['epoch'] == 0 and status['batch_index'] == 3)

    def test_changed_dataset_is_rejected(self):
        first = self.interrupted()
        with self.assertRaisesRegex(ValueError, 'dataset|Dataset|fingerprint|configuration|resume'):
            self.run_training('changed', self.make_dataset(changed=True), resume_state=first.status, model_state=first.weights)

    def test_changed_token_contents_at_the_same_path_are_rejected(self):
        first = self.interrupted()
        np.repeat(np.arange(31, 44, dtype=np.uint16), 4).tofile(self.paths[0])
        with self.assertRaisesRegex(ValueError, 'dataset|Dataset|fingerprint|configuration|resume|file'):
            self.run_training('changed', self.make_dataset(), resume_state=first.status, model_state=first.weights)

    def test_changed_batch_size_is_rejected(self):
        first = self.interrupted()
        with self.assertRaisesRegex(ValueError, 'batch|Batch|configuration|resume'):
            self.run_training('changed', self.make_dataset(), batch_size=2, resume_state=first.status, model_state=first.weights)

    def test_changed_optimizer_or_schedule_configuration_is_rejected(self):
        first = self.interrupted()
        with self.assertRaisesRegex(ValueError, 'learning|configuration|resume|schedule|optimizer'):
            self.run_training('changed', self.make_dataset(), learning_rate=0.02, resume_state=first.status, model_state=first.weights)

    def test_changed_seed_is_rejected(self):
        first = self.interrupted()
        with self.assertRaisesRegex(ValueError, 'seed|configuration|resume'):
            self.run_training('changed', self.make_dataset(), seed=74, resume_state=first.status, model_state=first.weights)

    def test_status_round_trip_preserves_sampler_and_resume_configuration(self):
        first = self.interrupted()
        state = torch.load(first.status['state_path'], map_location='cpu', weights_only=False)
        self.assertIn('sampler', state['data_state'])
        self.assertTrue(state['data_state']['sampler'])
        reloaded = TRAIN.load_training_status(str(self.root / 'interrupted/checkpoint.training_status.json'))
        self.assert_tree_equal(state['data_state'], reloaded['_state']['data_state'], 'saved_sampler')
        self.assert_tree_equal(state['resume_config'], reloaded['_state']['resume_config'], 'saved_configuration')
        self.assertEqual(reloaded['epoch'], 0)
        self.assertEqual(reloaded['batch_index'], 3)

    def test_modified_checkpoint_file_is_rejected(self):
        first = self.interrupted()
        with open(first.status['checkpoint_path'], 'ab') as checkpoint:
            checkpoint.write(b'tampered checkpoint')
        with self.assertRaisesRegex(ValueError, 'match|integrity|hash|incomplete|overwritten'):
            TRAIN.load_training_status(str(self.root / 'interrupted/checkpoint.training_status.json'))

    def test_modified_optimizer_state_file_is_rejected(self):
        first = self.interrupted()
        with open(first.status['state_path'], 'ab') as state:
            state.write(b'tampered optimizer state')
        with self.assertRaisesRegex(ValueError, 'match|integrity|hash|incomplete|overwritten'):
            TRAIN.load_training_status(str(self.root / 'interrupted/checkpoint.training_status.json'))

    def test_early_stopped_checkpoint_does_not_perform_additional_updates(self):
        def validation_loader():
            with redirect_stdout(io.StringIO()):
                dataset = CountingTokenBinDataset(self.paths[1], context_size=4)
            return DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=partial(TRAIN.token_bin_collate, context_size=4))

        first = self.run_training('early_stopped', self.make_dataset(), patience=1,
                                  validation_loader=validation_loader(), model_class=ConstantValidationModel)
        self.assertTrue(first.status['early_stopped'])
        self.assertEqual(first.status['step'], 2)
        resumed = self.run_training('terminal_resume', self.make_dataset(), patience=1,
                                    validation_loader=validation_loader(), model_class=ConstantValidationModel,
                                    resume_state=first.status, model_state=first.weights)
        self.assertEqual(resumed.model.seen_batches, [])
        self.assertEqual(resumed.logs, [])
        self.assertEqual(resumed.statuses, [])
        self.assert_tree_equal(first.weights, resumed.weights, 'terminal_weights')

    def test_legacy_mid_epoch_checkpoint_without_sampler_state_is_rejected(self):
        first = self.interrupted()
        legacy = copy.deepcopy(first.status)
        # Keep optimizer/RNG state but remove every newly introduced resume descriptor.
        legacy['_state'] = {key: value for key, value in legacy['_state'].items() if key in ('optimizer_state_dict', 'scaler_state_dict', 'rng_state')}
        legacy = {key: value for key, value in legacy.items() if key in (
            'checkpoint_path', 'best_checkpoint_path', 'state_path', 'epoch', 'step', 'batch_index',
            'best_val_loss', 'patience_counter', 'scheduler_step_count', 'kl_scheduler_step_count',
            'temp_scheduler_step_count', 'epoch_total_loss', 'epoch_num_batches', 'timestamp', '_state',
        )}
        with self.assertRaisesRegex(ValueError, 'mid.epoch|sampler|legacy|Legacy|resume'):
            self.run_training('legacy', self.make_dataset(), resume_state=legacy, model_state=first.weights)


if __name__ == '__main__':
    unittest.main()
