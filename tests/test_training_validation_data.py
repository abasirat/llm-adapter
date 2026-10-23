"""Offline regression checks for explicit held-out token bins.

Run from the repository with ``python -m unittest discover -s tests``.
For a staged copy, set LLM_ADAPTER_REPO to the repository directory.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from functools import partial
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
import yaml


REPO = Path(os.environ.get('LLM_ADAPTER_REPO', Path(__file__).resolve().parents[1])).resolve()
TRAIN_SCRIPT = Path(os.environ.get('LLM_ADAPTER_TRAIN_SOURCE', REPO / 'experiments/legal_adaptation/scripts/train.py')).resolve()
PIPELINE = Path(os.environ.get('LLM_ADAPTER_PIPELINE_SOURCE', REPO / 'experiments/legal_adaptation/run_pipline.sh')).resolve()


def load_training_module():
    """Import the actual trainer while isolating optional logging/model dependencies."""
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
    spec = importlib.util.spec_from_file_location('heldout_training_under_test', TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {'wandb': wandb, 'llm_adapter': adapter, 'peft': peft}):
        spec.loader.exec_module(module)
    return module


TRAIN = load_training_module()


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def chunk_ids(dataset):
    return [tuple(dataset[index]['input_ids'].tolist()) for index in range(len(dataset))]


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


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(n_positions=32)
        self.train_inputs = []
        self.val_inputs = []

    def forward(self, input_ids, **kwargs):
        values = input_ids.detach().cpu().flatten().tolist()
        if self.training:
            self.train_inputs.extend(values)
            loss = (self.weight - input_ids.float().mean() / 1000).square()
        else:
            self.val_inputs.extend(values)
            loss = torch.tensor(1.0, device=self.weight.device)
        return SimpleNamespace(loss=loss)


class HeldoutValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_bin(self, name, values):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(values, dtype=np.uint16).tofile(path)
        return str(path)

    def build(self, train_paths, val_paths=(), *, val_fraction=0.5, seed=42, max_train_tokens=None):
        with redirect_stdout(io.StringIO()):
            return TRAIN.build_token_bin_datasets(
                train_bin_paths=list(train_paths),
                val_bin_paths=list(val_paths),
                context_size=4,
                val_fraction=val_fraction,
                seed=seed,
                max_train_tokens=max_train_tokens,
            )

    def test_explicit_validation_overrides_fraction_without_removing_training_chunks(self):
        train_path = self.write_bin('train.bin', range(16))
        val_path = self.write_bin('validation.bin', range(100, 108))
        training, validation = self.build([train_path], [val_path], val_fraction=0.75)
        self.assertEqual(chunk_ids(training), [tuple(range(start, start + 4)) for start in range(0, 16, 4)])
        self.assertEqual(chunk_ids(validation), [(100, 101, 102, 103), (104, 105, 106, 107)])

    def test_multiple_validation_bins_are_concatenated_and_repeatable(self):
        train_path = self.write_bin('train.bin', range(16))
        val_one = self.write_bin('val-one.bin', range(100, 108))
        val_two = self.write_bin('val-two.bin', range(200, 212))
        _, validation = self.build([train_path], [val_one, val_two])
        self.assertIsInstance(validation, ConcatDataset)
        expected = [(100, 101, 102, 103), (104, 105, 106, 107), (200, 201, 202, 203), (204, 205, 206, 207), (208, 209, 210, 211)]
        self.assertEqual(chunk_ids(validation), expected)
        np.random.seed(1234)
        self.assertEqual(chunk_ids(validation), expected)
        np.random.seed(9999)
        self.assertEqual(chunk_ids(validation), expected)

    def test_training_token_cap_does_not_cap_validation(self):
        train_path = self.write_bin('train.bin', range(64))
        val_path = self.write_bin('validation.bin', range(100, 164))
        training, validation = self.build([train_path], [val_path], max_train_tokens=16)
        self.assertEqual(len(training), 4)
        self.assertEqual(len(validation), 16)
        self.assertEqual(chunk_ids(validation)[-1], (160, 161, 162, 163))

    def test_legacy_random_split_preserves_seeded_indices(self):
        train_path = self.write_bin('train.bin', range(40))
        training, validation = self.build([train_path], val_fraction=0.3, seed=73)
        expected_order = torch.randperm(10, generator=torch.Generator().manual_seed(73)).tolist()
        self.assertEqual(training.indices, expected_order[:7])
        self.assertEqual(validation.indices, expected_order[7:])
        again_training, again_validation = self.build([train_path], val_fraction=0.3, seed=73)
        self.assertEqual(chunk_ids(training), chunk_ids(again_training))
        self.assertEqual(chunk_ids(validation), chunk_ids(again_validation))

    def test_legacy_zero_fraction_has_no_validation(self):
        train_path = self.write_bin('train.bin', range(16))
        training, validation = self.build([train_path], val_fraction=0.0)
        self.assertEqual(len(training), 4)
        self.assertIsNone(validation)

    def test_legacy_fraction_rounding_to_zero_keeps_training_and_skips_empty_validation(self):
        train_path = self.write_bin('train.bin', range(16))
        training, validation = self.build([train_path], val_fraction=0.01)
        self.assertEqual(len(training), 4)
        self.assertIsNone(validation)

    def test_missing_paths_are_rejected(self):
        train_path = self.write_bin('train.bin', range(16))
        val_path = self.write_bin('validation.bin', range(100, 108))
        missing = str(self.root / 'missing.bin')
        with self.assertRaises(FileNotFoundError):
            self.build([missing], [val_path])
        with self.assertRaises(FileNotFoundError):
            self.build([train_path], [missing])

    def test_empty_or_incomplete_validation_bins_are_rejected(self):
        train_path = self.write_bin('train.bin', range(16))
        for name, values in [('empty.bin', []), ('short.bin', [1, 2, 3])]:
            with self.subTest(name=name):
                val_path = self.write_bin(name, values)
                with self.assertRaises(ValueError):
                    self.build([train_path], [val_path])

    def test_empty_or_incomplete_training_bins_are_rejected(self):
        val_path = self.write_bin('validation.bin', range(100, 108))
        for name, values in [('empty.bin', []), ('short.bin', [1, 2, 3])]:
            with self.subTest(name=name):
                train_path = self.write_bin(name, values)
                with self.assertRaises(ValueError):
                    self.build([train_path], [val_path])

    def test_identical_and_alias_paths_cannot_be_both_training_and_validation(self):
        train_path = self.write_bin('train.bin', range(16))
        aliases = [train_path]
        symbolic = self.root / 'symbolic.bin'
        symbolic.symlink_to(train_path)
        aliases.append(str(symbolic))
        hard = self.root / 'hard.bin'
        os.link(train_path, hard)
        aliases.append(str(hard))
        for alias in aliases:
            with self.subTest(alias=alias):
                with self.assertRaises(ValueError):
                    self.build([train_path], [alias])
        with working_directory(self.root):
            with self.assertRaises(ValueError):
                self.build([train_path], ['train.bin'])

    def write_configs(self, *, train_paths, val_paths, base=None):
        base = base or self.root
        configs = {
            'model.yaml': {'model_name': 'tiny', 'freeze_lm_heads': True},
            'adapter.yaml': {'adapter_type': 'layer_adapter', 'prefix_length': 0, 'variational_modeling': False},
            'training.yaml': {
                'output_dir': str(base / 'models'), 'history_dir': str(base / 'history'),
                'context_size': 4, 'batch_size': 2, 'num_workers': 0,
                'val_fraction': 0.75, 'num_epochs': 1, 'wandb': {'enabled': False},
            },
            'data.yaml': {'train_bin_paths': train_paths, 'val_bin_paths': val_paths},
        }
        paths = {}
        for name, data in configs.items():
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(data))
            paths[name] = path
        return paths

    def call_main(self, argv):
        model = TinyModel()
        with mock.patch.object(sys, 'argv', [str(TRAIN_SCRIPT), *argv]), \
             mock.patch.object(TRAIN, 'setup_model', return_value=(model, object())), \
             mock.patch.object(TRAIN, 'set_device', return_value=torch.device('cpu')), \
             mock.patch.object(TRAIN, 'train', return_value=model) as call, \
             mock.patch.object(TRAIN, 'save_model'), \
             redirect_stdout(io.StringIO()):
            TRAIN.main()
        self.assertEqual(call.call_count, 1)
        return call.call_args.kwargs

    def test_main_reads_config_validation_and_cli_overrides_it(self):
        train_path = self.write_bin('train.bin', range(16))
        config_val = self.write_bin('config-validation.bin', range(100, 108))
        cli_val = self.write_bin('cli-validation.bin', range(200, 212))
        configs = self.write_configs(train_paths=[train_path], val_paths=[config_val])
        output = self.root / 'output/checkpoint'
        argv = [
            '--model_config', str(configs['model.yaml']),
            '--adapter_config', str(configs['adapter.yaml']),
            '--training_config', str(configs['training.yaml']),
            '--data_config', str(configs['data.yaml']),
            '--model_path', str(output),
        ]
        called = self.call_main(argv)
        self.assertEqual(len(called['train_dataloader'].dataset), 4)
        self.assertEqual(chunk_ids(called['val_dataloader'].dataset)[0], (100, 101, 102, 103))
        called = self.call_main([*argv, '--val_bin_path', cli_val])
        self.assertEqual(chunk_ids(called['val_dataloader'].dataset)[0], (200, 201, 202, 203))
        resolved = yaml.safe_load(Path(str(output) + '.resolved_config.yaml').read_text())
        self.assertEqual(resolved['resolved']['val_bin_paths'], [cli_val])

    def test_empty_cli_validation_path_is_rejected_instead_of_using_config(self):
        train_path = self.write_bin('train.bin', range(16))
        config_val = self.write_bin('config-validation.bin', range(100, 108))
        configs = self.write_configs(train_paths=[train_path], val_paths=[config_val])
        with self.assertRaises(FileNotFoundError):
            self.call_main([
                '--model_config', str(configs['model.yaml']),
                '--adapter_config', str(configs['adapter.yaml']),
                '--training_config', str(configs['training.yaml']),
                '--data_config', str(configs['data.yaml']),
                '--model_path', str(self.root / 'output/checkpoint'),
                '--val_bin_path', '',
            ])

    def test_empty_config_validation_path_is_rejected_instead_of_splitting(self):
        train_path = self.write_bin('train.bin', range(16))
        for val_paths in ([''], ''):
            with self.subTest(val_paths=val_paths):
                configs = self.write_configs(train_paths=[train_path], val_paths=val_paths)
                with self.assertRaises(FileNotFoundError):
                    self.call_main([
                        '--model_config', str(configs['model.yaml']),
                        '--adapter_config', str(configs['adapter.yaml']),
                        '--training_config', str(configs['training.yaml']),
                        '--data_config', str(configs['data.yaml']),
                        '--model_path', str(self.root / 'output/checkpoint'),
                    ])

    def test_null_validation_config_uses_legacy_split(self):
        train_path = self.write_bin('train.bin', range(16))
        configs = self.write_configs(train_paths=[train_path], val_paths=None)
        called = self.call_main([
            '--model_config', str(configs['model.yaml']),
            '--adapter_config', str(configs['adapter.yaml']),
            '--training_config', str(configs['training.yaml']),
            '--data_config', str(configs['data.yaml']),
            '--model_path', str(self.root / 'output/checkpoint'),
        ])
        self.assertEqual(len(called['train_dataloader'].dataset), 1)
        self.assertEqual(len(called['val_dataloader'].dataset), 3)

    def test_pipeline_passes_prepared_validation_to_main_validation_loader(self):
        script = PIPELINE.read_text()
        script, changed_data = re.subn(r'(?m)^data_prepare=true$', 'data_prepare=false', script)
        script, changed_train = re.subn(r'(?m)^train=false$', 'train=true', script)
        self.assertEqual((changed_data, changed_train), (1, 1), 'pipeline switches changed; update this offline fixture')
        # Process substitution opens /dev/fd, which restricted test runners may deny.
        # Remove only the output mirror from this temporary offline fixture.
        script = script.replace('  exec > >(tee -a "${LOG_FILE}") 2>&1', '  : # offline test logging')
        fake_bin = self.root / 'bin'
        fake_bin.mkdir()
        calls_path = self.root / 'python-calls.jsonl'
        launcher = fake_bin / 'python'
        launcher.write_text(
            '#!' + sys.executable + '\nimport json,sys\n'
            + 'with open(' + repr(str(calls_path)) + ', "a") as stream:\n'
            + '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        )
        launcher.chmod(0o755)
        pipeline = self.root / 'pipeline.sh'
        pipeline.write_text(script)
        environment = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ.get('PATH', ''))
        completed = subprocess.run(['bash', str(pipeline)], cwd=self.root, env=environment, text=True, capture_output=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
        self.assertTrue(calls)
        command = calls[0]
        self.assertEqual(command[0], 'scripts/train.py')
        self.assertIn('--val_bin_path', command)
        args = command[1:]
        flags = dict(zip(args[::2], args[1::2]))
        self.assertTrue(flags['--val_bin_path'].endswith('/prepared_data/val/data.bin'))
        with working_directory(self.root):
            Path(flags['--train_bin_path']).parent.mkdir(parents=True, exist_ok=True)
            np.arange(16, dtype=np.uint16).tofile(flags['--train_bin_path'])
            Path(flags['--val_bin_path']).parent.mkdir(parents=True, exist_ok=True)
            np.arange(100, 108, dtype=np.uint16).tofile(flags['--val_bin_path'])
            configs = self.write_configs(train_paths=[], val_paths=[])
            for flag, source in [('--model_config', configs['model.yaml']), ('--adapter_config', configs['adapter.yaml']), ('--training_config', configs['training.yaml'])]:
                target = Path(flags[flag])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source.read_text())
            called = self.call_main(args)
        self.assertEqual(len(called['train_dataloader'].dataset), 4)
        self.assertEqual(chunk_ids(called['val_dataloader'].dataset), [(100, 101, 102, 103), (104, 105, 106, 107)])

    def test_actual_training_uses_explicit_validation_for_early_stopping(self):
        train_path = self.write_bin('train.bin', range(16))
        val_path = self.write_bin('validation.bin', range(100, 108))
        training, validation = self.build([train_path], [val_path])
        collate = partial(TRAIN.token_bin_collate, context_size=4)
        train_loader = DataLoader(training, batch_size=1, shuffle=False, collate_fn=collate)
        val_loader = DataLoader(validation, batch_size=1, shuffle=False, collate_fn=collate)
        model = TinyModel()
        with mock.patch.object(torch, 'compile', side_effect=lambda value, **kwargs: value), \
             mock.patch.object(TRAIN, 'validate', wraps=TRAIN.validate) as validate, \
             mock.patch.object(TRAIN, 'save_model') as save_model, \
             mock.patch.object(TRAIN, 'save_training_status'), \
             mock.patch.object(TRAIN, 'tqdm', QuietProgress), \
             redirect_stdout(io.StringIO()):
            TRAIN.train(
                model=model, train_dataloader=train_loader, device=torch.device('cpu'),
                model_path=str(self.root / 'checkpoint'), learning_rate=1e-3,
                adapter_type='none', adapter_config=None, num_epochs=3,
                val_dataloader=val_loader, val_interval=1, progress_interval=100,
                early_stopping_patience=1,
            )
        self.assertEqual(validate.call_count, 2)
        self.assertTrue(all(call.args[1] is val_loader for call in validate.call_args_list))
        self.assertEqual(len(model.train_inputs), 8, 'training should stop after two batches')
        self.assertTrue(all(value < 100 for value in model.train_inputs))
        self.assertEqual(model.val_inputs, list(range(100, 108)) * 2)
        best_saves = [call for call in save_model.call_args_list if call.args[-1].endswith('-best')]
        self.assertEqual(len(best_saves), 1)

    def test_validation_loss_is_invariant_to_a_partial_final_batch(self):
        class MeanLossModel(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(loss=input_ids[:, 0].float().mean())

        path = self.write_bin('validation.bin', [2] * 4 + [4] * 4 + [10] * 4)
        dataset = TRAIN.TokenBinDataset(path, context_size=4)
        collate = partial(TRAIN.token_bin_collate, context_size=4)
        losses = []
        with mock.patch.object(TRAIN, 'tqdm', QuietProgress):
            for batch_size in (1, 2, 3):
                loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
                loss, _ = TRAIN.validate(MeanLossModel(), loader, torch.device('cpu'), 'cpu', False)
                losses.append(loss)
        for loss in losses:
            self.assertAlmostEqual(loss, 16 / 3, places=6)

    def test_validation_weights_unmasked_causal_targets_and_skips_empty_batches(self):
        class TargetLossModel(torch.nn.Module):
            def forward(self, input_ids, labels, **kwargs):
                valid = labels[:, 1:] != -100
                values = input_ids[:, 1:].float()[valid]
                return SimpleNamespace(loss=values.mean())

        loader = [
            ({'input_ids': torch.tensor([[9, 2, 2, 2]]), 'labels': torch.tensor([[-100, 0, 0, 0]])}, None),
            ({'input_ids': torch.tensor([[9, 10, 10, 10]]), 'labels': torch.tensor([[-100, 0, -100, -100]])}, None),
            ({'input_ids': torch.tensor([[9, 99, 99, 99]]), 'labels': torch.full((1, 4), -100)}, None),
        ]
        with mock.patch.object(TRAIN, 'tqdm', QuietProgress):
            loss, _ = TRAIN.validate(TargetLossModel(), loader, torch.device('cpu'), 'cpu', False)
        self.assertAlmostEqual(loss, 4.0)
        with mock.patch.object(TRAIN, 'tqdm', QuietProgress):
            with self.assertRaises(ValueError):
                TRAIN.validate(TargetLossModel(), loader[-1:], torch.device('cpu'), 'cpu', False)

    def test_validation_without_labels_weights_examples(self):
        class MeanLossModel(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(loss=input_ids[:, 0].float().mean())

        loader = [
            ({'input_ids': torch.tensor([[2, 2], [4, 4]])}, None),
            ({'input_ids': torch.tensor([[10, 10]])}, None),
        ]
        with mock.patch.object(TRAIN, 'tqdm', QuietProgress):
            loss, _ = TRAIN.validate(MeanLossModel(), loader, torch.device('cpu'), 'cpu', False)
        self.assertAlmostEqual(loss, 16 / 3, places=6)

    def test_nonfinite_validation_loss_raises_and_restores_prior_mode(self):
        class NonfiniteLossModel(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, input_ids, **kwargs):
                self.assert_validation_context()
                return SimpleNamespace(loss=torch.tensor(self.value))

            def assert_validation_context(self):
                if self.training or torch.is_grad_enabled():
                    raise AssertionError('validation must run in evaluation mode without gradients')

        loader = [({'input_ids': torch.ones((1, 4), dtype=torch.long), 'labels': torch.ones((1, 4), dtype=torch.long)}, None)]
        for value in (float('nan'), float('inf'), float('-inf')):
            for prior_training in (True, False):
                with self.subTest(value=value, prior_training=prior_training):
                    model = NonfiniteLossModel(value)
                    model.train(prior_training)
                    with mock.patch.object(TRAIN, 'tqdm', QuietProgress):
                        with self.assertRaisesRegex(ValueError, 'Validation loss is not finite'):
                            TRAIN.validate(model, loader, torch.device('cpu'), 'cpu', False)
                    self.assertIs(model.training, prior_training)

    def test_patience_zero_still_validates_and_saves_best_checkpoint(self):
        train_path = self.write_bin('train.bin', range(16))
        val_path = self.write_bin('validation.bin', range(100, 108))
        training, validation = self.build([train_path], [val_path])
        collate = partial(TRAIN.token_bin_collate, context_size=4)
        train_loader = DataLoader(training, batch_size=1, shuffle=False, collate_fn=collate)
        val_loader = DataLoader(validation, batch_size=1, shuffle=False, collate_fn=collate)
        model = TinyModel()
        with mock.patch.object(torch, 'compile', side_effect=lambda value, **kwargs: value), \
             mock.patch.object(TRAIN, 'validate', wraps=TRAIN.validate) as validate, \
             mock.patch.object(TRAIN, 'save_model') as save_model, \
             mock.patch.object(TRAIN, 'save_training_status'), \
             mock.patch.object(TRAIN, 'tqdm', QuietProgress), \
             redirect_stdout(io.StringIO()):
            TRAIN.train(
                model=model, train_dataloader=train_loader, device=torch.device('cpu'),
                model_path=str(self.root / 'checkpoint'), learning_rate=1e-3,
                adapter_type='none', adapter_config=None, num_epochs=1,
                val_dataloader=val_loader, val_interval=1, progress_interval=100,
                early_stopping_patience=0,
            )
        self.assertEqual(validate.call_count, 4)
        self.assertEqual(len(model.train_inputs), 16)
        best_saves = [call for call in save_model.call_args_list if call.args[-1].endswith('-best')]
        self.assertEqual(len(best_saves), 1)


if __name__ == '__main__':
    unittest.main()
