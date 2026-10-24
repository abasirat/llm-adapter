"""Offline post-training regression tests using real trainer functions and torch gradients.

Source overrides for staged verification: LLM_ADAPTER_POST_TRAIN_SOURCE and
LLM_ADAPTER_RANKING_SOURCE. No model or dataset downloads are performed.
"""

import ast
import contextlib
import copy
from functools import partial
import io
import unittest
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional, Sequence, cast
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / 'experiments' / 'legal_adaptation' / 'scripts'
POST_TRAIN_SOURCE = Path(os.environ.get(
    'LLM_ADAPTER_POST_TRAIN_SOURCE', str(SCRIPTS_ROOT / 'post_train_task.py'),
))
RANKING_SOURCE = Path(os.environ.get(
    'LLM_ADAPTER_RANKING_SOURCE', str(SCRIPTS_ROOT / 'post_train_ledgar_ranking.py'),
))


def extract(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


class Tokenizer:
    truncation_side = 'right'
    eos_token_id = 0
    pad_token_id = 0
    pad_token = '<pad>'
    eos_token = '<eos>'

    def __call__(self, text, **kwargs):
        ids = {' A': [3], ' B': [4], ' C': [5]}.get(text, [1, 2])
        return SimpleNamespace(input_ids=ids)


class TinyDataset(Dataset):
    label_names = ['A', 'B', 'C']

    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {'text': 'text', 'label_id': index}


class Progress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass


class FakeOptimizer:
    def __init__(self, parameters, **kwargs):
        self.parameters = list(parameters)
        self.gradients = None

    def zero_grad(self, **kwargs):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        self.gradients = [parameter.grad.clone() for parameter in self.parameters]
        with torch.no_grad():
            for parameter in self.parameters:
                parameter.add_(-0.1 * parameter.grad)


class TinyAdapter:
    def __init__(self, model):
        self.model = model

    def get_delta_loss(self):
        return self.model.strength.square()

    def get_kl_loss(self):
        return 3 * self.model.strength.square()

    def get_steering_metrics(self):
        return {'strength': self.model.strength.detach().clone(), 'delta_norm_mean': 2.,
                'base_norm_mean': 4., 'relative_norm_mean': .5,
                'relative_norm_p50': .5, 'relative_norm_p95': .5}


class TinyModel(torch.nn.Module):
    def __init__(self, adapter=True):
        super().__init__()
        self.strength = torch.nn.Parameter(torch.tensor(2.))
        self.forward_calls = 0
        if adapter:
            self.transformer = SimpleNamespace(encoder=TinyAdapter(self))

    def forward(self, input_ids, **kwargs):
        self.forward_calls += 1
        return SimpleNamespace(logits=self.strength * torch.zeros(*input_ids.shape, 8))


_OMITTED = object()


class PostTrainingStrengthTests(unittest.TestCase):
    def setUp(self):
        self.logs = []
        self.optimizers = []
        self.wandb = SimpleNamespace(
            run=object(), log=lambda values, **kwargs: self.logs.append(copy.deepcopy(values)),
            init=lambda **kwargs: None, finish=lambda: None,
        )

        def optimizer_factory(parameters, **kwargs):
            optimizer = FakeOptimizer(parameters, **kwargs)
            self.optimizers.append(optimizer)
            return optimizer

        self.namespace = dict(
            torch=torch, F=F, math=math, random=random, Dict=Dict, List=List,
            Sequence=Sequence, cast=cast, DataLoader=DataLoader,
            LEDGARRankingDataset=TinyDataset, AdamW=optimizer_factory, tqdm=Progress,
            save_model=lambda *args, **kwargs: None, wandb=self.wandb,
        )
        extract(RANKING_SOURCE, {
            'verbalize_label', 'ranking_collate', '_as_int', '_as_float', '_as_str',
            '_sample_negative_label_ids', '_build_label_token_cache', '_build_scoring_batch',
            '_score_candidates', 'run_ledgar_label_ranking_post_training',
        }, self.namespace)
        self.ranking = self.namespace['run_ledgar_label_ranking_post_training']
        self.training_cfg = dict(
            batch_size=2, num_epochs=1, num_workers=0, progress_interval=1,
            ranking_num_negatives=2, ranking_margin=0., gradient_clip_norm=1000.,
            use_amp=False, weight_decay=0., kl_loss_weight=.25,
        )
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def run_ranking(self, weight=_OMITTED, enabled=True, adapter=True, variational=False,
                    metrics_available=True):
        self.logs.clear()
        model = TinyModel(adapter=adapter)
        if adapter and not metrics_available:
            model.transformer.encoder.get_steering_metrics = None
        kwargs = dict(
            model=model, tokenizer=Tokenizer(), adapter_type='layer_adapter' if adapter else 'none',
            adapter_config={}, training_cfg=self.training_cfg, data_cfg={},
            device=torch.device('cpu'), shift_regularization=enabled,
            aggregation_strategy='mean', variational_modeling=variational,
        )
        if weight is not _OMITTED:
            kwargs['shift_regularization_weight'] = weight
        with tempfile.TemporaryDirectory() as output_directory:
            self.ranking(**kwargs, output_dir=output_directory)
        gradient = self.optimizers[-1].gradients[0].item()
        return model, gradient, copy.deepcopy(self.logs[-1]) if self.logs else None

    def assert_weighted_step(self, weight, enabled, expected_gradient):
        model, gradient, log = self.run_ranking(weight, enabled)
        effective_weight = weight if enabled else 0.
        self.assertAlmostEqual(gradient, expected_gradient, places=6)
        self.assertEqual(log['train/ranking_loss'], 0.)
        self.assertEqual(log['train/shift_regularization_weight'], effective_weight)
        self.assertEqual(log['train/shift_penalty'], 4 * effective_weight)
        self.assertEqual(log['train/total_loss'], log['train/shift_penalty'])
        self.assertEqual(log['steering/strength'], 2., 'diagnostic must precede optimizer update')
        self.assertEqual(model.forward_calls, 1, 'training must score the complete candidate batch')

    def test_zero_weight_disables_penalty_and_its_gradient(self):
        self.assert_weighted_step(0., True, 0.)

    def test_positive_weight_scales_penalty_and_gradient(self):
        for weight, expected in ((.5, 2.), (2., 8.)):
            with self.subTest(weight=weight):
                self.assert_weighted_step(weight, True, expected)

    def test_existing_disabled_flag_keeps_penalty_disabled(self):
        self.assert_weighted_step(2., False, 0.)

    def test_default_weight_preserves_previous_unit_penalty(self):
        _, gradient, log = self.run_ranking()
        self.assertEqual(gradient, 4.)
        self.assertEqual(log['train/shift_regularization_weight'], 1.)
        self.assertEqual(log['train/shift_penalty'], 4.)

    def test_magnitude_and_kl_penalties_do_not_mutate_ranking_loss_logging(self):
        _, gradient, log = self.run_ranking(.5, variational=True)
        self.assertEqual(gradient, 5.)
        self.assertEqual(log['train/ranking_loss'], 0.)
        self.assertEqual(log['train/total_loss'], 5.)

    def test_non_adapter_model_needs_no_transformer_encoder(self):
        _, gradient, log = self.run_ranking(2., adapter=False)
        self.assertEqual(gradient, 0.)
        self.assertFalse(any(key.startswith('steering/') for key in log))

    def test_adapter_without_diagnostics_getter_remains_supported(self):
        _, gradient, log = self.run_ranking(.5, metrics_available=False)
        self.assertEqual(gradient, 2.)
        self.assertFalse(any(key.startswith('steering/') for key in log))

    def test_ranking_does_not_require_active_wandb_run(self):
        self.wandb.run = None
        _, gradient, log = self.run_ranking(.5)
        self.assertEqual(gradient, 2.)
        self.assertIsNone(log)

    def test_invalid_ranking_weights_fail_before_optimizer_creation(self):
        for invalid in (-1, float('nan'), float('inf'), 'bad', None, 10**1000):
            with self.subTest(value_type=type(invalid).__name__):
                with self.assertRaisesRegex(ValueError, 'shift_regularization_weight'):
                    self.run_ranking(invalid)
        self.assertEqual(self.optimizers, [])

    def post_training_namespace(self):
        namespace = dict(
            os=os, sys=sys, math=math, Path=Path, Optional=Optional, torch=torch,
            DataLoader=DataLoader, random_split=random_split, partial=partial,
            _SCRIPTS_DIR=POST_TRAIN_SOURCE.parent, set_seed=lambda seed: None, wandb=self.wandb,
            LEDGARGenerationDataset=TinyDataset, supervised_lm_collate=lambda *a, **k: None,
        )
        extract(POST_TRAIN_SOURCE, {'run_post_training'}, namespace)
        return namespace

    def check_post_training_route(self, strategy, weight=_OMITTED):
        captured = []

        def trainer(**kwargs):
            captured.append(copy.deepcopy({key: value for key, value in kwargs.items()
                                           if key not in {'model', 'tokenizer', 'train_dataloader', 'val_dataloader'}}))
            target = (Path(kwargs['output_dir']) / 'checkpoint-best' if 'output_dir' in kwargs
                      else Path(kwargs['model_path'] + '-best'))
            target.write_text('checkpoint')

        adapter_config = {'shift_regularization': True, 'aggregation_strategy': 'mean'}
        namespace = self.post_training_namespace()
        namespace.update(
            _llm_load_model=lambda path: (TinyModel(), Tokenizer(), adapter_config),
            run_ledgar_label_ranking_post_training=trainer, train=trainer,
        )
        utilities = ModuleType('evaluations.utils')
        utilities.select_device = lambda device: 'cpu'
        preparation = ModuleType('prepare_dataset')
        preparation.run_prepare_dataset = lambda cfg: None
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(sys.modules, {'evaluations.utils': utilities, 'prepare_dataset': preparation}), \
             patch.object(torch, 'load', return_value={'adapter_type': 'layer_adapter'}):
            source = Path(directory) / 'source.pt'
            source.touch()
            output = Path(directory) / 'output'
            output.mkdir()
            config = {'strategy': strategy, 'val_fraction': 0.}
            if weight is not _OMITTED:
                config['shift_regularization_weight'] = weight
            checkpoint = namespace['run_post_training'](
                str(source), config, {'task': 'ledgar'}, str(output), 'cpu',
            )
            self.assertTrue(Path(checkpoint).is_file())
            self.assertEqual(captured[0]['shift_regularization_weight'], 1. if weight is _OMITTED else weight)
            self.assertEqual(adapter_config, {'shift_regularization': True, 'aggregation_strategy': 'mean'})

    def test_weight_reaches_both_post_training_routes_without_adapter_config_mutation(self):
        for strategy in ('label_ranking', 'supervised_lm'):
            with self.subTest(strategy=strategy):
                self.check_post_training_route(strategy, .125)

    def test_missing_weight_defaults_to_one_in_both_post_training_routes(self):
        for strategy in ('label_ranking', 'supervised_lm'):
            with self.subTest(strategy=strategy):
                self.check_post_training_route(strategy)

    def test_invalid_post_training_weights_fail_before_preparation_or_loading(self):
        namespace = self.post_training_namespace()
        for invalid in (-1, float('nan'), float('inf'), 'bad', None, 10**1000):
            with self.subTest(value_type=type(invalid).__name__):
                with self.assertRaisesRegex(ValueError, 'shift_regularization_weight'):
                    namespace['run_post_training'](
                        'unused', {'shift_regularization_weight': invalid}, {}, 'unused',
                    )


if __name__ == '__main__':
    unittest.main(verbosity=2)
