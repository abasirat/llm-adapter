"""Offline steering-strength tests using real tiny GPT-2 weights and the trainer."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
from functools import partial
import importlib.util
import inspect
import io
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2Model


REPO = Path(os.environ.get('LLM_ADAPTER_REPO', Path(__file__).resolve().parents[1])).resolve()
PACKAGE = REPO / 'src/llm_adapter/autoregressive_adapt'
LAYER_SOURCE = Path(os.environ.get('LLM_ADAPTER_LAYER_SOURCE', PACKAGE / 'layer_adapter6.py'))
SETUP_SOURCE = Path(os.environ.get('LLM_ADAPTER_MODEL_SETUP_SOURCE', PACKAGE / 'model_setup.py'))


def import_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


try:
    import transformers.cache_utils  # noqa: F401
except ImportError:
    # Transformers 4.32 can exercise this feature's real GPT-2 computation.
    # These tests disable prefix/cache use; do not emulate any cache behavior.
    unused_cache = ModuleType('transformers.cache_utils')

    class UnusedDynamicCache:
        def __init__(self, *args, **kwargs):
            raise AssertionError('steering tests must not construct a prefix cache')

    unused_cache.DynamicCache = UnusedDynamicCache
    with mock.patch.dict(sys.modules, {'transformers.cache_utils': unused_cache}):
        LAYER = import_source('steering_strength_layer_under_test', LAYER_SOURCE)
else:
    LAYER = import_source('steering_strength_layer_under_test', LAYER_SOURCE)


class LocalGPT2Model(GPT2Model):
    def forward(self, *args, cache_position=None, **kwargs):
        # Drop only the newer, unused cache keyword on old installed runtimes.
        if 'cache_position' in inspect.signature(GPT2Model.forward).parameters:
            kwargs['cache_position'] = cache_position
        return super().forward(*args, **kwargs)


def config_values(**changes):
    values = dict(prefix_length=0, dropout=0.0, qk_dim=4, v_dim=8,
                  num_attention_heads=2, representation_type='pre_mlp',
                  aggregation_strategy='attention', need_weights=True)
    values.update(changes)
    return values


def make_adapter(**changes):
    config = GPT2Config(vocab_size=37, n_positions=16, n_embd=16, n_layer=2,
                       n_head=2, resid_pdrop=0.2, embd_pdrop=0.2, attn_pdrop=0.2)
    config._name_or_path = 'gpt2'
    return LAYER.LayerAdapter(LocalGPT2Model(config), **config_values(**changes))


def make_setup_module():
    package = ModuleType('steering_strength_setup_package')
    package.__path__ = [str(PACKAGE)]
    peft = ModuleType('peft')
    peft.get_peft_model = mock.Mock()
    wechsel = ModuleType('wechsel')
    wechsel.WECHSEL = mock.Mock()
    wechsel.load_embeddings = mock.Mock()
    with mock.patch.dict(sys.modules, {
        package.__name__: package,
        package.__name__ + '.layer_adapter6': LAYER,
        'peft': peft, 'wechsel': wechsel,
    }):
        return import_source(package.__name__ + '.model_setup', SETUP_SOURCE)


SETUP = make_setup_module()


class CheckpointModel(torch.nn.Module):
    def __init__(self, adapter_config):
        super().__init__()
        torch.manual_seed(217)  # stand in for deterministic pretrained base weights
        self.transformer = torch.nn.Module()
        self.transformer.encoder = make_adapter(**adapter_config)
        self.transformer.num_tailor_layers = 0
        self.config = self.transformer.encoder.config


class SteeringStrengthTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        self.mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    def forward(self, adapter, ids=None, mask=None):
        return adapter(input_ids=self.ids if ids is None else ids,
                       attention_mask=self.mask if mask is None else mask,
                       use_cache=False, return_dict=True).last_hidden_state

    def test_default_and_configured_strength_are_learnable_parameters(self):
        for changes, expected in [({}, 0.01), ({'steering_strength_init': 0.25}, 0.25)]:
            with self.subTest(changes=changes):
                adapter = make_adapter(**changes)
                self.assertIsInstance(adapter.steering_strength, torch.nn.Parameter)
                self.assertTrue(adapter.steering_strength.requires_grad)
                self.assertAlmostEqual(adapter.steering_strength.item(), expected, places=7)

    def test_strength_scales_deterministic_and_sampled_variational_steering(self):
        for variational in (False, True):
            with self.subTest(variational=variational):
                adapter = make_adapter(variational_modeling=variational)
                adapter.train()
                with torch.no_grad():
                    adapter.steering_strength.fill_(0.2)
                torch.manual_seed(981)
                self.forward(adapter)
                first = adapter.last_delta.detach().clone()
                with torch.no_grad():
                    adapter.steering_strength.fill_(0.4)
                torch.manual_seed(981)
                self.forward(adapter)
                torch.testing.assert_close(adapter.last_delta, 2 * first)
                self.assertTrue(torch.equal(adapter.last_delta[~self.mask.bool()], torch.zeros_like(adapter.last_delta[~self.mask.bool()])))

    def test_zero_strength_preserves_base_hidden_states(self):
        adapter = make_adapter(steering_strength_init=0.0).eval()
        with torch.no_grad():
            base = adapter.encoder(input_ids=self.ids, attention_mask=self.mask, use_cache=False).last_hidden_state
            output = self.forward(adapter)
        expected = base.masked_fill(~self.mask.bool().unsqueeze(-1), 0)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(adapter.get_delta_loss().item(), 0.0)

    def test_metrics_and_penalty_use_applied_delta_valid_tokens_and_fp32(self):
        adapter = make_adapter(steering_strength_init=0.2).half().eval()
        captured = {}
        handle = adapter.encoder.register_forward_hook(lambda module, inputs, output: captured.update(base=output.last_hidden_state.detach().float()))
        self.addCleanup(handle.remove)
        self.forward(adapter)
        valid = self.mask.bool()
        applied = adapter.last_delta.float()
        expected_penalty = applied[valid].square().mean()
        loss = adapter.get_delta_loss()
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, expected_penalty)
        metrics = adapter.get_steering_metrics()
        delta_norms = applied.detach().norm(dim=-1)[valid]
        base_norms = captured['base'].norm(dim=-1)[valid]
        ratios = delta_norms / base_norms.clamp_min(1e-8)
        expected = dict(delta_norm_mean=delta_norms.mean(), base_norm_mean=base_norms.mean(),
                        relative_norm_mean=ratios.mean(), relative_norm_p50=torch.quantile(ratios, 0.5),
                        relative_norm_p95=torch.quantile(ratios, 0.95))
        self.assertEqual(set(metrics), {'strength', *expected})
        for name, value in metrics.items():
            self.assertEqual(value.shape, torch.Size([]), name)
            self.assertEqual(value.dtype, torch.float32, name)
            self.assertFalse(value.requires_grad, name)
            self.assertIsNone(value.grad_fn, name)
            self.assertTrue(torch.isfinite(value), name)
        for name, value in expected.items():
            torch.testing.assert_close(metrics[name], value)
        strength_snapshot = metrics['strength'].clone()
        with torch.no_grad():
            adapter.steering_strength.add_(1)
        self.assertTrue(torch.equal(adapter.get_steering_metrics()['strength'], strength_snapshot))

    def test_right_padding_does_not_change_penalty_or_metrics(self):
        adapter = make_adapter().eval()
        self.forward(adapter)
        penalty = adapter.get_delta_loss().detach()
        metrics = adapter.get_steering_metrics()
        self.forward(adapter, torch.cat([self.ids, torch.zeros((2, 2), dtype=torch.long)], dim=1),
                     torch.cat([self.mask, torch.zeros((2, 2), dtype=torch.long)], dim=1))
        torch.testing.assert_close(adapter.get_delta_loss(), penalty)
        for name, value in metrics.items():
            torch.testing.assert_close(adapter.get_steering_metrics()[name], value)

    def test_all_padding_returns_zero_penalty_and_norm_metrics(self):
        adapter = make_adapter().eval()
        self.forward(adapter, mask=torch.zeros_like(self.mask))
        self.assertEqual(adapter.get_delta_loss().item(), 0.0)
        for name, value in adapter.get_steering_metrics().items():
            self.assertTrue(torch.isfinite(value))
            if name != 'strength':
                self.assertEqual(value.item(), 0.0, name)

    def test_optimizer_updates_strength_and_aggregator_but_frozen_base_stays_fixed(self):
        adapter = make_adapter()
        base = {name: value.detach().clone() for name, value in adapter.encoder.state_dict().items()}
        alpha = adapter.steering_strength.detach().clone()
        aligner = adapter.aggregator.linear_aligners[0].weight.detach().clone()
        optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=0.01)
        adapter.train()
        output = self.forward(adapter)
        (output.square().mean() + adapter.get_delta_loss()).backward()
        self.assertFalse(adapter.encoder.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in adapter.encoder.parameters()))
        self.assertGreater(adapter.steering_strength.grad.abs().item(), 0)
        self.assertGreater(adapter.aggregator.linear_aligners[0].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(alpha, adapter.steering_strength))
        self.assertFalse(torch.equal(aligner, adapter.aggregator.linear_aligners[0].weight))
        for name, value in adapter.encoder.state_dict().items():
            self.assertTrue(torch.equal(base[name], value), name)

    def checkpoint_roundtrip(self, legacy=False, missing=False, misplaced=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.pt'
            configuration = config_values(steering_strength_init=0.03)
            original = CheckpointModel(configuration).eval()
            with torch.no_grad():
                original.transformer.encoder.steering_strength.fill_(1.0 if legacy else 0.23)
            with redirect_stdout(io.StringIO()):
                SETUP.save_model(original, 'layer_adapter', configuration, str(path))
            payload = torch.load(path, weights_only=False)
            if misplaced:
                name = next(name for name in payload['learnable_params'] if name.endswith('.steering_strength'))
                payload['learnable_params']['wrong.steering_strength'] = payload['learnable_params'].pop(name)
                torch.save(payload, path)
            if legacy or missing:
                payload['learnable_params'] = {name: value for name, value in payload['learnable_params'].items() if not name.endswith('.steering_strength')}
                if legacy:
                    payload['adapter_config'].pop('steering_strength_init')
                torch.save(payload, path)

            def setup(**kwargs):
                return CheckpointModel(kwargs['adapter_config']), object()

            with mock.patch.object(SETUP, 'setup_model', side_effect=setup), redirect_stdout(io.StringIO()):
                if missing or misplaced:
                    with self.assertRaisesRegex(ValueError, 'steering_strength'):
                        SETUP.load_model(str(path))
                    return
                if legacy:
                    with self.assertWarnsRegex(UserWarning, 'legacy|Legacy'):
                        loaded, _, configuration = SETUP.load_model(str(path))
                else:
                    loaded, _, configuration = SETUP.load_model(str(path))
            loaded.eval()
            self.assertAlmostEqual(loaded.transformer.encoder.steering_strength.item(), 1.0 if legacy else 0.23, places=6)
            torch.testing.assert_close(self.forward(loaded.transformer.encoder), self.forward(original.transformer.encoder), rtol=0, atol=0)
            self.assertEqual(configuration['steering_strength_init'], 1.0 if legacy else 0.03)

    def test_checkpoint_roundtrip_preserves_learned_strength(self):
        self.checkpoint_roundtrip()

    def test_save_records_actual_initial_strength_without_mutating_config(self):
        model = CheckpointModel(config_values(steering_strength_init=0.07))
        caller_config = config_values()
        original_config = copy.deepcopy(caller_config)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / 'adapter.pt'
            SETUP.save_model(model, 'layer_adapter', caller_config, str(path))
            saved = torch.load(path, weights_only=False)
        self.assertEqual(caller_config, original_config)
        self.assertEqual(saved['adapter_config']['steering_strength_init'], 0.07)

    def test_legacy_checkpoint_keeps_full_strength_predictions(self):
        self.checkpoint_roundtrip(legacy=True)

    def test_new_checkpoint_missing_its_learned_strength_is_rejected(self):
        self.checkpoint_roundtrip(missing=True)

    def test_checkpoint_with_misplaced_strength_parameter_is_rejected(self):
        self.checkpoint_roundtrip(misplaced=True)


if str(REPO / 'tests') not in sys.path:
    sys.path.insert(0, str(REPO / 'tests'))
import test_training_resume as resume_checks


TRAIN = resume_checks.TRAIN


class MetricsEncoder(resume_checks.TinySteeringEncoder):
    def __init__(self):
        super().__init__()
        self.steering_strength = torch.nn.Parameter(torch.tensor(0.01))
        self.penalties = []

    def forward(self, x):
        self.base = x.detach()
        self.delta = self.steering_strength * super().forward(x)
        self.strength_snapshot = self.steering_strength.detach().clone()
        return self.delta

    def get_delta_loss(self):
        loss = self.delta.float().square().mean()
        self.penalties.append(loss.detach().item())
        return loss

    def get_steering_metrics(self):
        delta = self.delta.detach().float().norm(dim=-1)
        base = self.base.float().norm(dim=-1)
        ratios = delta / base.clamp_min(1e-8)
        return dict(strength=self.strength_snapshot, delta_norm_mean=delta.mean(), base_norm_mean=base.mean(),
                    relative_norm_mean=ratios.mean(), relative_norm_p50=torch.quantile(ratios, 0.5),
                    relative_norm_p95=torch.quantile(ratios, 0.95))


class MetricsModel(resume_checks.TinyStochasticModel):
    def __init__(self):
        super().__init__()
        self.transformer.encoder = MetricsEncoder()


class SteeringTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = resume_checks.TrainingResumeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def run_training(self, name, *, enabled=True, weight=0.25, variational=False, resume=None, interrupt=None):
        output = self.fixture.root / name
        output.mkdir()
        TRAIN.set_seed(73)
        model = MetricsModel()
        if resume:
            model.load_state_dict(resume.weights)
        loader = self.fixture.make_loader(self.fixture.make_dataset())
        logs = []
        real_save = TRAIN.save_training_status

        def save(*args, **kwargs):
            real_save(*args, **kwargs)
            if interrupt and kwargs['epoch'] == 0 and kwargs['batch_index'] == interrupt:
                raise resume_checks.TrainingInterrupted()

        def log(values, step=None, **kwargs):
            if 'train/total_loss' in values:
                logs.append((step, dict(values)))

        with mock.patch.object(torch, 'compile', side_effect=resume_checks.compile_identity), \
             mock.patch.object(TRAIN, 'save_model', side_effect=resume_checks.save_tiny_model), \
             mock.patch.object(TRAIN, 'save_training_status', side_effect=save), \
             mock.patch.object(TRAIN.wandb, 'log', side_effect=log), \
             mock.patch.object(TRAIN, 'tqdm', resume_checks.QuietProgress), redirect_stdout(io.StringIO()):
            try:
                TRAIN.train(model=model, train_dataloader=loader, device=torch.device('cpu'),
                            model_path=str(output / 'checkpoint'), learning_rate=0.01,
                            adapter_type='layer_adapter', adapter_config={}, variational_modeling=variational,
                            num_epochs=2, early_stopping_patience=0, progress_interval=1, val_interval=1,
                            shift_regularization=enabled, shift_regularization_weight=weight,
                            seed=73, resume_state=resume.status if resume else None)
            except resume_checks.TrainingInterrupted:
                self.assertIsNotNone(interrupt)
        status = TRAIN.load_training_status(str(output / 'checkpoint.training_status.json'))
        return SimpleNamespace(model=model, logs=logs, status=status,
                               weights=torch.load(status['checkpoint_path'], weights_only=True))

    def test_weighted_penalty_and_steering_logs_keep_lm_loss_separate(self):
        result = self.run_training('weighted')
        for index, (_, log) in enumerate(result.logs):
            self.assertAlmostEqual(log['train/lm_loss'], result.model.losses[index], places=7)
            self.assertAlmostEqual(log['train/shift_loss'], result.model.transformer.encoder.penalties[index], places=7)
            self.assertAlmostEqual(log['train/shift_penalty'], 0.25 * log['train/shift_loss'], places=7)
            self.assertAlmostEqual(log['train/total_loss'], log['train/lm_loss'] + log['train/shift_penalty'], places=6)
            self.assertEqual(log['train/shift_regularization_weight'], 0.25)
            for key in ('strength', 'delta_norm_mean', 'base_norm_mean', 'relative_norm_mean', 'relative_norm_p50', 'relative_norm_p95'):
                self.assertIsInstance(log['steering/' + key], float)

    def test_disabled_penalty_and_variational_penalty_do_not_mutate_lm_loss(self):
        for variational in (False, True):
            with self.subTest(variational=variational):
                result = self.run_training('unregularized-' + str(variational), enabled=False, variational=variational)
                for index, (_, log) in enumerate(result.logs):
                    self.assertEqual(log['train/shift_regularization_weight'], 0.0)
                    self.assertEqual(log['train/shift_penalty'], 0.0)
                    self.assertAlmostEqual(log['train/lm_loss'], result.model.losses[index], places=7)

    def test_resume_preserves_strength_diagnostics_losses_and_weights(self):
        full = self.run_training('full')
        first = self.run_training('partial', interrupt=3)
        resumed = self.run_training('resumed', resume=first)
        self.assertEqual(full.logs, first.logs + resumed.logs)
        self.fixture.assert_tree_equal(full.weights, resumed.weights, 'strength_weights')
        self.fixture.assert_tree_equal(full.status['_state']['optimizer_state_dict'], resumed.status['_state']['optimizer_state_dict'], 'optimizer')


if __name__ == '__main__':
    unittest.main()
