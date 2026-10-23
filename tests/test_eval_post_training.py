#!/usr/bin/env python3
"""Offline regression tests for the real eval.py cache and result wiring."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get(
    'LLM_ADAPTER_EVAL_SOURCE',
    str(REPOSITORY_ROOT / 'experiments' / 'legal_adaptation' / 'scripts' / 'eval.py'),
))
spec = importlib.util.spec_from_file_location('llm_adapter_eval_cache_under_test', SOURCE)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class PostTrainingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='llm-adapter-post-training-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / 'post-training'
        self.model_a = self.root / 'adapter-a.pt'
        self.model_b = self.root / 'adapter-b.pt'
        self.model_a.write_bytes(b'source adapter A')
        self.model_b.write_bytes(b'source adapter B')
        self.training_path = self.root / 'training.yaml'
        self.data_path = self.root / 'data.yaml'
        self.training = {'seed': 42, 'num_epochs': 1, 'learning_rate': 0.001}
        self.data = {'task': 'ledgar', 'train_split': 'train', 'validation_split': 'validation'}
        self.training_path.write_text(json.dumps(self.training))
        self.data_path.write_text(json.dumps(self.data))
        self.cfg = {'output_dir': str(self.cache), 'training_config': str(self.training_path),
                    'data_config': str(self.data_path), 'force_retrain': False}
        self.calls = []
        self.raise_after_write = False
        self.return_missing_checkpoint = False
        self.mutate_source_during_training = False
        self.return_outside_attempt = False
        self.return_name = 'checkpoint-best'
        module = ModuleType('post_train_task')
        module.run_post_training = self.fake_train
        self.module_patch = patch.dict(sys.modules, {'post_train_task': module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def fake_train(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        out = Path(kwargs['output_dir'])
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = out / self.return_name
        # Deliberately emulate a stale inner trainer short-circuit unless the
        # caller forces a fresh run after deciding that its manifest is invalid.
        if checkpoint.exists() and not kwargs['training_cfg'].get('force_retrain'):
            return str(checkpoint)
        if not self.return_missing_checkpoint:
            checkpoint.write_text(json.dumps({'training_call': len(self.calls), 'source': kwargs['model_path']}))
        if self.mutate_source_during_training:
            Path(kwargs['model_path']).write_bytes(b'source changed during post-training')
        if self.return_outside_attempt:
            return str(self.model_b)
        if self.raise_after_write:
            raise RuntimeError('injected post-training failure after checkpoint write')
        return str(checkpoint)

    def invoke(self, model=None, cfg=None, seed=None):
        kwargs = {'model_name_or_path': str(model or self.model_a), 'device': 'cpu',
                  'post_training_cfg': copy.deepcopy(self.cfg if cfg is None else cfg)}
        if seed is not None:
            kwargs['seed'] = seed
        checkpoint = evaluation.post_train_for_eval(**kwargs)
        self.last_checkpoint = checkpoint
        return checkpoint

    def manifest_for(self, checkpoint):
        for parent in Path(checkpoint).parents:
            candidate = parent / 'manifest.json'
            if candidate.is_file():
                return candidate
            if parent == self.root:
                break
        self.fail(f'no cache manifest found above checkpoint {checkpoint}')

    def assert_latest_checkpoint(self):
        payload = json.loads(Path(self.last_checkpoint).read_text())
        self.assertEqual(payload['training_call'], len(self.calls),
                         'retraining returned a stale checkpoint')

    def test_identical_source_and_configs_reuse_verified_checkpoint(self):
        first = self.invoke()
        second = self.invoke()
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.manifest_for(first).is_file())

    def test_distinct_source_checkpoints_isolate_cache(self):
        first = self.invoke()
        second = self.invoke(self.model_b)
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.calls), 2)

    def test_same_source_path_with_changed_content_invalidates_cache(self):
        first = self.invoke()
        self.model_a.write_bytes(b'updated adapter A at unchanged path')
        second = self.invoke()
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.calls), 2)

    def test_training_config_change_isolates_cache(self):
        first = self.invoke()
        self.training['learning_rate'] = 0.002
        self.training_path.write_text(json.dumps(self.training))
        second = self.invoke()
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.calls), 2)

    def test_data_config_change_isolates_cache(self):
        first = self.invoke()
        self.data['max_train_examples'] = 123
        self.data_path.write_text(json.dumps(self.data))
        second = self.invoke()
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.calls), 2)

    def test_config_formatting_and_key_order_do_not_invalidate_cache(self):
        first = self.invoke()
        self.training_path.write_text(json.dumps(dict(reversed(list(self.training.items()))), indent=4))
        self.data_path.write_text(json.dumps(dict(reversed(list(self.data.items()))), indent=2))
        self.assertEqual(self.invoke(), first)
        self.assertEqual(len(self.calls), 1)

    def test_local_model_directory_content_changes_invalidate_cache(self):
        directory = self.root / 'hf-model-directory'
        directory.mkdir()
        (directory / 'config.json').write_text('{"model_type":"gpt2"}')
        weights = directory / 'model.safetensors'
        weights.write_bytes(b'original model weights')
        first = self.invoke(directory)
        self.assertEqual(self.invoke(directory), first)
        weights.write_bytes(b'updated model weights')
        second = self.invoke(directory)
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.calls), 2)

    def test_output_inside_source_model_directory_is_rejected_before_training(self):
        directory = self.root / 'local-model-directory'
        directory.mkdir()
        (directory / 'config.json').write_text('{"model_type":"gpt2"}')
        for output_dir in (directory, directory / 'post-training'):
            with self.subTest(output_dir=output_dir):
                with self.assertRaises(ValueError):
                    self.invoke(directory, cfg={**self.cfg, 'output_dir': str(output_dir)})
        self.assertEqual(self.calls, [])
        self.assertEqual([path.name for path in directory.iterdir()], ['config.json'])

    def test_effective_data_override_changes_cache_and_reaches_trainer(self):
        first = self.invoke()
        cfg = {**self.cfg, 'max_label_length': 11, 'truncation_side': 'left'}
        second = self.invoke(cfg=cfg)
        self.assertNotEqual(first, second)
        self.assertEqual(self.calls[-1]['data_cfg']['max_label_length'], 11)
        self.assertEqual(self.calls[-1]['data_cfg']['truncation_side'], 'left')

    def test_effective_seed_changes_isolate_cache_and_reach_trainer(self):
        first = self.invoke(seed=17)
        second = self.invoke(seed=29)
        self.assertNotEqual(first, second)
        self.assertEqual([call['training_cfg']['seed'] for call in self.calls], [17, 29])
        self.assertEqual(self.invoke(seed=29), second)
        self.assertEqual(len(self.calls), 2)

    def test_distinct_hf_identifiers_isolate_without_downloading(self):
        first = self.invoke('organization/model-a')
        second = self.invoke('organization/model-b')
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.calls), 2)

    def test_legacy_unverified_root_checkpoint_is_never_reused(self):
        self.cache.mkdir()
        legacy = self.cache / 'checkpoint-best'
        legacy.write_bytes(b'legacy checkpoint from another experiment')
        actual = self.invoke()
        self.assertNotEqual(Path(actual), legacy)
        self.assertEqual(len(self.calls), 1)

    def test_missing_manifest_retrains_and_bypasses_inner_skip(self):
        checkpoint = self.invoke()
        self.manifest_for(checkpoint).unlink()
        replacement = self.invoke()
        self.assertEqual(len(self.calls), 2)
        self.assert_latest_checkpoint()
        self.assertNotEqual(checkpoint, replacement)
        self.assertTrue(Path(replacement).is_file())

    def test_malformed_manifest_retrains_and_bypasses_inner_skip(self):
        checkpoint = self.invoke()
        self.manifest_for(checkpoint).write_text('{malformed json')
        self.invoke()
        self.assertEqual(len(self.calls), 2)
        self.assert_latest_checkpoint()

    def test_non_mapping_manifest_retrains_safely(self):
        checkpoint = self.invoke()
        self.manifest_for(checkpoint).write_text('[]')
        self.invoke()
        self.assertEqual(len(self.calls), 2)
        self.assert_latest_checkpoint()

    def test_stale_manifest_retrains_safely(self):
        checkpoint = self.invoke()
        self.manifest_for(checkpoint).write_text(json.dumps({'schema_version': -1, 'fingerprint': 'stale'}))
        self.invoke()
        self.assertEqual(len(self.calls), 2)
        self.assert_latest_checkpoint()

    def test_modified_cached_checkpoint_retrains_safely(self):
        checkpoint = self.invoke()
        Path(checkpoint).write_bytes(b'tampered checkpoint')
        replacement = self.invoke()
        self.assertEqual(len(self.calls), 2)
        self.assert_latest_checkpoint()
        self.assertNotEqual(checkpoint, replacement)
        self.assertNotEqual(Path(replacement).read_bytes(), b'tampered checkpoint')

    def test_missing_cached_checkpoint_retrains_safely(self):
        checkpoint = self.invoke()
        Path(checkpoint).unlink()
        replacement = self.invoke()
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(Path(replacement).is_file())

    def test_force_retrain_refreshes_even_verified_cache(self):
        checkpoint = self.invoke()
        previous = Path(checkpoint).read_bytes()
        replacement = self.invoke(cfg={**self.cfg, 'force_retrain': True})
        self.assertEqual(len(self.calls), 2)
        self.assert_latest_checkpoint()
        self.assertNotEqual(checkpoint, replacement)
        self.assertNotEqual(Path(replacement).read_bytes(), previous)
        self.assertEqual(self.invoke(), replacement)
        self.assertEqual(len(self.calls), 2)

    def test_failed_forced_retrain_cannot_leave_valid_old_manifest(self):
        checkpoint = self.invoke()
        manifest = self.manifest_for(checkpoint)
        self.raise_after_write = True
        with self.assertRaisesRegex(RuntimeError, 'injected post-training failure'):
            self.invoke(cfg={**self.cfg, 'force_retrain': True})
        self.assertFalse(manifest.exists(), 'old valid manifest survived failed forced retraining')
        self.raise_after_write = False
        self.invoke()
        self.assertEqual(len(self.calls), 3)
        self.assert_latest_checkpoint()

    def test_failed_initial_training_checkpoint_is_not_trusted_on_retry(self):
        self.raise_after_write = True
        with self.assertRaisesRegex(RuntimeError, 'injected post-training failure'):
            self.invoke()
        self.assertEqual(list(self.cache.rglob('*.json')), [])
        self.raise_after_write = False
        self.invoke()
        self.assertEqual(len(self.calls), 2)
        self.assert_latest_checkpoint()

    def test_nonexistent_returned_checkpoint_cannot_create_valid_manifest(self):
        self.return_missing_checkpoint = True
        with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
            self.invoke()
        self.assertEqual(list(self.cache.rglob('*.json')), [])

    def test_source_changed_during_training_cannot_publish_valid_manifest(self):
        self.mutate_source_during_training = True
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke()
        self.assertEqual(list(self.cache.rglob('manifest.json')), [])

    def test_returned_checkpoint_outside_fresh_attempt_is_rejected(self):
        self.return_outside_attempt = True
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke()
        self.assertEqual(list(self.cache.rglob('manifest.json')), [])

    def test_returned_trace_checkpoint_is_reused_as_actual_checkpoint(self):
        self.return_name = 'checkpoint-trace'
        actual = self.invoke()
        self.assertEqual(Path(actual).name, 'checkpoint-trace')
        self.assertEqual(self.invoke(), actual)
        self.assertEqual(len(self.calls), 1)

    def test_fresh_trace_retraining_cannot_select_stale_best_checkpoint(self):
        old_best = self.invoke()
        self.return_name = 'checkpoint-trace'
        new_trace = self.invoke(cfg={**self.cfg, 'force_retrain': True})
        self.assertEqual(Path(old_best).name, 'checkpoint-best')
        self.assertEqual(Path(new_trace).name, 'checkpoint-trace')
        self.assertNotEqual(Path(old_best).parent, Path(new_trace).parent)
        self.assertEqual(self.invoke(), new_trace)
        self.assertEqual(len(self.calls), 2)

    def run_main_case(self, evaluator_fails=False):
        out = self.root / 'eval-output'
        cfg = {'model_name_or_path': str(self.model_a), 'device': 'cpu', 'seed': 97, 'output_dir': str(out),
               'evaluations': {'ledgar': {'enabled': True, 'post_training': {**self.cfg, 'enabled': True, 'output_dir': None}}},
               'output': {'save_summary_json': True, 'save_summary_csv': False}}
        config_path = self.root / 'eval.yaml'
        config_path.write_text(json.dumps(cfg))
        args = SimpleNamespace(config=str(config_path), output_dir=str(out))
        evaluator = ModuleType('cache_test_evaluator')
        received = []

        def evaluate(model_name_or_path, device, **kwargs):
            received.append(model_name_or_path)
            if evaluator_fails:
                raise RuntimeError('injected evaluator failure')
            return {'accuracy': 0.5, '_raw': {'samples': 2}}

        evaluator.evaluate = evaluate
        utils = ModuleType('evaluations.utils')
        utils.select_device = lambda device: 'cpu'

        def safe_save_json(data, output_file):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text(json.dumps(data))

        utils.safe_save_json = safe_save_json
        package = ModuleType('evaluations')
        package.__path__ = []
        with patch.dict(sys.modules, {'cache_test_evaluator': evaluator, 'evaluations': package, 'evaluations.utils': utils}), \
             patch.object(evaluation, 'EVALUATORS', [('ledgar', 'cache_test_evaluator')]), \
             patch.object(evaluation, 'parse_args', return_value=args), \
             patch.object(evaluation, 'apply_cli_overrides', side_effect=lambda supplied, arguments: supplied), \
             contextlib.redirect_stderr(io.StringIO()):
            evaluation.main()
        result = json.loads((out / 'results.json').read_text())
        self.assertEqual(self.calls[-1]['training_cfg']['seed'], 97)
        self.assertEqual(result['model_name_or_path'], str(self.model_a))
        self.assertEqual(len(received), 1)
        self.assertNotEqual(received[0], str(self.model_a))
        self.assertEqual(result['results']['ledgar']['evaluated_model_path'], received[0])
        Path(received[0]).resolve().relative_to(out.resolve())
        return result

    def test_global_seed_and_actual_checkpoint_metadata_survive_json_roundtrip(self):
        result = self.run_main_case()
        self.assertEqual(result['results']['ledgar']['status'], 'ok')
        self.assertEqual(result['results']['ledgar']['summary'], {'accuracy': 0.5})

    def test_actual_checkpoint_metadata_is_recorded_on_evaluator_failure(self):
        result = self.run_main_case(evaluator_fails=True)
        self.assertEqual(result['results']['ledgar']['status'], 'failed')
        self.assertIn('injected evaluator failure', result['results']['ledgar']['error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
