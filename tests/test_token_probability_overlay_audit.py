"""Small JSON-only contracts for the separate corrected-overlay verifier."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'reports/token-probability/validation/audit_recomputed_evidence.py'
spec = importlib.util.spec_from_file_location('overlay_audit', PATH)
overlay_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay_audit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


class OverlayAuditContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run, self.overlay = self.root/'original', self.root/'overlay'
        self.reference = self.root/'reference.json'
        reference = save(self.reference, {'frozen': True})
        save(self.run/'campaign.json', dict(status='complete', source_sha256={
            '/remote/repo/src/asr_lab/token_probability.py': overlay_audit.ORIGINAL_WORKER_SHA256}))
        self.manifest = dict(schema_version=1, original_run=str(self.run),
            reference_sha256=reference['sha256'], original_source_sha256={
                'src/asr_lab/token_probability.py': overlay_audit.ORIGINAL_WORKER_SHA256},
            corrected_artifacts={})
        for stage in overlay_audit.STAGES:
            artifacts = {}
            for name in sorted(overlay_audit.expected_keys()):
                if not name.startswith(stage+'/'):
                    continue
                old = save(self.run/name, {'value': 'immutable'})
                new = save(self.overlay/name, {'value': 'corrected'})
                artifacts[name.removeprefix(stage+'/worker/')] = old
                self.manifest['corrected_artifacts'][name] = dict(
                    new, path=name, original_sha256=old['sha256'], original_bytes=old['bytes'])
            save(self.run/stage/'worker/result.json', dict(status='complete', stage=stage, artifacts=artifacts))
        self.write_manifest()

    def write_manifest(self):
        save(self.overlay/'manifest.json', self.manifest)

    def validate(self):
        return overlay_audit.validate_manifest(self.overlay, self.run, self.reference)

    def evidence(self):
        original = overlay_audit.import_original()
        redirects = self.validate()[-1]
        return overlay_audit.overlay_evidence_class(original, redirects)(None, original.Budget(10))

    def test_original_arithmetic_source_and_tolerances_remain_pinned(self):
        original = overlay_audit.import_original()
        self.assertEqual(original.TOLERANCES['abs_probability'], 1e-6)
        self.assertEqual(original.TOLERANCES['abs_logprob_entropy_margin'], 5e-5)
        self.assertEqual(original.TOLERANCES['abs_difference_of_two_logprobs'], 1e-4)
        self.assertEqual(len(self.validate()[-1]), 15)

    def test_original_is_hashed_loaded_before_corrected_selected_and_reads_once(self):
        evidence = self.evidence()
        name = sorted(self.manifest['corrected_artifacts'])[0]
        record = self.manifest['corrected_artifacts'][name]
        for _ in range(2):
            self.assertEqual(evidence.load(self.run/name, record['original_sha256'], record['original_bytes']),
                             {'value': 'corrected'})
        self.assertEqual(len(evidence.disk_reads), 2)
        self.assertTrue(all(n == 1 for n in evidence.disk_reads.values()))
        self.assertEqual(evidence.objects[(self.run/name).resolve()], {'value': 'immutable'})

    def test_changed_original_rejected_even_when_corrected_is_valid(self):
        evidence = self.evidence()
        name = sorted(self.manifest['corrected_artifacts'])[0]
        save(self.run/name, {'value': 'tampered'})
        with self.assertRaisesRegex(ValueError, 'Actual original derived bytes'):
            evidence.load(self.run/name)

    def test_changed_overlay_rejected(self):
        evidence = self.evidence()
        name = sorted(self.manifest['corrected_artifacts'])[0]
        save(self.overlay/name, {'value': 'tampered'})
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            evidence.load(self.run/name)

    def test_original_link_reference_and_exact_set_are_required(self):
        name = sorted(self.manifest['corrected_artifacts'])[0]
        self.manifest['corrected_artifacts'][name]['original_sha256'] = '0'*64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'original identity'):
            self.validate()
        self.setUp()
        self.manifest['reference_sha256'] = '0'*64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'reference SHA'):
            self.validate()
        self.setUp()
        self.manifest['corrected_artifacts'].pop(name)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'exactly12'):
            self.validate()

    def test_overlay_path_escape_or_duplicate_rejected(self):
        names = sorted(self.manifest['corrected_artifacts'])
        self.manifest['corrected_artifacts'][names[0]]['path'] = '../outside.json'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'relative and contained'):
            self.validate()
        self.setUp()
        self.manifest['corrected_artifacts'][names[1]]['path'] = names[0]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.validate()

    def test_auxiliary_hashes_are_mandatory_and_verified(self):
        self.manifest['auxiliary_artifacts'] = {}
        for name in ('comparisons.json', 'change-summary.json'):
            self.manifest['auxiliary_artifacts'][name] = dict(save(self.overlay/name, {}), path=name)
        self.assertEqual(set(overlay_audit.checked_auxiliary(self.overlay, self.manifest)),
                         {'comparisons.json', 'change-summary.json'})
        save(self.overlay/'change-summary.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'Auxiliary artifact identity'):
            overlay_audit.checked_auxiliary(self.overlay, self.manifest)

    def test_failed_finalization_leaves_only_pending(self):
        output = self.root/'accepted.json'
        pending = self.root/'accepted.pending.json'
        def expired():
            raise TimeoutError('deadline')
        with self.assertRaises(TimeoutError):
            overlay_audit.atomic_accept(output, pending, {'status': 'passed'}, 'pending', expired)
        self.assertTrue(pending.exists())
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.md').exists())

    def test_remaining_alarm_uses_absolute_deadline(self):
        with mock.patch.object(overlay_audit.time, 'monotonic', return_value=15), \
             mock.patch.object(overlay_audit.signal, 'signal'), \
             mock.patch.object(overlay_audit.signal, 'setitimer') as timer:
            overlay_audit.arm_remaining(20)
            timer.assert_called_once_with(overlay_audit.signal.ITIMER_REAL, 5)
        with mock.patch.object(overlay_audit.time, 'monotonic', return_value=21):
            with self.assertRaisesRegex(ValueError, 'deadline exhausted'):
                overlay_audit.arm_remaining(20)

    def test_progress_wrapper_preserves_function_result_and_counts_returned_rows(self):
        result = object()
        function = mock.Mock(return_value=result)
        original = SimpleNamespace(full_vocab_statistics=function)
        progress = self.root/'progress'
        with mock.patch.dict(overlay_audit.os.environ, {'CPU_EVIDENCE_STAGE_DIR':str(progress)}), \
             mock.patch.object(overlay_audit, 'arm_remaining'):
            overlay_audit.install_progress(original, 600)
            self.assertIs(original.full_vocab_statistics('matrix', [1,2,3]), result)
        function.assert_called_once_with('matrix', [1,2,3])
        saved=json.loads((progress/'progress.json').read_text())
        self.assertEqual((saved['phase'],saved['completed_rows'],saved['completed_matrices']),
                         ('matrix_return',3,1))

    def test_cross_model_summary_is_independently_reconstructed(self):
        original = overlay_audit.import_original()
        reference = dict(trajectories={s: dict(token_ids=[1]) for s in overlay_audit.STAGES})
        row = dict(j=0, prefix={'length': 137, 'sha256': 'toy'}, target_id=1,
                   target=dict(id=1, prob=.5, logprob=-.7, rank=1),
                   eos=dict(total_prob=.1, total_logprob=-2.3),
                   candidates=[dict(id=1, rank=1, logprob=-.7)])
        models = {s: dict(target_logprob=-.7, target_prob=.5, target_rank=1,
                  target_logprob_delta_vs_base=0., eos_total_prob=.1, eos_total_logprob=-2.3,
                  eos_total_logprob_delta_vs_base=0.,
                  candidate_deltas=[dict(id=1, rank=1, base_rank=1, logprob_delta=0.)])
                  for s in overlay_audit.STAGES}
        comparisons = dict(free_trajectory_forks=original.independent_forks({s:[1] for s in overlay_audit.STAGES}),
                           fixed_trajectory_comparisons={})
        for trajectory in overlay_audit.STAGES:
            for stage in overlay_audit.STAGES:
                name=f'{stage}/worker/teacher/{trajectory}/derived.json'
                identity=save(self.overlay/name,dict(rows=[row],total_target_nll=.7,mean_target_nll=.7))
                self.manifest['corrected_artifacts'][name].update(identity)
            comparisons['fixed_trajectory_comparisons'][trajectory] = dict(
                rows=[dict(j=0,prefix=row['prefix'],target_id=1,models=json.loads(json.dumps(models)))],
                per_model_nll={s:dict(total=.7,mean=.7) for s in overlay_audit.STAGES})
        overlay_audit.validate_comparisons(original,self.manifest,self.overlay,reference,comparisons,lambda:None)
        comparisons['fixed_trajectory_comparisons']['base']['rows'][0]['models']['step100']['target_logprob_delta_vs_base']=.001
        with self.assertRaisesRegex(ValueError,'target_logprob_delta_vs_base'):
            overlay_audit.validate_comparisons(original,self.manifest,self.overlay,reference,comparisons,lambda:None)


if __name__ == '__main__':
    unittest.main()
