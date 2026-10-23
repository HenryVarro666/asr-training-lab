import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

_SPEC=importlib.util.spec_from_file_location('prediction_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_asr_predictions.py')
analysis=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def row(identifier,reference,hypothesis):
    return dict(id=identifier,reference=reference,hypothesis=hypothesis,
        reference_normalized=reference,hypothesis_normalized=hypothesis)


class PairedAnalysisTests(unittest.TestCase):
    def test_corpus_ratio_is_not_average_utterance_wer_and_pairing_ignores_order(self):
        baseline=[row('short','a','wrong'),row('long','one two three four five six seven eight nine','one two three four five six seven eight nine')]
        candidate=[row('long','one two three four five six seven eight nine','one two three four five six seven eight nine'),row('short','a','a')]
        result,details=analysis.analyze_records(baseline,candidate,replicates=200,seed=42)
        self.assertEqual(result['baseline']['wer'],.1)
        self.assertEqual(result['candidate']['wer'],0)
        self.assertEqual(result['comparison']['wer_delta'],-.1)
        self.assertEqual(result['utterance_changes']['improved'],1)
        self.assertEqual(result['edit_deltas']['substitutions'],-1)
        self.assertEqual(result['bootstrap']['utterance']['seed'],42)
        self.assertEqual(result,analysis.analyze_records(baseline,list(reversed(candidate)),replicates=200,seed=42)[0])

    def test_identical_predictions_have_exact_zero_interval(self):
        rows=[row('a','a b','a c'),row('b','d','d x')]
        result,_=analysis.analyze_records(rows,rows,replicates=100)
        self.assertEqual(result['bootstrap']['utterance']['wer_delta_95_ci'],[0,0])
        self.assertEqual(result['utterance_changes']['tied'],2)

    def test_speaker_bootstrap_resamples_complete_clusters_with_variable_denominators(self):
        baseline=[row('a','a','wrong'),row('b','b','wrong'),row('c','c d e f g h i j','c d e f g h i j')]
        candidate=[row('a','a','a'),row('b','b','b'),row('c','c d e f g h i j','c d e f g h i j')]
        manifest=[dict(id='a',speaker_id='s1'),dict(id='b',speaker_id='s1'),dict(id='c',speaker_id='s2')]
        result,_=analysis.analyze_records(baseline,candidate,manifest,bootstrap_unit='both',replicates=200)
        self.assertEqual(result['primary_bootstrap_unit'],'speaker_cluster')
        self.assertEqual(result['bootstrap']['speaker']['units'],2)
        self.assertEqual(result['bootstrap']['speaker']['wer_delta_95_ci'],[-1,0])
        self.assertEqual(result['baseline']['reference_words'],10)

    def test_missing_duplicate_and_mismatched_references_are_rejected(self):
        baseline=[row('a','hello','hello')]
        bad_cases=[[],[row('b','hello','hello')],[row('a','different','hello')],baseline+baseline]
        for candidate in bad_cases:
            with self.subTest(candidate=candidate),self.assertRaises(ValueError):
                analysis.analyze_records(baseline,candidate,replicates=100)
        changed=row('a','hello','hello');changed['reference_normalized']='HELLO'
        with self.assertRaisesRegex(ValueError,'reference_normalized'):
            analysis.analyze_records(baseline,[changed],replicates=100)

    def test_zero_reference_words_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'zero reference-word denominator'):
            analysis.analyze_records([row('a','','x')],[row('a','','')],replicates=100)

    def test_normalizer_and_published_denominator_mismatch_are_rejected(self):
        rows=[row('a','one two','one')]
        with self.assertRaisesRegex(ValueError,'Normalizer metadata mismatch'):
            analysis.analyze_records(rows,rows,baseline_metrics=dict(normalizer='A'),candidate_metrics=dict(normalizer='B'),replicates=100)
        with self.assertRaisesRegex(ValueError,'reference_words'):
            analysis.analyze_records(rows,rows,baseline_metrics=dict(reference_words=3),replicates=100)
        result,_=analysis.analyze_records(rows,rows,replicates=100)
        self.assertEqual(result['validation']['normalizer_metadata']['status'],'metadata_incomplete')

    def test_manifest_mapping_must_be_complete_and_reference_consistent(self):
        rows=[row('a','one two','one')]
        for manifest in ([],[dict(id='a')],[dict(id='a',speaker_id='s1',text='different')]):
            with self.subTest(manifest=manifest),self.assertRaises(ValueError):
                analysis.analyze_records(rows,rows,manifest,bootstrap_unit='speaker',replicates=100)
        with self.assertRaisesRegex(ValueError,'requires --manifest'):
            analysis.analyze_records(rows,rows,bootstrap_unit='speaker',replicates=100)

    def test_edit_breakdown_matches_substitution_deletion_and_insertion(self):
        self.assertEqual(analysis.count_edits('a b c','a x c y'),dict(hits=2,substitutions=1,deletions=0,insertions=1,reference_words=3,edits=2))
        self.assertEqual(analysis.count_edits('a b c','a c')['deletions'],1)


def complete_metrics(rows):
    counts=analysis.summarize([analysis.count_edits(row['reference_normalized'],row['hypothesis_normalized']) for row in rows])
    return dict(counts,examples=len(rows),selected_ids=[row['id'] for row in rows],
        normalizer='Whisper EnglishTextNormalizer',normalizer_revision='pinned-revision',
        normalizer_settings={'spelling_map_revision':'same-pin'},manifest_sha256='a'*64,
        evaluation_batch_size=16,generation_max_new_tokens=512,
        generation_truncated_count=0,generation_truncated_fraction=0.0)


class ComparisonValidityTests(unittest.TestCase):
    def test_complete_identical_conditions_are_valid_and_provenance_is_fingerprinted(self):
        rows=[row('a','one two','one x'),row('b','three','three')]
        metadata=complete_metrics(rows)
        manifest=[dict(id='a',speaker_id='s1'),dict(id='b',speaker_id='s2')]
        result,_=analysis.analyze_records(rows,rows,manifest,metadata,metadata,
            bootstrap_unit='speaker',replicates=100,manifest_sha256='a'*64)
        self.assertTrue(result['comparison_valid'])
        self.assertEqual(result['comparison_invalid_reasons'],[])
        self.assertEqual(result['bootstrap']['speaker']['seed'],20261022)
        self.assertEqual(len(result['provenance']['analysis_source_sha256']),64)
        self.assertEqual(len(result['provenance']['jiwer_process_source_sha256']),64)
        self.assertIn('jiwer',result['provenance']['packages'])

    def test_manifest_batch_decode_and_truncation_mismatch_are_never_valid(self):
        rows=[row('a','one two','one x')]
        metadata=complete_metrics(rows)
        changes={'manifest_sha256':'b'*64,'evaluation_batch_size':8,'generation_max_new_tokens':440,
            'generation_truncated_count':1,'normalizer_settings':{'spelling_map_revision':'different'}}
        for field,value in changes.items():
            changed=dict(metadata,**{field:value})
            if field=='generation_truncated_count':
                changed['generation_truncated_fraction']=1.0
            with self.subTest(field=field):
                result,_=analysis.analyze_records(rows,rows,baseline_metrics=metadata,candidate_metrics=changed,
                    replicates=100,manifest_sha256='a'*64)
                self.assertFalse(result['comparison_valid'])
                self.assertTrue(any(field in issue or 'truncation' in issue for issue in result['comparison_invalid_reasons']))
        result,_=analysis.analyze_records(rows,rows,baseline_metrics=metadata,candidate_metrics=metadata,
            replicates=100,manifest_sha256='b'*64)
        self.assertFalse(result['comparison_valid'])
        self.assertIn('supplied_manifest_file_hash_mismatch',result['comparison_invalid_reasons'])

    def test_missing_declared_conditions_or_actual_manifest_check_cannot_be_valid(self):
        rows=[row('a','one','one')]
        full=complete_metrics(rows)
        for field in ('normalizer','normalizer_revision','manifest_sha256','evaluation_batch_size',
                      'generation_max_new_tokens','generation_truncated_count','substitutions'):
            missing=dict(full);missing.pop(field)
            with self.subTest(field=field):
                result,_=analysis.analyze_records(rows,rows,baseline_metrics=full,candidate_metrics=missing,
                    replicates=100,manifest_sha256='a'*64)
                self.assertFalse(result['comparison_valid'])
        result,_=analysis.analyze_records(rows,rows,baseline_metrics=full,candidate_metrics=full,replicates=100)
        self.assertFalse(result['comparison_valid'])
        self.assertIn('manifest_file_hash_not_verified',result['comparison_invalid_reasons'])

    def test_same_wer_but_incorrect_stored_edit_breakdown_is_rejected(self):
        rows=[row('a','one two','one x')]
        full=complete_metrics(rows)
        wrong=dict(full,substitutions=0,deletions=1)
        with self.assertRaisesRegex(ValueError,'substitutions'):
            analysis.analyze_records(rows,rows,baseline_metrics=wrong,replicates=100)
        bad_row=dict(rows[0],substitutions=0)
        with self.assertRaisesRegex(ValueError,'stored substitutions'):
            analysis.analyze_records([bad_row],rows,replicates=100)

    def test_truncation_flag_and_fraction_conflicts_are_rejected(self):
        rows=[row('a','one','one')]
        full=complete_metrics(rows)
        with self.assertRaisesRegex(ValueError,'fraction/count'):
            analysis.analyze_records(rows,rows,baseline_metrics=dict(full,generation_truncated_fraction=.5),replicates=100)
        with self.assertRaisesRegex(ValueError,'flags disagree'):
            analysis.analyze_records([dict(rows[0],generation_truncated=True)],rows,baseline_metrics=full,replicates=100)

    def test_defaults_are_ten_thousand_replicates_and_seed_20261022(self):
        rows=[row('a','one','one'),row('b','two','two')]
        result,_=analysis.analyze_records(rows,rows)
        self.assertEqual(result['bootstrap']['utterance']['replicates'],10000)
        self.assertEqual(result['bootstrap']['utterance']['seed'],20261022)


class CommandLineTests(unittest.TestCase):
    def test_real_cli_validates_actual_manifest_hash_and_fails_closed_on_batch_mismatch(self):
        rows=[row('a','one two','one x'),row('b','three','three')]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest=root/'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(dict(id=item['id'],speaker_id=item['id'],text=item['reference']))+'\n' for item in rows))
            metadata=dict(complete_metrics(rows),manifest_sha256=analysis.sha256(manifest))
            for name in ('baseline','candidate'):
                (root/f'{name}.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in rows))
                (root/f'{name}.metrics.json').write_text(json.dumps(metadata))
            output=root/'analysis.json'
            command=[sys.executable,str(Path(analysis.__file__)),
                '--baseline',str(root/'baseline.jsonl'),'--candidate',str(root/'candidate.jsonl'),
                '--manifest',str(manifest),'--bootstrap-unit','speaker','--output',str(output),'--require-valid']
            completed=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            result=json.loads(output.read_text())
            self.assertTrue(result['comparison_valid'])
            self.assertEqual(result['bootstrap']['speaker']['seed'],20261022)
            self.assertEqual(result['bootstrap']['speaker']['replicates'],10000)
            self.assertEqual(result['inputs']['manifest']['sha256'],metadata['manifest_sha256'])
            (root/'candidate.metrics.json').write_text(json.dumps(dict(metadata,evaluation_batch_size=8)))
            rejected=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(rejected.returncode,2,rejected.stderr)
            self.assertFalse(json.loads(output.read_text())['comparison_valid'])
            self.assertIn('evaluation_batch_size_mismatch',json.loads(output.read_text())['comparison_invalid_reasons'])


if __name__=='__main__':
    unittest.main()
