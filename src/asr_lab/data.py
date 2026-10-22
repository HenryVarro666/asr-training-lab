"""Manifest validation, lazy FLAC loading, and deterministic feature preparation."""
import json
from collections import Counter
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .training_utils import mask_labels


class ManifestDataset(Dataset):
    def __init__(self, manifest, tokenizer, max_audio_seconds=30.0, max_label_length=448, limit=None, subset_seed=42):
        self.records = []
        self.rejected = []
        self.manifest = str(manifest)
        seen = set()
        with open(manifest) as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record['id'] in seen:
                    raise ValueError(f'Duplicate utterance id: {record["id"]}')
                seen.add(record['id'])
                reason = None
                duration = float(record['duration'])
                if not np.isfinite(duration) or duration <= 0 or duration > max_audio_seconds:
                    reason = 'duration_out_of_range'
                elif int(record['sample_rate']) != 16000:
                    reason = 'sample_rate_not_16000'
                elif not record['text'].strip():
                    reason = 'empty_transcript'
                labels = tokenizer(record['text']).input_ids if reason is None else []
                # Decoder start is stripped by collator; the remaining target must fit.
                target_length = len(labels) - int(bool(labels) and labels[0] == tokenizer.convert_tokens_to_ids('<|startoftranscript|>'))
                if reason is None and target_length > max_label_length:
                    reason = 'labels_too_long'
                if reason:
                    self.rejected.append(dict(id=record['id'], line=line_number, reason=reason))
                else:
                    self.records.append(dict(record, labels=labels))
        # Audit every row before selecting a smoke/evaluation subset.
        if limit is not None:
            if int(limit) < 1:
                raise ValueError('Subset limit must be positive')
            self.records.sort(key=lambda record: record['id'])
            indices = sorted(random.Random(subset_seed).sample(range(len(self.records)), min(int(limit), len(self.records))))
            self.records = [self.records[index] for index in indices]
        if not self.records:
            raise ValueError(f'No usable utterances: {manifest}')

    def report(self):
        return dict(manifest=self.manifest, selected=len(self.records),
                    selected_hours=sum(float(r['duration']) for r in self.records) / 3600,
                    rejected_count=len(self.rejected),
                    rejected_by_reason=dict(Counter(r['reason'] for r in self.rejected)),
                    rejected=self.rejected,
                    selected_ids=[r['id'] for r in self.records])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        audio, sample_rate = sf.read(record['audio_filepath'], dtype='float32', always_2d=False)
        if sample_rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError(f'Invalid waveform at training read: {record["audio_filepath"]}')
        if not 0 < len(audio) <= 16000 * 30:
            raise ValueError(f'Waveform outside (0,30] seconds: {record["id"]}')
        if abs(len(audio) / sample_rate - float(record['duration'])) > 0.02:
            raise ValueError(f'Manifest duration mismatch: {record["id"]}')
        return dict(record, audio=audio)


class WhisperCollator:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, records):
        features = self.processor.feature_extractor(
            [r['audio'] for r in records], sampling_rate=16000,
            padding='max_length', max_length=480000, truncation=False,
            return_attention_mask=True, return_tensors='pt',
        )
        padded = self.processor.tokenizer.pad(
            [{'input_ids': r['labels']} for r in records], return_tensors='pt',
            return_attention_mask=True,
        )
        return dict(input_features=features.input_features,
                    attention_mask=features.attention_mask,
                    labels=mask_labels(padded.input_ids, padded.attention_mask, self.decoder_start_token_id),
                    ids=[r['id'] for r in records], texts=[r['text'] for r in records],
                    durations=torch.tensor([r['duration'] for r in records], dtype=torch.float64))
