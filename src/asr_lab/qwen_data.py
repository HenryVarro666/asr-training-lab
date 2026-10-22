"""Qwen3-ASR manifest/prompt contracts; prefix masking adapted from Alibaba's SFT.

Copyright 2026 The Alibaba Qwen team. Copyright 2026 ASR Lab contributors.
SPDX-License-Identifier: Apache-2.0
Source: QwenLM/Qwen3-ASR finetuning/qwen3_asr_sft.py @
7c6daf77a2421100f5fb066495372c00129d39ff.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

MODEL_ID = 'Qwen/Qwen3-ASR-1.7B'
MODEL_REVISION = '7278e1e70fe206f11671096ffdd38061171dd6e5'
OFFICIAL_REVISION = '7c6daf77a2421100f5fb066495372c00129d39ff'
LANGUAGE_PREFIX = 'language English<asr_text>'


def select_records(records, limit=None, seed=42):
    records = sorted(records, key=lambda row: row['id'])
    if limit is None or limit >= len(records):
        return records
    if limit < 1:
        raise ValueError('Subset limit must be positive')
    rng = random.Random(seed)
    by_speaker = defaultdict(list)
    for row in records:
        by_speaker[str(row.get('speaker_id', row['id'].split('-')[0]))].append(row)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    for rows in by_speaker.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < limit:
        for speaker in speakers:
            if by_speaker[speaker]:
                selected.append(by_speaker[speaker].pop())
                if len(selected) == limit:
                    break
    return sorted(selected, key=lambda row: row['id'])


def read_manifest(path, limit=None, seed=42):
    records, ids = [], set()
    with open(path) as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row['id'] in ids:
                raise ValueError(f'Duplicate utterance: {row["id"]}')
            ids.add(row['id'])
            duration = float(row['duration'])
            if not math.isfinite(duration) or not 0 < duration <= 30:
                raise ValueError(f'Invalid duration, line {number}: {duration}')
            if int(row.get('sample_rate', 16000)) != 16000 or not row['text'].strip():
                raise ValueError(f'Invalid sample rate/text, line {number}')
            if not Path(row['audio_filepath']).is_file():
                raise ValueError(f'Missing audio: {row["audio_filepath"]}')
            records.append(row)
    if not records:
        raise ValueError('Manifest is empty')
    return select_records(records, limit, seed)


def load_audio(row):
    import numpy as np
    import soundfile as sf
    audio, rate = sf.read(row['audio_filepath'], dtype='float32', always_2d=False)
    if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all() or not 0 < len(audio) <= 480000:
        raise ValueError(f'Invalid waveform: {row["id"]}')
    if abs(len(audio) / rate - float(row['duration'])) > .02:
        raise ValueError(f'Duration changed: {row["id"]}')
    return audio


def prompt_text(processor, force_language=False):
    messages = [dict(role='system', content=''),
                dict(role='user', content=[dict(type='audio', audio='')])]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return text + (LANGUAGE_PREFIX if force_language else '')


def build_causal_labels(input_ids, attention_mask, prefix_ids, prefix_mask):
    """Mask prompt/audio and padding; preserve EOS even when EOS == PAD.

    Labels remain aligned with input tokens. The causal LM shifts them internally.
    """
    import torch
    labels = input_ids.clone()
    for i in range(input_ids.shape[0]):
        full_length = int(attention_mask[i].sum())
        prefix_length = int(prefix_mask[i].sum())
        if not bool(attention_mask[i, :full_length].eq(1).all()) or not bool(attention_mask[i, full_length:].eq(0).all()):
            raise ValueError('Training requires RIGHT padding; full mask is not right padded')
        if not bool(prefix_mask[i, :prefix_length].eq(1).all()) or not bool(prefix_mask[i, prefix_length:].eq(0).all()):
            raise ValueError('Training requires RIGHT padding; prefix mask is not right padded')
        if prefix_length >= full_length:
            raise ValueError('No supervised target tokens after prompt')
        if not torch.equal(input_ids[i, :prefix_length], prefix_ids[i, :prefix_length]):
            raise ValueError('Full input does not begin with exact tokenized audio/chat prefix')
        labels[i, :prefix_length] = -100
    labels.masked_fill_(attention_mask.ne(1), -100)
    return labels


class QwenManifestDataset:
    def __init__(self, manifest, limit=None, seed=42):
        self.records = read_manifest(manifest, limit, seed)
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        return self.records[index]


class QwenCollator:
    def __init__(self, processor):
        self.processor = processor
        processor.tokenizer.padding_side = 'right'
        self.prefix = prompt_text(processor)
    def __call__(self, rows):
        audios = [load_audio(row) for row in rows]
        prefixes = [self.prefix] * len(rows)
        eos = self.processor.tokenizer.eos_token
        if not eos:
            raise ValueError('Tokenizer must define EOS')
        texts = [self.prefix + LANGUAGE_PREFIX + row['text'].strip() + eos for row in rows]
        kwargs = dict(audio=audios, return_tensors='pt', padding=True, padding_side='right', truncation=False)
        full = self.processor(text=texts, **kwargs)
        prefix = self.processor(text=prefixes, **kwargs)
        import torch
        full['_audio_seconds'] = torch.tensor(sum(float(row['duration']) for row in rows), dtype=torch.float64)
        full['_examples'] = torch.tensor(len(rows), dtype=torch.float64)
        full['labels'] = build_causal_labels(full['input_ids'], full['attention_mask'], prefix['input_ids'], prefix['attention_mask'])
        return full


def local_model_path(model=MODEL_ID, revision=MODEL_REVISION):
    if Path(model).is_dir():
        return str(Path(model).resolve())
    from huggingface_hub import snapshot_download
    # Resolve first: the official wrapper does not forward revision to AutoProcessor.
    return snapshot_download(model, revision=revision, token=False,
        ignore_patterns=['*.bin', '*.onnx', '*.msgpack', '*.h5'])


def main():
    parser = argparse.ArgumentParser(description='Write a fixed speaker-balanced eval subset shared by both models')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = read_manifest(args.manifest, args.limit, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(json.dumps(dict(examples=len(rows), speakers=len({row.get('speaker_id',row['id'].split('-')[0]) for row in rows}), ids=[r['id'] for r in rows])))


if __name__ == '__main__':
    main()
