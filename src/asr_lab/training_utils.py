"""Contracts shared by training, evaluation, and checkpoint recovery."""
import hashlib
import json
import os
from pathlib import Path


def mask_labels(input_ids, attention_mask, decoder_start_token_id):
    # EOS and PAD share an id in Whisper: the attention mask is the authority.
    labels = input_ids.masked_fill(attention_mask.ne(1), -100)
    if labels.shape[1] and labels[:, 0].eq(decoder_start_token_id).all().item():
        labels = labels[:, 1:]
    return labels


RESUME_KEYS = ('world_size', 'batch_size', 'gradient_accumulation_steps', 'seed',
               'train_manifest_sha256', 'eval_manifest_sha256', 'model_id', 'revision',
               'learning_rate', 'weight_decay', 'warmup_steps', 'total_steps',
               'limit_train', 'limit_eval', 'max_label_length', 'max_audio_seconds', 'precision',
               'attention', 'gradient_checkpointing', 'max_wall_seconds', 'fused_adamw')


def validate_resume(previous, current):
    for key in RESUME_KEYS:
        if previous.get(key) != current.get(key):
            raise ValueError(f'Resume invariant changed: {key}: {previous.get(key)!r} -> {current.get(key)!r}')


def update_groups(num_batches, accumulation, start_batch=0):
    if accumulation < 1 or start_batch < 0 or start_batch > num_batches:
        raise ValueError('Invalid accumulation or resume cursor')
    if start_batch != num_batches and start_batch % accumulation:
        raise ValueError('Resume cursor must be at an optimizer-update boundary')
    for start in range(start_batch, num_batches, accumulation):
        yield start, min(accumulation, num_batches - start)


def normalized_wer(references, hypotheses, normalizer):
    from jiwer import process_words
    if len(references) != len(hypotheses) or not references:
        raise ValueError('WER requires matching, nonempty reference and hypothesis lists')
    refs = [normalizer(s).strip() for s in references]
    hyps = [normalizer(s).strip() for s in hypotheses]
    if not sum(len(s.split()) for s in refs):
        raise ValueError('WER reference has no normalized words')
    result = process_words(refs, hyps)
    return dict(wer=result.wer, reference_words=result.hits + result.substitutions + result.deletions,
                substitutions=result.substitutions, deletions=result.deletions,
                insertions=result.insertions, hits=result.hits)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
