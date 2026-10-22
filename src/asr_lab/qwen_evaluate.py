"""Fixed-subset generation WER for official Qwen3-ASR, independent of SFT loss."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import WhisperTokenizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from .qwen_data import MODEL_REVISION, QwenManifestDataset, load_audio, local_model_path, prompt_text
from .training_utils import atomic_json, normalized_wer, sha256_file

WHISPER_REVISION = '973afd24965f72e36ca33b3055d56a652f456b4d'


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', default=MODEL_REVISION)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--all', action='store_true', help='Evaluate all accepted manifest rows')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--normalizer-model', default='openai/whisper-small')
    parser.add_argument('--normalizer-revision', default=WHISPER_REVISION)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error('Batch size and max-new-tokens must be positive')
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import parse_asr_output
    device = torch.device('cuda:0')
    model_path = local_model_path(args.model, args.revision)
    wrapper = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.bfloat16,
        device_map=None, attn_implementation='sdpa', token=False)
    model, processor = wrapper.model.to(device), wrapper.processor
    model.eval()
    processor.tokenizer.padding_side = 'left'
    tokenizer = WhisperTokenizer.from_pretrained(args.normalizer_model,
        revision=args.normalizer_revision, token=False)
    normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)
    dataset = QwenManifestDataset(args.manifest, None if args.all else args.limit, args.seed)
    predictions, truncated = [], 0
    started = time.monotonic()
    for start in range(0, len(dataset), args.batch_size):
        rows = dataset.records[start:start+args.batch_size]
        inputs = processor(text=[prompt_text(processor, force_language=True)]*len(rows),
            audio=[load_audio(row) for row in rows], return_tensors='pt', padding=True,
            padding_side='left', truncation=False)
        inputs = {key: value.to(device, dtype=torch.bfloat16) if value.is_floating_point()
                  else value.to(device) for key,value in inputs.items()}
        result = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
            do_sample=False, num_beams=1, use_cache=True)
        continuation = result.sequences[:, inputs['input_ids'].shape[1]:]
        raw = processor.batch_decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for row, text, tokens in zip(rows, raw, continuation):
            _, hypothesis = parse_asr_output(text, user_language='English')
            eos = {151645,151643}
            exhausted = len(tokens) >= args.max_new_tokens and not any(int(token) in eos for token in tokens)
            truncated += int(exhausted)
            predictions.append(dict(id=row['id'],reference=row['text'],hypothesis=hypothesis,
                reference_normalized=normalizer(row['text']).strip(),hypothesis_normalized=normalizer(hypothesis).strip(),
                generation_truncated=exhausted))
        print(json.dumps(dict(phase='generation',completed=len(predictions),total=len(dataset))),flush=True)
    metrics = normalized_wer([row['reference'] for row in predictions],[row['hypothesis'] for row in predictions],normalizer)
    metrics.update(model=args.model, model_revision=args.revision if not Path(args.model).is_dir() else None,
        examples=len(predictions), evaluation_batch_size=args.batch_size, elapsed_seconds=time.monotonic()-started, manifest_sha256=sha256_file(args.manifest),
        selected_ids=[r['id'] for r in predictions], seed=args.seed,selection='speaker-balanced deterministic round-robin',
        normalizer='Whisper EnglishTextNormalizer',normalizer_revision=args.normalizer_revision,
        generation_max_new_tokens=args.max_new_tokens,generation_truncated_count=truncated,
        generation_truncated_fraction=truncated/len(predictions))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in predictions))
    atomic_json(args.output.with_suffix('.metrics.json'),metrics)
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()
