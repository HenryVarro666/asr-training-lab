"""Generation-based normalized WER; callable alone or from the training loop."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from .data import ManifestDataset, WhisperCollator
from .training_utils import atomic_json, normalized_wer, sha256_file


def autocast_context(device, precision):
    if precision == 'bf16':
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def evaluate_model(model, processor, loader, device, output_path, precision='bf16', max_new_tokens=440):
    was_training = model.training
    model.eval()
    predictions = []
    truncated_count = 0
    total_loss = 0.0
    total_tokens = 0
    started = time.monotonic()
    try:
        for batch in loader:
            inputs = {key: batch[key].to(device, non_blocking=True)
                      for key in ('input_features', 'attention_mask', 'labels')}
            with autocast_context(device, precision):
                loss = model(**inputs).loss
                generated = model.generate(
                    input_features=inputs['input_features'], attention_mask=inputs['attention_mask'],
                    language='english', task='transcribe', return_timestamps=False,
                    max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, use_cache=True,
                )
            tokens = inputs['labels'].ne(-100).sum().item()
            total_loss += float(loss) * tokens
            total_tokens += tokens
            # A sequence that fills the generation budget without EOS was truncated.
            for token_ids in generated:
                truncated_count += int(token_ids.shape[0] >= max_new_tokens and token_ids[-1].item() != model.config.eos_token_id)
            decoded = processor.batch_decode(generated, skip_special_tokens=True)
            for index, hypothesis in enumerate(decoded):
                predictions.append(dict(id=batch['ids'][index], reference=batch['texts'][index], hypothesis=hypothesis))
        normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)
        result = normalized_wer([p['reference'] for p in predictions], [p['hypothesis'] for p in predictions], normalizer)
        for p in predictions:
            p['reference_normalized'] = normalizer(p['reference']).strip()
            p['hypothesis_normalized'] = normalizer(p['hypothesis']).strip()
        result.update(loss=total_loss / max(total_tokens, 1), examples=len(predictions),
                      elapsed_seconds=time.monotonic() - started, generation_max_new_tokens=max_new_tokens,
                      generation_truncated_count=truncated_count,
                      generation_truncated_fraction=truncated_count / len(predictions))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction, ensure_ascii=False) + '\n')
        atomic_json(output_path.with_suffix('.metrics.json'), result)
        return result
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='HF model id or saved checkpoint directory')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--precision', choices=['bf16', 'float32'], default='bf16')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    processor = WhisperProcessor.from_pretrained(args.model, revision=args.revision, token=False, language='english', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(args.model, revision=args.revision, token=False, attn_implementation='sdpa').to(device)
    model.generation_config.forced_decoder_ids = None
    dataset = ManifestDataset(args.manifest, processor.tokenizer, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=WhisperCollator(processor, model.config.decoder_start_token_id), pin_memory=device.type == 'cuda')
    result = evaluate_model(model, processor, loader, device, args.output, args.precision)
    result.update(model=args.model, resolved_revision=getattr(model.config, '_commit_hash', None),
                  manifest_sha256=sha256_file(args.manifest), subset=dataset.report())
    atomic_json(Path(args.output).with_suffix('.metrics.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
