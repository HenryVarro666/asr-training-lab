"""Zero-optimizer-update save/reload control workers; preparation and eval MUST be separate processes."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import torch
from transformers import AutoModel, AutoProcessor, LogitsProcessor, LogitsProcessorList, Trainer

from .qwen_data import MODEL_ID, MODEL_REVISION, QwenManifestDataset, load_audio, local_model_path, prompt_text
from .training_utils import atomic_json, normalized_wer, sha256_file

REPO=Path(__file__).resolve().parents[2]
FROZEN_NAMES={'qwen_train.py','qwen_data.py','qwen_evaluate.py','benchmark.py'}
WHISPER_REVISION='973afd24965f72e36ca33b3055d56a652f456b4d'
EOS_IDS=[151645,151643]
file_hash=sha256_file


def freeze_receipt(receipt,manifest,repo=REPO):
    receipt_path=Path(receipt)
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('model_id')!=MODEL_ID or receipt.get('model_revision')!=MODEL_REVISION:
        raise ValueError('Frozen model revision/id differs')
    sources=receipt.get('source_sha256',{})
    if len(sources)!=4 or {Path(key).name for key in sources}!=FROZEN_NAMES:
        raise ValueError('Exactly the four original frozen source hashes are required')
    mapped={}
    for previous,expected in sources.items():
        target=Path(repo)/'src/asr_lab'/Path(previous).name
        if file_hash(target)!=expected:
            raise ValueError(f'Frozen source changed: {target}')
        mapped[str(target.resolve())]=expected
    matches=[value for key,value in receipt.get('manifests',{}).items() if Path(key).name=='dev-clean-fixed128.jsonl']
    if len(matches)!=1 or file_hash(manifest)!=matches[0]:
        raise ValueError('Frozen fixed128 manifest differs')
    return dict(source_sha256=mapped,manifests={str(Path(manifest).resolve()):matches[0]},
                model_id=MODEL_ID,model_revision=MODEL_REVISION,receipt_sha256=file_hash(receipt_path))


def tensor_fingerprint(tensor,cast=None):
    dtype=cast if cast is not None and tensor.is_floating_point() else tensor.dtype
    digest=hashlib.sha256()
    flattened=tensor.detach().reshape(-1)
    for start in range(0,flattened.numel(),1024**2):
        block=flattened[start:start+1024**2].to(device='cpu',dtype=dtype).contiguous()
        if (block.is_floating_point() or block.is_complex()) and not bool(torch.isfinite(block).all()):
            raise FloatingPointError('Nonfinite tensor in control evidence')
        digest.update(block.view(torch.uint8).numpy().tobytes())
    return dict(shape=list(tensor.shape),dtype=str(dtype),sha256=digest.hexdigest(),numel=tensor.numel())


def fingerprint_model(model,cast=None):
    params=dict(model.named_parameters(remove_duplicate=False))
    buffers=dict(model.named_buffers(remove_duplicate=False))
    state=model.state_dict(keep_vars=True)
    if not set(params)<=set(state):
        raise ValueError('Model state omits actual named parameters')
    fingerprints={}
    aliases={}
    for name,tensor in sorted(state.items()):
        if not isinstance(tensor,torch.Tensor):
            raise ValueError(f'Unsupported non-tensor model state: {name}')
        if name not in params and name not in buffers:
            raise ValueError(f'Unclassified model state: {name}')
        record=tensor_fingerprint(tensor,cast)
        record['kind']='parameter' if name in params else 'persistent_buffer'
        fingerprints[name]=record
        if tensor.numel():
            key=(str(tensor.device),tensor.untyped_storage().data_ptr())
            aliases.setdefault(key,[]).append(name)
    return dict(tensors=fingerprints,aliases=sorted(sorted(names) for names in aliases.values() if len(names)>1))


def canonical_tokens(tokens,eos_ids):
    result=[]
    for token in tokens:
        result.append(int(token))
        if int(token) in eos_ids:
            break
    return result


def json_value(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value,torch.dtype):
        return str(value)
    if isinstance(value,dict):
        return {str(key):json_value(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_value(item) for item in value]
    if isinstance(value,float) and not math.isfinite(value):
        raise FloatingPointError('Nonfinite generation-config value')
    return value


def effective_generation(config):
    if str(config.get_generation_mode().value)!='greedy_search':
        raise ValueError('Control requires actual greedy-search generation mode')
    value=json_value(config.to_dict())
    # These sampling knobs are inactive only after verifying GREEDY_SEARCH above.
    inactive={'temperature','top_k','top_p','min_p','typical_p','epsilon_cutoff','eta_cutoff'}
    metadata={'transformers_version','_from_model_config','_commit_hash','_original_object_hash'}
    return {key:item for key,item in value.items() if key not in inactive|metadata}


class GreedyTrace(LogitsProcessor):
    """Observe final processed next-token logits without changing any score."""
    def __init__(self):
        self.steps=[]
    def __call__(self,input_ids,scores):
        if bool((torch.isnan(scores)|torch.isposinf(scores)).any()):
            raise FloatingPointError('Nonfinite generation decision scores')
        values,indices=torch.topk(scores,2,dim=-1)
        values,indices=values.detach().float().cpu().tolist(),indices.detach().cpu().tolist()
        current=[]
        for value,index in zip(values,indices):
            if not math.isfinite(value[0]):
                raise FloatingPointError('No finite greedy top1 logit')
            second=value[1] if math.isfinite(value[1]) else None
            current.append(dict(top2_token_ids=index,top2_logits=[value[0],second],
                                margin=value[0]-second if second is not None else None))
        self.steps.append(current)
        return scores


def align_special_tokens(trainer):
    # This is the actual pinned Trainer method normally called from train(), not a reimplementation.
    Trainer._align_special_tokens(trainer)


def compare_evaluations(base,candidate,projection=None):
    differences=[]
    for label,evidence in [('base',base),('reload',candidate)]:
        if 'post_generation_model' in evidence and evidence['model']!=evidence['post_generation_model']:
            differences.append(label+'_persistent_state_changed_during_generation')
        if projection is not None and evidence['model']!=projection:
            differences.append('prepared_bf16_projection_differs_from_'+label)
    for key in ('model','post_generation_model','inputs','generation','loading_info','runtime'):
        if base.get(key)!=candidate.get(key):
            differences.append(key)
    left,right=base['predictions'],candidate['predictions']
    first=None
    if [r['id'] for r in left]!=[r['id'] for r in right]:
        differences.append('prediction_order')
    else:
        for l,r in zip(left,right):
            if l['token_ids']==r['token_ids']:
                continue
            maximum=max(len(l['token_ids']),len(r['token_ids']))
            position=next(i for i in range(maximum) if (l['token_ids'][i] if i<len(l['token_ids']) else None)!=(r['token_ids'][i] if i<len(r['token_ids']) else None))
            first=dict(id=l['id'],position=position,
                base_token=l['token_ids'][position] if position<len(l['token_ids']) else None,
                candidate_token=r['token_ids'][position] if position<len(r['token_ids']) else None,
                base_trace=l.get('top2_trace',[])[position] if position<len(l.get('top2_trace',[])) else None,
                candidate_trace=r.get('top2_trace',[])[position] if position<len(r.get('top2_trace',[])) else None,
                common_prefix=l['token_ids'][:position])
            break
    if first:
        differences.append('generation_tokens')
    return dict(passed=not differences,differences=differences,
                token_only_difference=differences==['generation_tokens'],first_token_difference=first,
                conclusion='pass_on_this_fixed128_only' if not differences else 'control_not_passed_cause_undetermined')


def runtime_info():
    return dict(packages={name:importlib.metadata.version(name) for name in ('torch','transformers','accelerate','qwen-asr')},
        python=sys.version,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),eval_dtype='torch.bfloat16',attention='sdpa',
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def load_official(source,dtype,training_receipt):
    import qwen_asr  # Registers precisely the same AutoModel/AutoProcessor used by the official wrapper.
    model,loading=AutoModel.from_pretrained(str(source),dtype=dtype,device_map=None,
        attn_implementation='sdpa',token=False,output_loading_info=True)
    processor=AutoProcessor.from_pretrained(str(source),fix_mistral_regex=True,token=False)
    info={key:json_value(loading.get(key,[])) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')}
    package_hashes={label:file_hash(inspect.getsourcefile(component.__class__)) for label,component in [('model',model),('processor',processor)]}
    for name,expected in training_receipt['packages'].items():
        if importlib.metadata.version(name)!=expected:
            raise ValueError(f'Original software version changed: {name}')
    if package_hashes!=training_receipt['package_source_sha256']:
        raise ValueError('Original official model/processor source changed')
    return model,processor,info,package_hashes


def verify_loading(info):
    if any(info.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError(f'Unexpected model loading diagnostics: {info}')


def prepare(args):
    # patch_outer_forward mutates a CLASS. This worker must exit before either evaluation starts.
    from .qwen_train import SafeTrainer,patch_outer_forward,sanitize_generation_configs
    from .qwen_data import QwenCollator
    from transformers import TrainingArguments,set_seed
    original=json.loads(args.training_receipt.read_text())
    set_seed(42)
    model,processor,loading,package_hashes=load_official(args.source,torch.float32,original)
    atomic_json(args.output/'loading_info.json',loading)
    verify_loading(loading)
    patch_outer_forward(model)
    sanitize_generation_configs(model)
    model.config.use_cache=False
    model.thinker.config.use_cache=False
    if any(p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Preparation weights are not all FP32')
    dataset=QwenManifestDataset(args.manifest,None,42)
    config=original['config']
    training_args=TrainingArguments(output_dir=str(args.output/'trainer'),
        per_device_train_batch_size=int(config['batch_size']),
        gradient_accumulation_steps=int(config['gradient_accumulation_steps']),
        learning_rate=float(config['learning_rate']),bf16=True,fp16=False,tf32=True,
        gradient_checkpointing=bool(config['gradient_checkpointing']),
        gradient_checkpointing_kwargs={'use_reentrant':False},remove_unused_columns=False,
        label_names=['labels'],report_to=[],save_safetensors=True,push_to_hub=False)
    trainer=SafeTrainer(model=model,args=training_args,train_dataset=dataset,eval_dataset=dataset,
        data_collator=QwenCollator(processor),processing_class=processor)
    trainer.model_accepts_loss_kwargs=False
    align_special_tokens(trainer)
    before=fingerprint_model(model)
    atomic_json(args.output/'state_before_save.json',before)
    atomic_json(args.output/'bf16_projection.json',fingerprint_model(model,torch.bfloat16))
    args.snapshot.mkdir(parents=True,exist_ok=False)
    trainer.save_model(str(args.snapshot),_internal_call=True)
    processor.save_pretrained(args.snapshot)
    after=fingerprint_model(model)
    atomic_json(args.output/'state_after_save.json',after)
    if before!=after or trainer.optimizer is not None or trainer.state.global_step!=0:
        raise ValueError('Preparation changed model state or created an optimizer/update')
    atomic_json(args.output/'prepared.json',dict(status='complete',optimizer_updates=0,backward_calls=0,
        train_called=False,snapshot=str(args.snapshot),loading_info=loading,package_source_sha256=package_hashes,
        trainer_alignment_method_sha256=hashlib.sha256(inspect.getsource(Trainer._align_special_tokens).encode()).hexdigest(),
        outer_generation=json_value(model.generation_config.to_dict()),thinker_generation=json_value(model.thinker.generation_config.to_dict()),
        outer_use_cache=model.config.use_cache,thinker_use_cache=model.thinker.config.use_cache,
        runtime=runtime_info()))


@torch.inference_mode()
def evaluate(args):
    from transformers import WhisperTokenizer,set_seed
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    from qwen_asr.inference.utils import parse_asr_output
    original=json.loads(args.training_receipt.read_text())
    set_seed(42)
    model,processor,loading,package_hashes=load_official(args.source,torch.bfloat16,original)
    atomic_json(args.output/'loading_info.json',loading)
    verify_loading(loading)
    if model.__class__.forward.__module__=='asr_lab.qwen_train':
        raise ValueError('Class-level training forward leaked into generation worker')
    model=model.to(torch.device('cuda:0')).eval()
    processor.tokenizer.padding_side='left'
    fingerprint=fingerprint_model(model)
    atomic_json(args.output/'model.json',fingerprint)
    normalizer_tokenizer=WhisperTokenizer.from_pretrained('openai/whisper-small',revision=WHISPER_REVISION,token=False)
    normalizer=EnglishTextNormalizer(normalizer_tokenizer.english_spelling_normalizer)
    dataset=QwenManifestDataset(args.manifest,None,42)
    if len(dataset)!=128:
        raise ValueError('Exactly the frozen128 examples are required')
    if [r['id'] for r in dataset.records]!=original['eval_ids']:
        raise ValueError('Fixed128 order differs from original training receipt')
    generation=[]
    input_records=[]
    predictions=[]
    original_get=model.thinker._get_logits_processor
    def capture(*positional,**kwargs):
        config=kwargs.get('generation_config',positional[0] if positional else None)
        processors=original_get(*positional,**kwargs)
        generation.append(dict(active=effective_generation(config),raw=json_value(config.to_dict()),
            special_tokens={name:json_value(getattr(config,name,None)) for name in
                ('_bos_token_tensor','_eos_token_tensor','_pad_token_tensor','_decoder_start_token_tensor')},
            logits_processor_types=[type(p).__module__+'.'+type(p).__qualname__ for p in processors]))
        return processors
    model.thinker._get_logits_processor=capture
    try:
        for start in range(0,len(dataset),16):
            rows=dataset.records[start:start+16]
            inputs=processor(text=[prompt_text(processor,force_language=True)]*len(rows),
                audio=[load_audio(row) for row in rows],return_tensors='pt',padding=True,
                padding_side='left',truncation=False)
            inputs={key:value.to('cuda:0',dtype=torch.bfloat16) if value.is_floating_point() else value.to('cuda:0') for key,value in inputs.items()}
            hashes={key:tensor_fingerprint(value) for key,value in sorted(inputs.items())}
            input_records.append(dict(ids=[row['id'] for row in rows],tensors=hashes))
            torch.save({key:value.detach().cpu() for key,value in inputs.items()},args.output/f'inputs-{start//16:02d}.pt')
            trace=GreedyTrace()
            # Outer.generate itself sets return_dict_in_generate=True; never pass that twice.
            generated=model.generate(**inputs,max_new_tokens=512,do_sample=False,num_beams=1,use_cache=True,
                                     logits_processor=LogitsProcessorList([trace]))
            continuation=generated.sequences[:,inputs['input_ids'].shape[1]:]
            raw_text=processor.batch_decode(continuation,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            for index,(row,raw,tokens) in enumerate(zip(rows,raw_text,continuation)):
                full=tokens.detach().cpu().tolist()
                canonical=canonical_tokens(full,EOS_IDS)
                _,hypothesis=parse_asr_output(raw,user_language='English')
                predictions.append(dict(id=row['id'],reference=row['text'],raw_decoded=raw,hypothesis=hypothesis,
                    reference_normalized=normalizer(row['text']).strip(),hypothesis_normalized=normalizer(hypothesis).strip(),
                    raw_generated_token_ids=full,token_ids=canonical,
                    top2_trace=[step[index] for step in trace.steps[:len(canonical)]],
                    generation_truncated=len(full)>=512 and not any(token in EOS_IDS for token in full)))
            atomic_json(args.output/'progress.json',dict(completed=len(predictions),total=128))
    finally:
        model.thinker._get_logits_processor=original_get
    if len(generation)!=8:
        raise ValueError('Missing effective generation-config captures')
    metrics=normalized_wer([row['reference'] for row in predictions],[row['hypothesis'] for row in predictions],normalizer)
    metrics.update(examples=128,evaluation_batch_size=16,manifest_sha256=file_hash(args.manifest),
        model_revision=MODEL_REVISION,role=args.role,generation_truncated_count=sum(row['generation_truncated'] for row in predictions))
    if metrics['generation_truncated_count']:
        raise ValueError('Control generation exhausted token budget')
    with (args.output/'predictions.jsonl').open('w') as stream:
        for row in predictions:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    atomic_json(args.output/'metrics.json',metrics)
    # Compare active rules; retain every raw config separately so inactive differences are visible.
    atomic_json(args.output/'generation_raw.json',generation)
    atomic_json(args.output/'evidence.json',dict(model=fingerprint,inputs=input_records,
        generation=[dict(active=row['active'],special_tokens=row['special_tokens'],logits_processor_types=row['logits_processor_types']) for row in generation],
        loading_info=loading,runtime=runtime_info(),package_source_sha256=package_hashes,predictions=predictions,
        post_generation_model=fingerprint_model(model),metrics=metrics,forward_module=model.__class__.forward.__module__))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','evaluate'])
    parser.add_argument('--receipt',required=True,type=Path)
    parser.add_argument('--training-receipt',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--role',choices=['base','reload','base-repeat'],default='base')
    args=parser.parse_args()
    freeze_receipt(args.receipt,args.manifest)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('Control workers require exclusively physical GPU1 and one process')
    if not torch.cuda.is_available():raise RuntimeError('GPU1 is required; no CPU training fallback')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase=='prepare' and args.snapshot is None:parser.error('--snapshot is required for prepare')
    prepare(args) if args.phase=='prepare' else evaluate(args)


if __name__=='__main__':
    main()
