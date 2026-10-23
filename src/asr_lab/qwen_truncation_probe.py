"""One transparent, inference-only capture of the frozen512-token Qwen generation path."""
from __future__ import annotations
import argparse
import csv
from contextlib import contextmanager
import importlib.metadata
import inspect
import json
import os
import subprocess
from pathlib import Path
import time

import jiwer
import numpy as np
import torch
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from .qwen_data import load_audio,prompt_text
from .qwen_roundtrip_control import json_value,effective_generation,tensor_fingerprint,runtime_info
from .training_utils import atomic_json,sha256_file

GENERATION_KWARGS=dict(max_new_tokens=512,do_sample=False,num_beams=1,use_cache=True)
TARGET='652-129742-0006'
STAGES=('batch16-first','batch16-repeat','target-single')


def require_gpu1_idle(ignore_own=False):
    def query(fields,kind):
        result=subprocess.run(['nvidia-smi',f'--query-{kind}='+fields,'--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True,timeout=10)
        return [row for row in csv.reader(result.stdout.splitlines(),skipinitialspace=True) if row]
    devices=query('index,uuid','gpu')
    matches=[row[1].strip() for row in devices if len(row)==2 and row[0].strip()=='1']
    if len(matches)!=1:raise ValueError('Cannot identify physical GPU1 UUID')
    applications=query('gpu_uuid,pid,process_name','compute-apps')
    own=os.getpid() if ignore_own else None
    busy=[row for row in applications if row[0].strip()==matches[0] and (len(row)<2 or row[1].strip()!=str(own))]
    if busy:raise RuntimeError(f'Physical GPU1 has other compute applications; no launch: {busy}')
    return dict(physical_gpu=1,uuid=matches[0],other_compute_apps=[],checked_unix=time.time(),
                ignored_own_pid=own,limitation='Point-in-time check cannot reserve a GPU against another scheduler.')


def select_probe_rows(reference,stage):
    if stage not in STAGES:raise ValueError('Only the three preset probes are authorized')
    rows=[value['manifest_row'] for value in reference['rows']]
    ids=[row['id'] for row in rows]
    if len(rows)!=16 or len(set(ids))!=16 or ids!=reference['ordered_ids'] or reference['target_id']!=TARGET or reference['target_index_in_batch']!=11 or ids[11]!=TARGET:
        raise ValueError('Original ordered batch16/target invariant differs')
    return [rows[11]] if stage=='target-single' else rows


def verify_identity(identity):
    for path,expected in identity['files'].items():
        if sha256_file(path)!=expected:raise ValueError(f'Identity file changed:{path}')
    for name,expected in identity['packages'].items():
        if importlib.metadata.version(name)!=expected:raise ValueError(f'Software version changed:{name}')


def token_evidence(ids,eos,pad,special):
    ids=[int(value) for value in ids];eos=set(eos)
    positions=[index for index,value in enumerate(ids) if value in eos]
    hardcoded=[index for index,value in enumerate(ids) if value in (151645,151643)]
    return dict(token_ids=ids,returned_continuation_width=len(ids),effective_eos_positions=positions,
        first_effective_eos_position=positions[0] if positions else None,
        legacy_eos_positions=hardcoded,pad_id=pad,pad_id_positions=[i for i,value in enumerate(ids) if value==pad],
        special_token_positions=[dict(position=i,token_id=value) for i,value in enumerate(ids) if value in special],
        positions_after_first_eos=list(range(positions[0]+1,len(ids))) if positions else [],
        legacy_generation_truncated=len(ids)>=512 and not hardcoded,
        effective_eos_generation_truncated=len(ids)>=512 and not positions,
        interpretation='Returned width includes batch padding. EOS/PAD overlap does not alone label any occurrence as padding; full tokens are retained.')


@contextmanager
def capture_generation(model,records):
    original=model.thinker._get_logits_processor
    def observe(*args,**kwargs):
        config=kwargs.get('generation_config',args[0] if args else None)
        processors=original(*args,**kwargs)
        records.append(dict(effective=effective_generation(config),raw=json_value(config.to_dict()),
            special_tokens={name:json_value(getattr(config,name,None)) for name in
                ('_bos_token_tensor','_eos_token_tensor','_pad_token_tensor','_decoder_start_token_tensor')},
            logits_processor_types=[type(value).__module__+'.'+type(value).__qualname__ for value in processors]
                if isinstance(processors,(list,tuple)) else None))
        return processors
    model.thinker._get_logits_processor=observe
    try:yield
    finally:model.thinker._get_logits_processor=original


def atomic_torch_save(path,value):
    temporary=path.with_name(path.name+'.tmp')
    try:
        torch.save(value,temporary)
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def generate_and_persist_raw(model,inputs,output):
    result=model.generate(**inputs,**GENERATION_KWARGS)
    sequences=result.sequences.detach().cpu()
    width=inputs['input_ids'].shape[1]
    shape_valid=(sequences.ndim==2 and sequences.shape[0]==inputs['input_ids'].shape[0] and sequences.shape[1]>=width)
    tensors=dict(sequences=sequences)
    raw=dict(prompt_padded_width=width,full_sequences=sequences.tolist(),sequence_shape_valid=shape_valid)
    if shape_valid:
        tensors['continuation']=sequences[:,width:]
        raw['continuation_token_ids']=sequences[:,width:].tolist()
    # Preserve the complete returned tensor even when the subsequent shape/prefix contract fails.
    atomic_torch_save(output/'raw-tokens.pt',tensors)
    atomic_json(output/'raw-tokens.json',raw)
    if not shape_valid:raise ValueError('Unexpected generated sequence dimensions; complete raw sequences retained')
    if not torch.equal(sequences[:,:width],inputs['input_ids'].detach().cpu()):
        raise ValueError('Returned sequence prompt prefix differs; complete raw sequences retained')
    return raw


def decode_records(processor,raw,rows,parser,normalizer,eos,pad,repeat_fix=None):
    ids=raw['continuation_token_ids']
    keep=processor.batch_decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    skipped=processor.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    records=[]
    for row,tokens,full,text in zip(rows,ids,keep,skipped):
        parsed=parser(text,user_language='English')
        normalized_ref=normalizer(row['text']).strip();normalized_hyp=normalizer(parsed[1]).strip()
        score=jiwer.process_words(normalized_ref,normalized_hyp)
        record=dict(id=row['id'],reference=row['text'],decoded_keep_special=full,decoded_skip_special=text,
            parser_input=text,parser_kwargs=dict(user_language='English'),parser_output=json_value(parsed),
            reference_normalized=normalized_ref,hypothesis_normalized=normalized_hyp,
            counts=dict(substitutions=score.substitutions,deletions=score.deletions,insertions=score.insertions,
                hits=score.hits,reference_words=score.hits+score.substitutions+score.deletions),
            token_pieces=processor.tokenizer.convert_ids_to_tokens(tokens),
            **token_evidence(tokens,eos,pad,set(processor.tokenizer.all_special_ids)))
        if repeat_fix is not None:
            fixed=repeat_fix(text.strip())
            record['repetition_fix_observation']=dict(input=text.strip(),output=fixed,
                input_characters=len(text.strip()),output_characters=len(fixed),
                input_words=len(text.strip().split()),output_words=len(fixed.split()),
                equals_actual_parser_hypothesis=fixed==parsed[1],
                method='Original helper called after the unmodified parser for observation; never replaces parser output.')
        records.append(record)
    return records


@torch.inference_mode()
def worker(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Worker requires physical GPU1 only')
    identity=json.loads(args.identity.read_text());verify_identity(identity)
    reference=json.loads(Path(identity['reference']).read_text());rows=select_probe_rows(reference,args.stage)
    audios=[];audio_evidence=[]
    for row in rows:
        audio=load_audio(row)
        pcm=__import__('hashlib').sha256(np.asarray(audio,dtype='<f4').tobytes()).hexdigest()
        if pcm!=row['sha256']:raise ValueError('Decoded PCM identity changed')
        audios.append(audio)
        audio_evidence.append(dict(id=row['id'],file_sha256=sha256_file(row['audio_filepath']),pcm_sha256=pcm,duration=row['duration']))
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import parse_asr_output,detect_and_fix_repetitions
    wrapper=Qwen3ASRModel.from_pretrained(identity['checkpoint'],dtype=torch.bfloat16,
        device_map=None,attn_implementation='sdpa',token=False)
    gpu_preflight=require_gpu1_idle(ignore_own=True)
    atomic_json(args.output/'gpu-preflight.json',gpu_preflight)
    model,processor=wrapper.model.to(torch.device('cuda:0')),wrapper.processor
    model.eval();processor.tokenizer.padding_side='left'
    if model.__class__.forward.__module__=='asr_lab.qwen_train':raise ValueError('Training forward leaked into probe')
    for label,component in (('model',model),('processor',processor)):
        if sha256_file(inspect.getsourcefile(component.__class__))!=identity['package_source_sha256'][label]:
            raise ValueError('Actual official model/processor class source changed')
    if any(p.is_floating_point() and p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('Model weights are notBF16')
    normalizer=EnglishTextNormalizer(json.loads(Path(identity['spelling']).read_text()))
    prompts=[prompt_text(processor,force_language=True)]*len(rows)
    inputs=processor(text=prompts,audio=audios,
        return_tensors='pt',padding=True,padding_side='left',truncation=False)
    inputs={key:value.to('cuda:0',dtype=torch.bfloat16) if value.is_floating_point() else value.to('cuda:0') for key,value in inputs.items()}
    atomic_torch_save(args.output/'inputs.pt',{key:value.detach().cpu() for key,value in inputs.items()})
    input_evidence=dict(ids=[row['id'] for row in rows],prompts=prompts,tensors={key:tensor_fingerprint(value) for key,value in sorted(inputs.items())},
        input_ids=inputs['input_ids'].cpu().tolist(),attention_mask=inputs['attention_mask'].cpu().tolist(),
        prompt_padded_width=inputs['input_ids'].shape[1],effective_prompt_tokens=inputs['attention_mask'].sum(-1).cpu().tolist())
    if 'feature_attention_mask' in inputs:input_evidence['feature_attention_mask']=inputs['feature_attention_mask'].cpu().tolist()
    atomic_json(args.output/'inputs.json',input_evidence)
    metadata=dict(stage=args.stage,optimizer_updates=0,backward_calls=0,train_called=False,
        ids=[row['id'] for row in rows],audio=audio_evidence,runtime=runtime_info(),
        generation_kwargs=GENERATION_KWARGS,raw_generation_missing=True,
        outer_generation=json_value(model.generation_config.to_dict()),thinker_generation=json_value(model.thinker.generation_config.to_dict()),
        tokenizer=dict(pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,all_special_ids=processor.tokenizer.all_special_ids),
        model_class=str(type(model)),processor_class=str(type(processor)),inference_mode=torch.is_inference_mode_enabled(),model_training=model.training,
        replayed_prior_batches=False,logits_collected=False,seed_reset=False,
        limitation='Fresh process handles original batch137 directly; earlier137 batches and original numerical history are not replayed.')
    atomic_json(args.output/'metadata.json',metadata)
    captured=[]
    try:
        with capture_generation(model,captured):raw=generate_and_persist_raw(model,inputs,args.output)
    finally:
        atomic_json(args.output/'effective-generation.json',captured)
    metadata['raw_generation_missing']=False
    atomic_json(args.output/'metadata.json',metadata)
    if len(captured)!=1:raise ValueError('Exactly one generation-config observation required')
    config=captured[0]['effective']
    if any(config.get(key)!=value for key,value in GENERATION_KWARGS.items()):raise ValueError('Effective generation kwargs differ')
    eos=captured[0]['special_tokens'].get('_eos_token_tensor')
    if eos is None:eos=config.get('eos_token_id')
    eos=[] if eos is None else ([eos] if isinstance(eos,int) else eos)
    if set(eos)!={151645,151643}:raise ValueError('Effective stopping EOS differs from frozen official path')
    pad=captured[0]['special_tokens'].get('_pad_token_tensor')
    if pad is None:pad=config.get('pad_token_id')
    if isinstance(pad,list):
        if len(pad)!=1:raise ValueError('Expected one effective PAD ID')
        pad=pad[0]
    decoded=decode_records(processor,raw,rows,parse_asr_output,normalizer,eos,pad,detect_and_fix_repetitions)
    atomic_json(args.output/'decoded.json',decoded)
    if len(decoded)!=len(rows):raise ValueError('Missing decoded records')
    verify_identity(identity)
    artifacts={path.name:dict(sha256=sha256_file(path),bytes=path.stat().st_size) for path in args.output.iterdir()
        if path.is_file() and path.name not in ('status.json','stdout.log','monitor.log','stage.json')}
    atomic_json(args.output/'result.json',dict(status='complete',stage=args.stage,generate_calls=1,consumed_examples=len(rows),
        optimizer_updates=0,raw_generation_missing=False,legacy_truncated_ids=[record['id'] for record in decoded if record['legacy_generation_truncated']],
        effective_eos_ids=eos,legacy_eos_ids=[151645,151643],eos_sets_equal=set(eos)=={151645,151643},
        artifacts=artifacts,identity_sha256=sha256_file(args.identity)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity',required=True,type=Path)
    parser.add_argument('--stage',required=True,choices=STAGES)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    atomic_json(args.output/'status.json',dict(status='running',stage=args.stage))
    try:
        worker(args)
        atomic_json(args.output/'status.json',dict(status='complete',elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        atomic_json(args.output/'status.json',dict(status='failed',error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic()-started,raw_generation_missing=not(args.output/'raw-tokens.pt').exists()))
        raise


if __name__=='__main__':main()
