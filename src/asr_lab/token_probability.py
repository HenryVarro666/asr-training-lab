"""Bounded inference-only token distributions and CPU statistics for frozen trajectories."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from . import qwen_truncation_probe as probe
from .qwen_roundtrip_control import tensor_fingerprint,runtime_info,json_value
from .qwen_data import load_audio,prompt_text
from .training_utils import atomic_json,sha256_file
from .prepare_data import tree_size,check_space
from .probe_supervision import worker_gpu_preflight

ORDER=('base','step050','step100')
FLAGS=dict(output_logits=True,output_scores=True)
CAP=2*1024**3
TOLERANCES=dict(logprob_entropy_margin=5e-5,probability=1e-6,tf_generation_logprob_difference=1e-4)


def estimated_bytes(vocab,trajectory_lengths,gen_lengths):
    return (len(gen_lengths)*sum(trajectory_lengths)+sum(gen_lengths))*vocab*4


def teacher_inputs(original,targets):
    if not targets or original['input_ids'].ndim!=2 or original['input_ids'].shape[0]!=1:raise ValueError('Nonempty singleton trajectory required')
    result=dict(original)
    tail=torch.tensor([targets[:-1]],device=original['input_ids'].device,dtype=original['input_ids'].dtype)
    result['input_ids']=torch.cat([original['input_ids'],tail],dim=1)
    result['attention_mask']=torch.cat([original['attention_mask'],torch.ones_like(tail,dtype=original['attention_mask'].dtype)],dim=1)
    return result


def aligned_teacher_logits(logits,prefix_width,targets):
    expected=prefix_width+len(targets)-1
    if logits.ndim!=3 or logits.shape[0]!=1 or logits.shape[1]!=expected:raise ValueError('Teacher forward shape/shift differs')
    result=logits[0,prefix_width-1:prefix_width-1+len(targets)].detach().to(device='cpu',dtype=torch.float32,copy=True).contiguous()
    if result.shape[0]!=len(targets):raise ValueError('Missing target-aligned rows')
    return result


def check_generation_rules(current,expected):
    if current['logits_processor_types'] or expected['logits_processor_types']:raise ValueError('Only empty processors are admitted')
    actual=dict(current['effective']);prior=dict(expected['effective'])
    for name in FLAGS:
        if actual.pop(name,None) is not True:raise ValueError('Missing native observation output flag')
        prior.pop(name,None)
    if actual!=prior or current['special_tokens']!=expected['special_tokens']:raise ValueError('Effective generation-rule identity drift')


def require_finite(matrix):
    if matrix.ndim!=2 or matrix.dtype!=torch.float32:raise ValueError('Expected FP32 [rows,vocab] matrix')
    if not bool(torch.isfinite(matrix).all()):raise FloatingPointError('Nonfinite unmasked distribution')


def verify_choices(raw,scores,ids):
    require_finite(raw);require_finite(scores)
    if raw.shape!=scores.shape or raw.shape[0]!=len(ids):raise ValueError('Missing generation distribution rows')
    chosen=torch.argmax(scores,dim=-1).tolist()
    if chosen!=list(ids):raise ValueError('Processed-score argmax differs from actual chosen IDs')


def persist_distributions(path,raw,scores=None):
    path=Path(path)
    if path.exists():raise FileExistsError('Refusing distribution overwrite')
    equal=scores is not None and torch.equal(raw,scores)
    tensors={'logits':raw} if scores is None else {'raw_logits':raw,'scores':raw if equal else scores}
    probe.atomic_torch_save(path,tensors)
    return dict(path=str(path),sha256=sha256_file(path),bytes=path.stat().st_size,shape=list(raw.shape),dtype=str(raw.dtype),
        raw_scores_equal=equal if scores is not None else None,shared_storage_roles=['raw_logits','scores'] if equal else [],
        interpretation='Roles share storage only after full exact value comparison.' if equal else 'Distinct distributions retained.')


def row_statistics(row,target,candidates,eos_ids,pieces):
    if row.ndim!=1 or row.dtype!=torch.float32 or not bool(torch.isfinite(row).all()):raise FloatingPointError('Nonfinite/unexpected row')
    eos=sorted(set(eos_ids));candidates=sorted(set(candidates))
    wanted=sorted(set(candidates+eos+[int(target)]))
    if min(wanted)<0 or max(wanted)>=len(row):raise ValueError('Candidate outside vocabulary')
    logp=row-torch.logsumexp(row,dim=0);p=torch.exp(logp)
    if not bool(torch.isfinite(logp).all()) or abs(float(p.sum())-1.)>1e-4:raise FloatingPointError('Invalid normalized distribution')
    rank={};ties={}
    for offset in range(0,len(wanted),8):
        ids=wanted[offset:offset+8];values=row[ids]
        greater=(row[:,None]>values[None,:]).sum(dim=0).tolist()
        equal=(row[:,None]==values[None,:]).sum(dim=0).tolist()
        for identifier,count,tied in zip(ids,greater,equal):rank[identifier]=1+count;ties[identifier]=tied
    names=dict(zip(wanted,pieces(wanted)))
    def item(identifier):return dict(id=identifier,piece=names.get(identifier),logit=float(row[identifier]),
        logprob=float(logp[identifier]),prob=float(p[identifier]),rank=rank[identifier],tie_count=ties[identifier])
    k=min(10,len(row));threshold=torch.topk(row,k).values[-1]
    greater_ids=torch.nonzero(row>threshold).flatten().tolist()
    equal_ids=torch.nonzero(row==threshold).flatten()[:k-len(greater_ids)].tolist()
    top=sorted(greater_ids+equal_ids,key=lambda i:(-float(row[i]),i))
    top_pieces=pieces(top)
    top_items=[dict(id=i,piece=piece,logit=float(row[i]),logprob=float(logp[i]),prob=float(p[i])) for i,piece in zip(top,top_pieces)]
    top2=torch.topk(row,min(2,len(row))).values
    entropy=float(-(p*logp).sum())
    if not math.isfinite(entropy):raise FloatingPointError('Nonfinite entropy')
    eos_items=[item(identifier) for identifier in eos]
    return dict(target=item(int(target)),candidates=[item(identifier) for identifier in candidates],top10=top_items,
        top1_minus_top2_margin=float(top2[0]-top2[1]) if len(top2)>1 else None,
        top1_tie_count=int((row==row.max()).sum()),entropy=entropy,
        eos=dict(ids=eos,individual=eos_items,total_prob=float(p[eos].sum()),total_logprob=float(torch.logsumexp(logp[eos],0))))


def prefix_hash(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()


def trajectory_forks(trajectories):
    result=[]
    for i,left in enumerate(ORDER):
        for right in ORDER[i+1:]:
            a,b=trajectories[left],trajectories[right];j=0
            while j<min(len(a),len(b)) and a[j]==b[j]:j+=1
            result.append(dict(left=left,right=right,lcp_length=j,first_difference_index=j if a!=b else None,
                shared_continuation_prefix=a[:j],left_next_id=a[j] if j<len(a) else None,right_next_id=b[j] if j<len(b) else None))
    return result


def matrix_statistics(matrix,targets,prompt,candidates,eos,pieces):
    require_finite(matrix)
    if matrix.shape[0]!=len(targets):raise ValueError('Distribution target mapping differs')
    rows=[]
    for j,target in enumerate(targets):
        row=row_statistics(matrix[j],target,candidates,eos,pieces)
        row.update(j=j,input_position=len(prompt)-1+j,target_id=target,
            prefix=dict(length=len(prompt)+j,sha256=prefix_hash(prompt+list(targets[:j]))))
        rows.append(row)
    nll=-sum(row['target']['logprob'] for row in rows)
    return dict(rows=rows,total_target_nll=nll,mean_target_nll=nll/len(rows),candidate_ids=candidates,
        prefix_hash_format="sha256(json.dumps(full_prefix_ids,separators=(',',':')).encode())",audit_tolerances=TOLERANCES,
        ranking='1 + count(logit > candidate_logit); tie_count counts equal logits; top10 ordered by logit descending then token ID ascending')


def reserve(identity,bytes_needed):
    root=Path(identity['project_root']);run=Path(identity['run_root'])
    growth=max(tree_size(run),tree_size(root)-identity['initial_project_bytes'])
    if growth+bytes_needed+64*1024**2>CAP:raise RuntimeError('Token probability2GiB incremental budget would be exceeded')
    check_space(root,500,300,bytes_needed+1024**2)


def input_gate(inputs,metadata,reference,physical_gpu=1):
    common=reference['common_input']
    for key in ('input_ids','attention_mask','prompt_padded_width','effective_prompt_tokens','tensors'):
        if inputs[key]!=common[key]:raise ValueError('Frozen singleton preprocessing identity drift')
    expected=reference['common_runtime_metadata']
    for key,value in expected.items():
        if key=='logits_collected':
            if metadata[key] is not True:raise ValueError('Observation metadata must record collected logits')
        elif key=='generation_kwargs':
            actual=dict(metadata[key])
            for name in FLAGS:
                if actual.pop(name,None) is not True:raise ValueError('Missing explicit observation flags')
            if actual!=value:raise ValueError('Decision kwargs changed')
        elif key=='runtime':
            if physical_gpu not in (0,1):raise ValueError('Unsupported physical GPU identity')
            resource_expected=dict(value,physical_gpu=str(physical_gpu))
            if metadata.get(key)!=resource_expected:raise ValueError('Runtime/model input identity drift:runtime')
        elif metadata.get(key)!=value:raise ValueError(f'Runtime/model input identity drift:{key}')


def generation_capture(model,inputs,output,identity):
    result=model.generate(**inputs,**probe.GENERATION_KWARGS,**FLAGS)
    seq=result.sequences.detach().cpu();width=inputs['input_ids'].shape[1]
    probe.atomic_torch_save(output/'raw-tokens.pt',dict(sequences=seq))
    raw=dict(prompt_padded_width=width,full_sequences=seq.tolist())
    valid=seq.ndim==2 and seq.shape[0]==1 and seq.shape[1]>=width
    if valid:raw['continuation_token_ids']=seq[:,width:].tolist()
    atomic_json(output/'raw-tokens.json',raw)
    if not valid or not torch.equal(seq[:,:width],inputs['input_ids'].cpu()):raise ValueError('Generated prefix/shape mismatch; raw retained')
    if result.logits is None or result.scores is None:raise ValueError('Missing official logits/scores return fields')
    dtype=dict(raw=[str(x.dtype) for x in result.logits],scores=[str(x.dtype) for x in result.scores])
    raw_matrix=torch.cat([x.detach().to(device='cpu',dtype=torch.float32) for x in result.logits],dim=0)
    scores=torch.cat([x.detach().to(device='cpu',dtype=torch.float32) for x in result.scores],dim=0)
    reserve(identity,(raw_matrix.numel()+ (0 if torch.equal(raw_matrix,scores) else scores.numel()))*4)
    receipt=persist_distributions(output/'distributions.pt',raw_matrix,scores)
    receipt['scores_shape']=list(scores.shape)
    receipt['original_return_dtypes']=dtype
    atomic_json(output/'distribution-manifest.json',receipt)
    ids=raw['continuation_token_ids'][0]
    verify_choices(raw_matrix,scores,ids)
    if not torch.equal(raw_matrix,scores):raise ValueError('Empty-processor raw/scores unexpectedly differ; both retained')
    return raw,raw_matrix,receipt


def difference_rows(teacher,generation,teacher_stats,generation_stats):
    if teacher.shape!=generation.shape:raise ValueError('Own trajectory distribution rows differ')
    rows=[]
    for j,(left,right) in enumerate(zip(teacher,generation)):
        delta=(left-right).abs()
        rows.append(dict(j=j,max_absolute_logit_difference=float(delta.max()),mean_absolute_logit_difference=float(delta.mean()),
            target_logprob_difference=teacher_stats['rows'][j]['target']['logprob']-generation_stats['rows'][j]['target']['logprob'],
            eos_total_logprob_difference=teacher_stats['rows'][j]['eos']['total_logprob']-generation_stats['rows'][j]['eos']['total_logprob'],
            argmax_equal=int(left.argmax())==int(right.argmax()),teacher_margin=teacher_stats['rows'][j]['top1_minus_top2_margin'],
            generation_margin=generation_stats['rows'][j]['top1_minus_top2_margin']))
    return dict(rows=rows,interpretation='Full no-cache teacher forcing and cached generation may differ numerically; these are observations, not an equality acceptance test.')


@torch.inference_mode()
def worker(args):
    identity=json.loads(args.identity.read_text())
    worker_gpu_preflight(identity);probe.verify_identity(identity)
    reference=json.loads(Path(identity['trajectory_reference']).read_text());stage=identity['stage']
    row=reference['target']['manifest_row'];audio=load_audio(row)
    pcm=hashlib.sha256(np.asarray(audio,dtype='<f4').tobytes()).hexdigest()
    if pcm!=reference['target']['pcm_sha256']:raise ValueError('Decoded audio PCM differs')
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import parse_asr_output,detect_and_fix_repetitions
    wrapper=Qwen3ASRModel.from_pretrained(identity['checkpoint'],dtype=torch.bfloat16,device_map=None,attn_implementation='sdpa',token=False)
    atomic_json(args.output/'gpu-preflight.json',worker_gpu_preflight(identity))
    model=wrapper.model.to('cuda:0').eval();processor=wrapper.processor;processor.tokenizer.padding_side='left'
    if model.__class__.forward.__module__=='asr_lab.qwen_train':raise ValueError('Training forward leaked')
    for label,component in (('model',model),('processor',processor)):
        if sha256_file(inspect.getsourcefile(component.__class__))!=identity['package_source_sha256'][label]:raise ValueError('Official class source changed')
    if any(p.is_floating_point() and p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('Expected BF16 model weights')
    vocab=model.thinker.config.get_text_config().vocab_size
    if vocab!=identity['vocab_size']:raise ValueError('Vocabulary/budget differs')
    prompts=[prompt_text(processor,force_language=True)]
    inputs=processor(text=prompts,audio=[audio],return_tensors='pt',padding=True,padding_side='left',truncation=False)
    inputs={key:value.to('cuda:0',dtype=torch.bfloat16) if value.is_floating_point() else value.to('cuda:0') for key,value in inputs.items()}
    input_record=dict(ids=[row['id']],prompts=prompts,tensors={key:tensor_fingerprint(value) for key,value in sorted(inputs.items())},
        input_ids=inputs['input_ids'].cpu().tolist(),attention_mask=inputs['attention_mask'].cpu().tolist(),
        prompt_padded_width=inputs['input_ids'].shape[1],effective_prompt_tokens=inputs['attention_mask'].sum(-1).cpu().tolist(),
        feature_attention_mask=inputs['feature_attention_mask'].cpu().tolist())
    probe.atomic_torch_save(args.output/'inputs.pt',{key:value.cpu() for key,value in inputs.items()})
    atomic_json(args.output/'inputs.json',input_record)
    metadata=dict(stage=stage,phase='input_gate',optimizer_updates=0,backward_calls=0,train_called=False,
        runtime=runtime_info(),generation_kwargs=dict(probe.GENERATION_KWARGS,**FLAGS),
        audio=[dict(id=row['id'],file_sha256=sha256_file(row['audio_filepath']),pcm_sha256=pcm,duration=row['duration'])],
        model_class=str(type(model)),processor_class=str(type(processor)),inference_mode=torch.is_inference_mode_enabled(),model_training=model.training,
        logits_collected=True,seed_reset=False,generate_calls=0,teacher_forward_calls=0,vocab_size=vocab)
    def phase(name):metadata['phase']=name;atomic_json(args.output/'metadata.json',metadata)
    phase('input_gate');input_gate(input_record,metadata,reference,physical_gpu=identity.get('physical_gpu',1))
    forks=trajectory_forks({name:reference['trajectories'][name]['token_ids'] for name in ORDER})
    for fork in forks:
        prefix=input_record['input_ids'][0]+fork['shared_continuation_prefix']
        fork.update(full_prefix_ids=prefix,full_prefix_sha256=prefix_hash(prefix),
            decoded_full_prefix=processor.tokenizer.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False),
            decoded_shared_continuation=processor.tokenizer.decode(fork['shared_continuation_prefix'],skip_special_tokens=False,clean_up_tokenization_spaces=False))
    atomic_json(args.output/'prefix-forks.json',forks)
    generation=args.output/'generation';generation.mkdir()
    captured=[];metadata['generate_calls']=1;phase('generate')
    try:
        with probe.capture_generation(model,captured):raw,generated,generation_manifest=generation_capture(model,inputs,generation,identity)
    finally:atomic_json(generation/'effective-generation.json',captured)
    if len(captured)!=1:raise ValueError('Missing generation configuration')
    check_generation_rules(captured[0],reference['effective_generation_rules'])
    actual_ids=raw['continuation_token_ids'][0]
    if actual_ids!=reference['trajectories'][stage]['token_ids']:raise ValueError('instrumented_trajectory_drift: evidence retained; no teacher forwards')
    eos=sorted(set(reference['effective_eos_ids']));pad=reference['effective_pad_id']
    normalizer=EnglishTextNormalizer(json.loads(Path(identity['spelling']).read_text()))
    decoded=probe.decode_records(processor,raw,[row],parse_asr_output,normalizer,eos,pad,detect_and_fix_repetitions)
    atomic_json(generation/'decoded.json',decoded)
    pieces=processor.tokenizer.convert_ids_to_tokens
    candidates=sorted(set(eos+[token for name in ORDER for token in reference['trajectories'][name]['token_ids']]))
    prompt=input_record['input_ids'][0]
    phase('generation_cpu_statistics')
    generation_stats=matrix_statistics(generated,actual_ids,prompt,candidates,eos,pieces)
    generation_stats['scores_rows_ref']='rows';generation_stats['raw_scores_exactly_equal']=True
    atomic_json(generation/'derived.json',generation_stats)
    manifests={'generation':generation_manifest}
    teacher_root=args.output/'teacher';teacher_root.mkdir()
    for trajectory in ORDER:
        folder=teacher_root/trajectory;folder.mkdir()
        targets=reference['trajectories'][trajectory]['token_ids']
        extended=teacher_inputs(inputs,targets)
        if int((extended['input_ids']==model.thinker.config.audio_token_id).sum())!=int((inputs['input_ids']==model.thinker.config.audio_token_id).sum()):
            raise ValueError('Teacher trajectory changes audio placeholders')
        probe.atomic_torch_save(folder/'inputs.pt',{key:value.cpu() for key,value in extended.items()})
        tf_input=dict(trajectory=trajectory,prompt_width=len(prompt),target_ids=targets,input_positions=list(range(len(prompt)-1,len(prompt)-1+len(targets))),
            input_ids=extended['input_ids'].cpu().tolist(),attention_mask=extended['attention_mask'].cpu().tolist(),
            tensors={key:tensor_fingerprint(value) for key,value in sorted(extended.items())},source_trajectory_files=reference['trajectories'][trajectory]['source_files'])
        atomic_json(folder/'inputs.json',tf_input)
        prior_rope=json_value(getattr(model.thinker,'rope_deltas',None));model.thinker.rope_deltas=None
        forward_receipt=dict(status='calling',trajectory=trajectory,use_cache=False,past_key_values=None,cache_position=None,position_ids=None,
            labels=None,rope_deltas_before_reset=prior_rope,rope_deltas_after_reset=None,explicit_teacher_forward=1,
            input_length=extended['input_ids'].shape[1],target_count=len(targets),shift='P-1+j',model_training=model.training,
            inference_mode=torch.is_inference_mode_enabled(),source_forward='Official unpatched model.thinker',
            source_forward_sha256=sha256_file(inspect.getsourcefile(model.thinker.__class__)))
        atomic_json(folder/'forward-receipt.json',forward_receipt)
        metadata['teacher_forward_calls']+=1;phase('teacher_forward_'+trajectory)
        outputs=model.thinker(**extended,use_cache=False,past_key_values=None,cache_position=None,position_ids=None,rope_deltas=None,labels=None,return_dict=True)
        original_dtype=str(outputs.logits.dtype)
        returned_cache_none=outputs.past_key_values is None
        matrix=aligned_teacher_logits(outputs.logits,len(prompt),targets)
        del outputs
        if matrix.shape[1]!=vocab:raise ValueError('Teacher vocabulary differs')
        reserve(identity,matrix.numel()*4)
        manifest=persist_distributions(folder/'distributions.pt',matrix)
        manifest['original_return_dtype']=original_dtype
        atomic_json(folder/'distribution-manifest.json',manifest);manifests[trajectory]=manifest
        require_finite(matrix)
        forward_receipt.update(status='complete',output_source_dtype=original_dtype,rope_deltas_after_forward=json_value(model.thinker.rope_deltas),
            saved_target_shape=list(matrix.shape),past_cache_returned=not returned_cache_none)
        atomic_json(folder/'forward-receipt.json',forward_receipt)
        if not returned_cache_none:raise ValueError('Teacher forward unexpectedly returned a cache')
        phase('teacher_cpu_statistics_'+trajectory)
        stats=matrix_statistics(matrix,targets,prompt,candidates,eos,pieces)
        atomic_json(folder/'derived.json',stats)
        if trajectory==stage:atomic_json(args.output/'own-trajectory-comparison.json',difference_rows(matrix,generated,stats,generation_stats))
        del matrix,stats
    probe.verify_identity(identity);phase('complete')
    if metadata['generate_calls']!=1 or metadata['teacher_forward_calls']!=3:raise ValueError('Incorrect top-level call budget')
    known={Path(item['path']).resolve():item for item in manifests.values()}
    artifacts={}
    for path in args.output.rglob('*'):
        if not path.is_file() or path.name in ('stdout.log','monitor.log','status.json','stage.json') or 'monitor' in path.relative_to(args.output).parts:continue
        item=known.get(path.resolve())
        artifacts[str(path.relative_to(args.output))]=dict(bytes=path.stat().st_size,sha256=item['sha256'] if item else sha256_file(path))
    atomic_json(args.output/'result.json',dict(status='complete',stage=stage,generate_calls=1,teacher_forward_calls=3,optimizer_updates=0,
        candidate_ids=candidates,manifests=manifests,artifacts=artifacts,identity_sha256=sha256_file(args.identity)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    atomic_json(args.output/'status.json',dict(status='running',phase='starting'))
    try:
        worker(args)
        atomic_json(args.output/'status.json',dict(status='complete',elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        atomic_json(args.output/'status.json',dict(status='failed',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
