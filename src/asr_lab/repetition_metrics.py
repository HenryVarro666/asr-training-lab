"""Predeclared CPU-only exact-period statistics; no acoustic or training-cause inference."""
from __future__ import annotations


def longest_periodic_run(sequence,max_period):
    if not isinstance(max_period,int) or max_period<1:raise ValueError('Positive max_period required')
    values=list(sequence);n=len(values)
    def empty(period=None):return dict(period=period,start=None,end_exclusive=None,repeat_count=0,span=0,coverage=0.,unit=[])
    best=empty();by_period=[]
    for period in range(1,min(max_period,n//2)+1):
        current=empty(period);matching=0
        for start in range(n-period-1,-1,-1):
            matching=matching+1 if values[start]==values[start+period] else 0
            count=1+matching//period
            if count<2:continue
            span=count*period
            if span>current['span'] or (span==current['span'] and start<current['start']):
                current=dict(period=period,start=start,end_exclusive=start+span,repeat_count=count,
                    span=span,coverage=span/n,unit=values[start:start+period])
        end=current['end_exclusive']
        partial=[]
        if end is not None:
            for offset in range(period-1):
                if end+offset>=n or values[end+offset]!=current['unit'][offset]:break
                partial.append(values[end+offset])
        current['following_partial_period']=partial
        current['partial_period_counted_as_complete']=False
        by_period.append(current)
        if current['span']>best['span']:best=dict(current)
    return dict(best,sequence_length=n,by_period=by_period)


def repetition_observation(tokens,eos_ids,preparse_text):
    tokens=list(tokens);eos=set(eos_ids)
    if len(tokens)>512:raise ValueError('Returned singleton width exceeds frozen512 budget')
    first=next((index for index,value in enumerate(tokens) if value in eos),None)
    analysis=tokens if first is None else tokens[:first]
    token_run=longest_periodic_run(analysis,16)
    word_run=longest_periodic_run(preparse_text.split(),8)
    def strong(result):return any(row['repeat_count']>=20 and row['coverage']>=.5 for row in result['by_period'])
    token_strong=strong(token_run);word_strong=strong(word_run)
    exhausted=len(tokens)==512 and first is None
    category=('eos_terminated' if first is not None else
        ('budget_exhausted_with_periodic_repetition' if token_strong or word_strong else 'budget_exhausted_without_detected_periodicity')
        if exhausted else 'unexpected_stop_without_eos')
    return dict(returned_width=len(tokens),first_effective_eos_position=first,analysis_token_count=len(analysis),
        token_periodic_run=token_run,preparse_word_periodic_run=word_run,
        strong_token_repetition=token_strong,strong_word_repetition=word_strong,
        strong_periodic_repetition=token_strong or word_strong,budget_exhausted=exhausted,
        budget_loop=(token_strong or word_strong) and exhausted,operational_category=category,
        definition=dict(token_periods=[1,16],word_periods=[1,8],minimum_repeated_periods=20,minimum_sequence_coverage=.5,
            words='Whitespace-split raw skip-special decode before parser/repetition-fix; no lexical normalization.',
            tokens='Effective EOS and everything after its first occurrence excluded from derived token statistics; raw evidence unchanged.'))


def summarize_onset(rows):
    expected=[('base',0),('step025',25),('step050',50),('step100',100)]
    if [(row['stage'],row['step']) for row in rows]!=expected:raise ValueError('Exactly the four ordered saved stages required')
    summary=dict(exact_first_update_identified=False,training_cause_identified=False)
    for key in ('strong_periodic_repetition','budget_loop'):
        positive=next((index for index,row in enumerate(rows) if row[key]),None)
        seen=False;reversal=False
        for row in rows:
            if row[key]:seen=True
            elif seen:reversal=True
        summary[key]=dict(earliest_observed_stage=rows[positive]['stage'] if positive is not None else None,
            earliest_observed_step=rows[positive]['step'] if positive is not None else None,
            previous_sampled_negative_step=rows[positive-1]['step'] if positive is not None and positive>0 else None,
            observed_positive_then_negative=reversal,
            interpretation='First observed saved point only. Earlier negative sampled endpoints do not prove exact first onset, monotonicity, persistence, or training cause; unsampled updates are unknown.')
    return summary
