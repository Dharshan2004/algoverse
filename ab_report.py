"""Audit four arms and compare format validity separately from source-label quality."""
import argparse,json,math
from collections import Counter
from pathlib import Path
from pilot_report import audit, summarize, require,adapt_metrics
import metrics

def quality(rows):
    correct=lambda r:r['verdict']==('UNSAFE' if r['mapped_unsafe'] else 'SAFE')
    valid=[r for r in rows if r['verdict']!='INVALID']
    return {'rows':len(rows),'format_valid_count':len(valid),'format_valid_pct':100*len(valid)/len(rows),
        'correct_count_including_invalid_denominator':sum(map(correct,rows)),
        'accuracy_all_pct':100*sum(map(correct,rows))/len(rows),
        'accuracy_valid_only_pct':100*sum(map(correct,valid))/len(valid) if valid else None,
        'counts':dict(Counter(('source_unsafe' if r['mapped_unsafe'] else 'source_safe')+'/'+r['verdict'] for r in rows)),
        'ab_mass_min':min(r['ab_mass'] for r in rows),'ab_mass_below_001_count':sum(r['ab_mass']<.01 for r in rows),
        'per_condition':summarize(rows)}

def paired(a,b):
    key=lambda r:(r['source_id'],r['variant'],r['mapping'])
    index={key(r):r for r in b};require(len(a)==len(b)==len(index),'Comparison size mismatch')
    transitions=Counter();valid_changes=0;newly_correct=0;newly_wrong=0;max_logp_diff=0.;state_equal=0
    for r in a:
        q=index[key(r)]
        for field in ['mapped_unsafe','prompt_sha256','input_ids_sha256','model_revision']:
            require(r[field]==q[field],'Paired prompt/label/model mismatch: '+field)
        transitions[r['verdict']+'->'+q['verdict']]+=1
        if r['verdict']!='INVALID' and r['verdict']!=q['verdict']:valid_changes+=1
        if r['verdict']=='INVALID':
            if q['verdict']==('UNSAFE' if r['mapped_unsafe'] else 'SAFE'):newly_correct+=1
            elif q['verdict']!='INVALID':newly_wrong+=1
        max_logp_diff=max(max_logp_diff,abs(r['logp_A']-q['logp_A']),abs(r['logp_B']-q['logp_B']))
    return {'rows':len(a),'transitions':dict(transitions),'previously_valid_verdict_changes':valid_changes,
        'original_invalids_now_source_correct':newly_correct,'original_invalids_now_source_wrong':newly_wrong,
        'max_original_logp_absolute_difference':max_logp_diff}

def report(root,original=None):
    import torch
    root=Path(root);output={'protocol':'ab_constrained_v1','status':'exploratory_only','data_qualified':False,'arms':{},'comparisons':{}}
    loaded={}
    for cohort,file,partition in [('pilot','candidate_inputs.jsonl','development'),('controls','controls.jsonl','synthetic')]:
        for proto in ['native_v1','ab_constrained_v1']:
            name=cohort+'_'+proto
            config,rows,check=audit(root/name,root/file,True,partition=partition)
            require(config['output_protocol']==proto,'Wrong protocol metadata')
            for r in rows:
                require(r['output_protocol']==proto,'Wrong row protocol')
                if proto=='ab_constrained_v1':
                    code='A' if r['logp_A']>r['logp_B'] else 'B'
                    if r['logp_A']==r['logp_B']:code=min(config['code_ids'],key=config['code_ids'].get)
                    require(r['output_ids']==[config['code_ids'][code]],'Code differs from constrained raw-logit argmax')
                    require(r['raw_output']==code and r['raw_parsed_text']==code and r['parsed_text']==code,'Raw code mismatch')
                    require(r['generation_budget']==1 and r['stop_reason']=='fixed_code_complete' and not r['length_cap'],'Bad stopping semantics')
            table=metrics.Table(adapt_metrics(rows));stats=quality(rows)
            for v,c in [('B','B'),('R','R+B'),('N','N+B'),('R2','R2+B')]:
                require(abs(metrics.condition_summary(table,c)['false_safe_rate']-stats['per_condition']['mapping_averaged'][v]['false_safe_pct'])<1e-8,'Team FSR mismatch')
            output['arms'][name]={'audit':check,'configuration':config,'quality':stats};loaded[name]=rows
        native=cohort+'_native_v1';constrained=cohort+'_ab_constrained_v1'
        comp=paired(loaded[native],loaded[constrained]);equal=0
        for r,q in zip(loaded[native],loaded[constrained]):
            sa=torch.load(root/native/r['states_file'],weights_only=True,map_location='cpu')
            sb=torch.load(root/constrained/q['states_file'],weights_only=True,map_location='cpu')
            equal+=int(all(torch.equal(sa[k],sb[k]) for k in sa))
        comp['identical_prefill_state_rows']=equal
        output['comparisons'][cohort+'_same_session']=comp
    if original:
        config,rows,check=audit(original,root/'candidate_inputs.jsonl',True)
        output['original_pilot']={'audit':check,'quality':quality(rows)}
        output['comparisons']['original_to_native_rerun']=paired(rows,loaded['pilot_native_v1'])
        output['comparisons']['original_to_constrained']=paired(rows,loaded['pilot_ab_constrained_v1'])
    # Do not publish filesystem paths or data; config contains only aggregate settings.
    return output

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--original');ap.add_argument('--out',required=True)
    args=ap.parse_args();result=report(args.run,args.original)
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=False)
    (dest/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,arm in result['arms'].items():
        q=arm['quality'];print(name,'valid',q['format_valid_count'],'/',q['rows'],'source-label accuracy',round(q['accuracy_all_pct'],2),flush=True)
    print(json.dumps(result['comparisons'],indent=2))
if __name__=='__main__':main()
