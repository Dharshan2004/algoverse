"""Reproduce case-level evidence and explicitly post-hoc sensitivity; never edits runs."""
import argparse,hashlib,json
from pathlib import Path

def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
 from refusal_cues.paths import REPO_ROOT
 root=REPO_ROOT
 old=root/'runs/pilot-2026-10-17';new=root/'runs/gemma-ab-v1-2026-10-18'
 data=read(old/'candidate_inputs.jsonl');native=read(old/'gemma/results.jsonl');constrained=read(new/'pilot_ab_constrained_v1/results.jsonl')
 idx={(r['source_id'],r['variant'],r['mapping']):r for r in native}
 assert len(idx)==400
 flips=[(i,d) for i,d in enumerate(data,1) if d['mapped_unsafe'] and all(idx[d['source_id'],'B',m]['verdict']=='UNSAFE' and idx[d['source_id'],'R',m]['verdict']=='SAFE' for m in ('A_safe','A_unsafe'))]
 assert [i for i,_ in flips]==[25,47]
 result={'scope':'Exploratory, post-outcome case review; not causal evidence or corrected primary results','input_sha256':digest(old/'candidate_inputs.jsonl'),'files_sha256':{str(p.relative_to(root)):digest(p) for p in [old/'gemma/results.jsonl',new/'pilot_native_v1/results.jsonl',new/'pilot_ab_constrained_v1/results.jsonl']},'cases':[],'posthoc_sensitivity':{}}
 excluded={d['source_id'] for _,d in flips}
 for i,d in flips:
  case={'row':i,'source_id':d['source_id'],'source_label':'UNSAFE','prompt_sha256':hashlib.sha256(d['original_prompt'].encode()).hexdigest(),'response_sha256':hashlib.sha256(d['original_response'].encode()).hexdigest(),'judgments':{}}
  for name,rows in [('native',native),('constrained',constrained)]:
   case['judgments'][name]=[{k:r[k] for k in ('variant','mapping','verdict','raw_output','output_ids','logp_A','logp_B','ab_mass','semantic_margin')} for r in rows if r['source_id']==d['source_id']]
  result['cases'].append(case)
 for name,rows in [('native',native),('constrained',constrained)]:
  strata={}
  for label,omit in [('all_25_source_unsafe',set()),('set_aside_two_review_cases',excluded)]:
   counts={}
   for v in ['B','R','N','R2']:
    rs=[r for r in rows if r['mapped_unsafe'] and r['source_id'] not in omit and r['variant']==v]
    counts[v]={'false_safe':sum(r['verdict']=='SAFE' for r in rs),'judgments':len(rs),'source_records':len(rs)//2,'rate_pct':100*sum(r['verdict']=='SAFE' for r in rs)/len(rs)}
   strata[label]={'conditions':counts,'R_minus_B_pp':counts['R']['rate_pct']-counts['B']['rate_pct']}
  result['posthoc_sensitivity'][name]=strata
 fingerprints=json.loads((root/'runs/ab-v1-preparation/original_fingerprints.json').read_text())
 assert all(digest(old/k)==h for k,h in fingerprints.items())
 result['original_files_preserved']=len(fingerprints)
 result['source_label_reconciliation']='Pending original JAILJUDGE_ID.json; reattached candidate file is byte-identical, not independent source evidence'
 p=Path(args.out)
 with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('Case evidence saved; original files unchanged:',len(fingerprints))
if __name__=='__main__':main()
