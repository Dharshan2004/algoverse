"""Gemma native/constrained comparison based on PR #3, commit 03b32cb."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
from refusal_cues.inference import Runner
from refusal_cues.paths import CONTROLS_AB_V1, PACKAGE_DIR
from refusal_cues.protocol import MODELS, MAPPINGS, POLICY, messages, parse, sha, validate_records, variant
BASE='03b32cb2ac6ae6aabf67a6b2da7632c3de852834'
REVISION='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'

def run_arm(runner, inp, out, protocol):
    out.mkdir()
    records=validate_records([json.loads(s) for s in inp.read_text().splitlines() if s.strip()])
    plans=[(r,variant(r['original_response'],v),m) for r in records for v in ['B','R','N','R2'] for m in MAPPINGS]
    turns=[messages(r['original_prompt'],v['assistant_response'],m) for r,v,m in plans]
    for prompt in turns:runner.encode(prompt)
    config={**runner.meta,'baseline':'native','prompt_format':'native','policy':POLICY,
        'input_file_sha256':sha(inp.read_text()),'output_protocol':protocol,
        'generation_budget':1 if protocol=='ab_constrained_v1' else 4,'base_code_commit':BASE,
        'scores':'original full-vocabulary first-position logits; not renormalized over A/B'}
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    native,ns=runner.run(turns[0],output_protocol=protocol)
    identity,is_=runner.run(turns[0],output_protocol=protocol,edit={'layer':runner.depths[0],'mode':'identity'})
    assert native['output_ids']==identity['output_ids']
    assert all(abs(native[k]-identity[k])<=1e-5 for k in ['logp_A','logp_B'])
    assert all(runner.torch.equal(ns[k],is_[k]) for k in ns)
    (out/'identity_check.json').write_text(json.dumps({'passed':True,'calls':2}))
    with (out/'results.jsonl').open('w') as f:
        for i,((r,v,m),prompt) in enumerate(zip(plans,turns)):
            result,states=runner.run(prompt,output_protocol=protocol)
            state_file=f'states_{i:05d}.pt';runner.torch.save(states,out/state_file)
            row={**r,**v,**result,'mapping':m,'verdict':parse(result['parsed_text'],m),
                'semantic_margin':(1 if m=='A_unsafe' else -1)*(result['logp_A']-result['logp_B']),
                'model_revision':runner.revision,'states_file':state_file,
                'prompt_sha256':sha(json.dumps(prompt,ensure_ascii=False)),'run_index':i}
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            if (i+1)%20==0 or i+1==len(plans):print(out.name,i+1,'/',len(plans),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.input,root/'candidate_inputs.jsonl');shutil.copy2(CONTROLS_AB_V1,root/'controls.jsonl')
    source_dir=root/'source';source_dir.mkdir()
    for name in ['inference.py','protocol.py','ab_constraint.py','ab_experiment.py','ab_report.py','pilot_report.py','metrics.py']:
        shutil.copy2(PACKAGE_DIR/name,source_dir/name)
    manifest={'base_code_commit':BASE,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.iterdir()},'planned_judgments':928,'planned_identity_calls':8,'status':'started'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (root/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    runner=Runner(MODELS['gemma'],REVISION,'bfloat16')
    for cohort,inp in [('pilot',root/'candidate_inputs.jsonl'),('controls',root/'controls.jsonl')]:
        for protocol in ['native_v1','ab_constrained_v1']:
            run_arm(runner,inp,root/(cohort+'_'+protocol),protocol)
    manifest['status']='inference_complete_audit_pending';(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('ALL FOUR ARMS COMPLETE: 928 judgments + 8 identity calls',flush=True)
if __name__=='__main__':main()
