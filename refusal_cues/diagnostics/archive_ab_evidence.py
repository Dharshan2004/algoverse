"""Verify preserved artifacts and create a sanitized executed notebook, without rerunning cells."""
import copy,hashlib,json,subprocess
from pathlib import Path

from refusal_cues.paths import CONTROLS_AB_V1, REPO_ROOT

ROOT = REPO_ROOT
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 r=ROOT/'runs/gemma-ab-v1-2026-10-18'; out=ROOT/'reports/gemma-ab-v1-2026-10-18'
 old=ROOT/'runs/pilot-2026-10-17';guard=ROOT/'runs/ab-v1-preparation/original_fingerprints.json'
 fingerprints=json.loads(guard.read_text())
 assert all(sha(old/k)==v for k,v in fingerprints.items())
 manifest=json.loads((r/'manifest.json').read_text())
 assert manifest['base_code_commit']=='03b32cb2ac6ae6aabf67a6b2da7632c3de852834'
 for name,h in manifest['source_sha256'].items():
  assert sha(r/'source'/name)==h
  assert hashlib.sha256(subprocess.check_output(['git','show','20450d8:'+name],cwd=ROOT)).hexdigest()==h
 assert sha(r/'candidate_inputs.jsonl')==sha(old/'candidate_inputs.jsonl')
 assert sha(r/'controls.jsonl')==sha(CONTROLS_AB_V1)
 summary=json.loads((out/'summary.json').read_text());remote=json.loads((r/'audit/summary.json').read_text())
 assert summary['arms']==remote['arms']
 assert all(summary['comparisons'][k]==v for k,v in remote['comparisons'].items())
 original=r/'executed-colab.ipynb';n=json.loads(original.read_text());clean=json.loads((ROOT/'notebooks/gemma_ab_v1.ipynb').read_text())
 code=lambda nb:[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
 assert code(n)==code(clean)
 counts=[c['execution_count'] for c in n['cells'] if c['cell_type']=='code'];assert counts==[1,2,3,4]
 assert not any(o['output_type']=='error' for c in n['cells'] for o in c.get('outputs',[]))
 stdout=''.join(''.join(o['text']) for o in n['cells'][-1]['outputs'] if o['output_type']=='stream')
 assert json.loads(stdout[stdout.index('{'):])==remote['comparisons']
 for name,a in remote['arms'].items():
  q=a['quality'];assert f"{name} : valid {q['format_valid_count']} / {q['rows']} ; source-label accuracy {round(q['accuracy_all_pct'],2)} %" in stdout
 snapshot=copy.deepcopy(n);snapshot['metadata']={k:v for k,v in n['metadata'].items() if k in ('kernelspec','language_info')}
 for i,c in enumerate(snapshot['cells']):
  c['metadata']={}
  if c['cell_type']=='code':c['outputs']=[o for o in c['outputs'] if o['output_type']=='stream'] if i in (1,3,5) else []
 header={'cell_type':'markdown','metadata':{},'source':['# Sanitized executed Gemma A/B protocol comparison\n','Executed in Chrome/Colab on NVIDIA L4, 18 October 2026.\n','Code cells and execution counts are unchanged from the downloaded notebook. Runtime metadata, upload/download HTML and verbose inference logs were removed; original setup and final aggregate stdout retained verbatim. No cell has been rerun or output reconstructed in this snapshot.\n',f'Exact original SHA256: `{sha(original)}`.\n','[Full audited comparison](../../../reports/gemma-ab-v1-2026-10-18/README.md). Raw notebook and all results remain in ignored `runs/gemma-ab-v1-2026-10-18/`.\n']}
 snapshot['cells'].insert(0,header)
 hist=ROOT/'notebooks/history/2026-10-18';hist.mkdir(exist_ok=True)
 dest=hist/'gemma_ab_v1.executed.sanitized.ipynb';save(dest,snapshot)
 assert code(snapshot)==code(n)
 provenance={'status':'complete_audited','base_commit':manifest['base_code_commit'],'implementation_commit':'20450d80a88942388342cc7440f7b58d9ea36a54','colab_url':'https://colab.research.google.com/drive/1ziOhZiFOw0JgOqBI_CJFYNpkGHQwf5ww','original_pilot_files_unchanged':len(fingerprints),'original_fingerprints_sha256':sha(guard),'original_gemma_results_sha256':sha(old/'gemma/results.jsonl'),'original_invalids_preserved':10,'source_sha256':manifest['source_sha256'],'judgments':928,'identity_calls':8,'all_four_arms_and_original_audited':True,'local_and_remote_arm_audits_identical':True,'notebook_code_matches_frozen_launcher':True,'executed_notebook_sha256':sha(original),'sanitized_notebook_sha256':sha(dest),'execution_counts':counts,'notebook_errors':0,'archive_sha256':sha(r.parent/'gemma-ab-v1-2026-10-18.zip'),'summary_sha256':sha(out/'summary.json'),'manifest_note':'Raw manifest records inference_complete_audit_pending before notebook audit. Remote audit and independent local audit both subsequently passed; raw manifest preserved unchanged. Arm config run_purpose=protocol_smoke is a legacy Runner field; actual cohort and coverage are defined by this manifest and audited input metadata.'}
 save(out/'provenance.json',provenance)
 save(hist/'provenance.json',{'original_sha256':sha(original),'snapshot_sha256':sha(dest),'execution_counts':counts,'code_cells_unchanged':True,'sanitization':['Removed runtime and cell metadata','Removed upload and download display_data HTML','Removed verbose inference cell stdout','Retained original setup, input verification and final aggregate stdout','Added explicit provenance Markdown header'],'report':'../../../reports/gemma-ab-v1-2026-10-18/README.md'})
 print('Original files unchanged:',len(fingerprints),'Executed notebook verified; sanitized snapshot saved')
if __name__=='__main__':main()
