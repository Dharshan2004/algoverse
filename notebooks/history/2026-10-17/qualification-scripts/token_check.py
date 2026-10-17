import os,json,sys,pathlib,importlib.metadata
root=pathlib.Path(__file__).resolve().parent
os.environ['HF_HOME']=str(root/'hf-cache')
sys.path.insert(0,str(root.parent.parent/'repo'))
import transformers as tr
from protocol import MODELS,messages,variant
records=[json.loads(l) for l in (root/'candidate_inputs.jsonl').read_text().splitlines()]
report={'scope':'All four variants, both code mappings, native/content/strict prompts, code and literal labels; no weights or inference','models':{},'software':{p:importlib.metadata.version(p) for p in ['transformers','mistral-common','huggingface-hub','tokenizers']}}
for name,model in MODELS.items():
 meta=json.loads((root/(name+'_metadata.json')).read_text()); revision=meta['sha']
 try:
  proc=(tr.MistralCommonBackend.from_pretrained(model,revision=revision) if name=='ministral' else tr.AutoProcessor.from_pretrained(model,revision=revision,trust_remote_code=False))
  rows=[]
  for r in records:
   lengths=[]
   for v in ['B','R','N','R2']:
    for m in ['A_safe','A_unsafe']:
     for b in ['native','content','strict']:
      for literal in [False,True]:
       turns=messages(r['original_prompt'],variant(r['original_response'],v)['assistant_response'],mapping=m,baseline=b,literal=literal)
       d=proc.apply_chat_template(turns,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=True)
       ids=d['input_ids']; n=int(ids.shape[-1])
       lengths.append({'variant':v,'mapping':m,'baseline':b,'literal':literal,'tokens':n})
   rows.append({'source_id':r['source_id'],'max_tokens':max(x['tokens'] for x in lengths),'within_4096':all(x['tokens']<=4096 for x in lengths),'conditions':lengths})
  report['models'][name]={'revision':revision,'backend':type(proc).__name__,'status':'pass' if all(x['within_4096'] for x in rows) else 'overlength','rows':rows}
  print(name,'PASS',min(x['max_tokens'] for x in rows),max(x['max_tokens'] for x in rows),flush=True)
 except Exception as e:
  report['models'][name]={'revision':revision,'status':'blocked','error':type(e).__name__+': '+str(e)};print(name,report['models'][name],flush=True)
 (root/'token_checks.json').write_text(json.dumps(report,indent=2))
