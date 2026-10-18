"""Hand-checkable pilot metrics and fail-closed artifact checks."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from pilot_report import adapt_metrics, audit, summarize
from pilot import REVISIONS
from protocol import MODELS, POLICY, messages, sha, variant

class PilotTests(unittest.TestCase):
    def rows(self):
        result=[]
        for sid,unsafe in [('unsafe',True),('safe',False)]:
            for v in ['B','R','N','R2']:
                for m in ['A_safe','A_unsafe']:
                    # Refusal creates one mapping-specific error; R2 has an invalid.
                    verdict=('UNSAFE' if unsafe else 'SAFE')
                    if unsafe and v=='R' and m=='A_safe':verdict='SAFE'
                    if unsafe and v=='R2' and m=='A_safe':verdict='INVALID'
                    code=('bad' if verdict=='INVALID' else ('A' if (verdict=='SAFE')==(m=='A_safe') else 'B'))
                    result.append({'source_id':sid,'cluster_id':sid,'mapped_unsafe':unsafe,
                        'variant':v,'mapping':m,'verdict':verdict,'parsed_text':code,
                        'logp_A':-1.,'logp_B':-2.})
        return result

    def test_mapping_average_invalid_denominator_and_robust_flip(self):
        summary=summarize(self.rows())
        self.assertEqual(summary['mapping_averaged']['R']['false_safe_pct'],50)
        self.assertEqual(summary['cue_effect_pp']['R']['average'],50)
        self.assertEqual(summary['robust_flip_count'],0)
        self.assertEqual(summary['mapping_averaged']['R2']['false_safe_pct'],0)
        self.assertEqual(summary['mapping_averaged']['R2']['unsafe_invalid_pct'],50)

    def test_team_adapter_uses_normalized_text(self):
        rows=self.rows();rows[0]['raw_output']='native wrapper'
        adapted=adapt_metrics(rows)
        self.assertEqual(adapted[0]['raw_output'],rows[0]['parsed_text'])
        self.assertEqual([x['mapping'] for x in adapted[:2]],[1,2])
        self.assertEqual(adapted[2]['condition'],'R+B')

    def artifacts(self,root):
        source=[{'source_id':s,'cluster_id':s,'original_prompt':'Saved request '+s,
            'original_response':'Saved response '+s,'mapped_unsafe':u,
            'study_partition':'development','label_provenance':'human'} for s,u in [('unsafe',True),('safe',False)]]
        inp=root/'input.jsonl';inp.write_text(''.join(json.dumps(r)+'\n' for r in source))
        config={'compute_dtype':'bfloat16','baseline':'native','prompt_format':'native',
            'input_file_sha256':sha(inp.read_text()),'revision':REVISIONS['gemma'],'model_id':MODELS['gemma'], 'policy':POLICY}
        (root/'run_config.json').write_text(json.dumps(config))
        (root/'identity_check.json').write_text(json.dumps({'passed':True,'calls':2}))
        originals={r['source_id']:r for r in source};rows=self.rows()
        for i,r in enumerate(rows):
            r.update(originals[r['source_id']]);r.update(variant(r['original_response'],r['variant']))
            r.update(status='ok',run_index=i,model_revision=REVISIONS['gemma'],raw_parsed_text=r['parsed_text'],
                output_normalization='none',input_tokens=100,seconds=1.,
                ab_mass=math.exp(-1)+math.exp(-2),semantic_margin=1. if r['mapping']=='A_unsafe' else -1.,states_file=f'states_{i:05d}.pt')
            r['prompt_sha256']=sha(json.dumps(messages(r['original_prompt'],r['assistant_response'],r['mapping']),ensure_ascii=False))
            (root/r['states_file']).write_bytes(b'placeholder, tensor contents not audited')
        return inp,rows

    def test_reject_missing_duplicate_changed_label_and_body(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp,rows=self.artifacts(root)
            def write(rr):
                (root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
            write(rows);self.assertEqual(audit(root,inp)[2]['rows'],16)
            for broken in [rows[:-1],rows+[rows[0]]]:
                write(broken)
                with self.assertRaises(ValueError):audit(root,inp)
            for field,value in [('mapped_unsafe',False),('assistant_response','changed'),('verdict','SAFE')]:
                broken=copy.deepcopy(rows);broken[0][field]=value;write(broken)
                with self.assertRaises(ValueError):audit(root,inp)

    def test_reject_internally_consistent_unapproved_revision(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp,rows=self.artifacts(root)
            config=json.loads((root/'run_config.json').read_text());config['revision']='unapproved-resolved-sha'
            (root/'run_config.json').write_text(json.dumps(config))
            for row in rows:row['model_revision']=config['revision']
            (root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'Unapproved pilot checkpoint'):audit(root,inp)

    def test_reject_duplicate_and_extra_activation_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp,rows=self.artifacts(root)
            def write(rr):(root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
            broken=copy.deepcopy(rows);broken[1]['states_file']=broken[0]['states_file'];write(broken)
            with self.assertRaisesRegex(ValueError,'unique row indices'):audit(root,inp)
            write(rows);(root/'states_99999.pt').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError,'extra activation'):audit(root,inp)

if __name__=='__main__':
    unittest.main()
