"""Exercise Sabika's file pipeline against source-preservation edge cases."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from refusal_cues.protocol import validate_records, variant

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

class PreviewIntegrationTests(unittest.TestCase):
    def run_script(self,name,cwd,*args,ok=True):
        result=subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=cwd, capture_output=True, text=True)
        if ok:self.assertEqual(result.returncode,0,result.stderr)
        else:self.assertNotEqual(result.returncode,0,result.stdout)
        return result

    def test_source_text_survives_all_four_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            pairs=[('NA','00123',True),('null','  response\nwith "quotes", and a comma  ',False),('request','NA',True)]
            raw=[dict(user_prompt=a,model_response=b,is_jailbroken=c,
                hazard_cate_llamaguard3='test',prompt_complexity_cate='test',language='en') for a,b,c in pairs]
            # Truthy strings are not silently accepted as source boolean labels.
            raw.append(dict(raw[0],user_prompt='exclude nonboolean',is_jailbroken='False'))
            (p/'JAILJUDGE_ID.json').write_text(json.dumps(raw))
            for name in ['prepare_preview.py','qualify_preview.py']:
                self.run_script(name,p)
            self.run_script('convert_preview.py',p,ok=False)
            self.assertFalse((p/'data/preview/candidate_inputs.jsonl').exists())
            self.run_script('convert_preview.py',p,'--confirm-human-labels')
            self.run_script('create_variants.py',p)
            rows=[json.loads(x) for x in (p/'data/preview/candidate_inputs.jsonl').read_text().splitlines()]
            validate_records(rows)
            self.assertEqual({(r['original_prompt'],r['original_response'],r['mapped_unsafe']) for r in rows},set(pairs))
            variants=[json.loads(x) for x in (p/'data/preview/variants.jsonl').read_text().splitlines()]
            self.assertEqual(len(variants),12)
            for row in variants:
                self.assertEqual(row['variant_response'],variant(row['original_response'],row['variant_id'].split('+')[0])['assistant_response'])

    def test_csv_rejects_ambiguous_label_and_blank_response(self):
        from refusal_cues.preview_io import read_preview
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'preview.csv'
            for label,body in [('unknown','valid'),('False','')]:
                p.write_text(f'user_prompt,model_response,is_jailbroken\nrequest,{body},{label}\n')
                with self.assertRaises(ValueError):read_preview(p)
