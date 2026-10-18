"""Recreate the private Colab upload bundle for the frozen ab_constrained_v1 run."""
import argparse, ast, hashlib, json, zipfile, subprocess
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();repo=Path(__file__).resolve().parent
    notebook=json.loads((repo/'notebooks/gemma_ab_v1.ipynb').read_text())
    expected=None
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPECTED' for t in node.targets):
                expected=ast.literal_eval(node.value)
    if not expected:raise ValueError('Frozen overlay manifest missing')
    source=Path(args.input)
    if hashlib.sha256(source.read_bytes()).hexdigest()!='ae4f5d179571f290910c688ea59e68dae3110a74eebadd1810bfcaf91deda311':
        raise ValueError('Input differs from the original pilot; create a separate version')
    frozen = {name: subprocess.check_output(['git','show','20450d80a88942388342cc7440f7b58d9ea36a54:'+name],cwd=repo) for name in expected}
    for name,digest in expected.items():
        if hashlib.sha256(frozen[name]).hexdigest()!=digest:
            raise ValueError('Code differs from frozen experiment: '+name)
    dest=Path(args.out)
    if dest.suffix!='.zip':raise ValueError('Use a private .zip artifact')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for name in expected:z.writestr('overlay/'+name,frozen[name])
        z.write(source,'candidate_inputs.jsonl')
    print('Private bundle ready:',dest)

if __name__=='__main__':main()
