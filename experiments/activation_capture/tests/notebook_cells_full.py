"""Executes the notebook's own cells on CPU with Colab/Drive mocked and a small public stand-in
model replacing the four judges (only for plumbing). NOT research results.

Needs inputs/baseline.zip and summaries/jailjudge-full-470-results.zip under the
behavioral refusal screen (or BASELINE_ZIP and FOUR_MODEL_ZIP).
"""
import json, os, sys, types, tempfile, shutil, ast
from pathlib import Path
import nbformat
HERE = Path(__file__).resolve().parents[1]
SCREEN = HERE.parent / "behavioral_refusal_screen"
BASELINE_ZIP = os.environ.get("BASELINE_ZIP", str(SCREEN / "inputs" / "baseline.zip"))
FOUR_MODEL_ZIP = os.environ.get("FOUR_MODEL_ZIP", str(SCREEN / "summaries" / "jailjudge-full-470-results.zip"))
nb = nbformat.read(HERE / "activation_capture.ipynb", 4)
nbformat.validate(nb)
code = [c for c in nb.cells if c.cell_type == 'code']
for i, c in enumerate(code):
    src = c.source.split('\n', 1)[1] if c.source.startswith('%%writefile') else c.source
    compile(src, f'cell{i}', 'exec')
print('nbformat valid;', len(nb.cells), 'cells;', len(code), 'code cells compile')
root = Path(tempfile.mkdtemp()); content = root / 'content'
(content / 'drive/MyDrive/algoverse-jailjudge-inputs').mkdir(parents=True)
(content / 'drive/MyDrive/algoverse-jailjudge-outputs/2026-11-08-run').mkdir(parents=True)
shutil.copy(BASELINE_ZIP, content / 'drive/MyDrive/algoverse-jailjudge-inputs/')
shutil.copy(FOUR_MODEL_ZIP, content / 'drive/MyDrive/algoverse-jailjudge-outputs/2026-11-08-run/')
colab = types.ModuleType('google.colab')
colab.drive = types.SimpleNamespace(mount=lambda p: None)
colab.files = types.SimpleNamespace(upload=lambda: {}, download=lambda p: None)
colab.userdata = types.SimpleNamespace(get=lambda k: 'x')
import torch, transformers, accelerate  # import real deps before mocking
try:
    import google
except ImportError:
    sys.modules['google'] = types.ModuleType('google')
sys.modules['google.colab'] = colab
def src(i): 
    s = code[i].source
    if s.startswith('%%writefile'):
        path, body = s.split('\n', 1); Path(path.split()[1].replace('/content', str(content))).write_text(body); return 'pass'
    return s.replace('/content', str(content))
ns = {'display': print}
exec(src(0), ns)   # cell 1
exec(src(1), ns)   # cell 2 (%%writefile)
sys.path.insert(0, str(content))
exec(src(2), ns)   # cell 3 plan + preflight (full 4-model plan)
assert sum(len(u['source_ids']) for u in ns['PLAN']) == 19740
# --- cell 6 with a stand-in model; shrink plan for CPU
import torch, transformers as tr
torch.cuda.get_device_name = lambda *a: 'cpu-test'
tok = tr.AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
lm = tr.AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct', dtype=torch.float32).eval()
class Fake:
    def __init__(self, name_or_id, revision, *a):
        self.model, self.tokenizer, self.revision = lm, tok, revision
        self.meta = {'model_id': name_or_id, 'revision': revision, 'gpu': 'cpu'}
        import activation_tools as at
        self.layer_path, self.layers = at.find_decoder_layers(lm)
        from refusal_cues.protocol import depths; self.depths = depths(len(self.layers))
        self.codes = {'A': tok.encode('A')[0], 'B': tok.encode('B')[0]}
        self.codes = {'A': tok.encode('A')[0], 'B': tok.encode('B')[0]}
    def encode(self, turns):
        flat = [{'role': t['role'], 'content': t['content'] if isinstance(t['content'], str) else t['content'][0]['text']} for t in turns]
        return tok.apply_chat_template(flat, tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True)
import refusal_cues.inference as inf, refusal_cues.guard_judges as gj
inf.Runner = Fake; gj.GuardRunner = Fake
ns['EXPECTED_DEPTHS'] = {m: Fake('x', 'y').depths for m in ns['EXPECTED_DEPTHS']}
small = ns['RECORDS'][:3]
import activation_tools as at
ns['PLAN'] = at.build_plan(ns['MODELS_TO_RUN'], ['B', 'paper_Rgreet'], small, 3)
cell6 = src(4).replace("judge.decision_search, judge.budget = 16, 256", "judge.decision_search, judge.budget = 16, 4")
exec(cell6, ns)
exec(src(5), ns)   # cell 7 index + agreement
exec(src(6), ns)   # cell 8 README + zips
run = ns['RUN_DIRECTORY']
print(sorted(p.name for p in run.iterdir()))
print(sorted(p.name for p in (run / 'zips').iterdir()))
print((run / 'README.md').read_text())
# rerun capture: everything complete -> no model load
ns['PLAN'] = ns['PLAN']; exec(cell6, ns)
shutil.rmtree(root)
print('NOTEBOOK CELL EXECUTION PASSED (mocked Colab, stand-in model)')
