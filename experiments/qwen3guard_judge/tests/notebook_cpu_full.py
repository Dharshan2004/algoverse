"""Executes the notebook's own cells on CPU with Colab/Drive mocked and Qwen/Qwen3Guard-Gen-0.6B
(fp32) standing in for the 8B checkpoint, on a few records x 2 conditions x 2 repeats. Exercises the
behavioral run (+ resume), parsing, decision-position location, shard writing (+ resume), index,
agreement, metrics, README and zips. NOT research results.
Env: BASELINE_ZIP, FOUR_MODEL_ZIP (jailjudge-full-470-results.zip)."""
import json, os, sys, types, tempfile, shutil
from pathlib import Path
import nbformat
HERE = Path(__file__).resolve().parents[1]
SCREEN = HERE.parent / "behavioral_refusal_screen"
BASELINE_ZIP = os.environ.get('BASELINE_ZIP', str(SCREEN / 'inputs' / 'baseline.zip'))
FOUR_MODEL_ZIP = os.environ.get('FOUR_MODEL_ZIP', str(SCREEN / 'summaries' / 'jailjudge-full-470-results.zip'))
nb = nbformat.read(HERE / 'qwen3guard_experiment.ipynb', 4)
nbformat.validate(nb)
code = [c for c in nb.cells if c.cell_type == 'code']
heads = [c.source.splitlines()[0] for c in code]
print('code cells:', heads)
for i, c in enumerate(code):
    compile(c.source.split('\n', 1)[1] if c.source.startswith('%%writefile') else c.source, f'cell{i}', 'exec')
root = Path(tempfile.mkdtemp()); content = root / 'content'
(content / 'drive/MyDrive/algoverse-jailjudge-inputs').mkdir(parents=True)
(content / 'drive/MyDrive/algoverse-jailjudge-outputs/2026-11-08-run').mkdir(parents=True)
shutil.copy(BASELINE_ZIP, content / 'drive/MyDrive/algoverse-jailjudge-inputs/')
shutil.copy(FOUR_MODEL_ZIP, content / 'drive/MyDrive/algoverse-jailjudge-outputs/2026-11-08-run/')
import torch, transformers, pandas, accelerate  # real deps before mocking google
import transformers.models.qwen3.modeling_qwen3
colab = types.ModuleType('google.colab')
colab.drive = types.SimpleNamespace(mount=lambda p: None, flush_and_unmount=lambda: None)
colab.files = types.SimpleNamespace(upload=lambda: {}, download=lambda p: None)
colab.userdata = types.SimpleNamespace(get=lambda k: None)
try:
    import google
except ImportError:
    sys.modules['google'] = types.ModuleType('google')
sys.modules['google.colab'] = colab
def src(i):
    s = code[i].source
    if s.startswith('%%writefile'):
        path, body = s.split('\n', 1); Path(path.split()[1].replace('/content', str(content))).write_text(body); return 'pass'
    s = s.replace('/content', str(content))
    if i >= 6:   # keep the CPU stand-in patches: no module reloads in the run cells
        for m in ('qg', 'at'):
            s = s.replace(f'importlib.reload({m})', 'None')
    return s
ns = {'display': print}
for i in range(5):                      # 1, 2a, 2b, 2c, 3 (full 470-record preflight)
    exec(src(i), ns)
sys.path.insert(0, str(content))
# cell 4 on CPU: skip pip/GPU checks, keep its get_judge(); 0.6B stand-in, fp32
c4 = src(5); c4 = c4[c4.index('JUDGE = None'):]
import qwen3guard_judge as qg
class CpuRunner(qg.Qwen3GuardRunner):
    def __init__(self, model_id, revision):
        super().__init__('Qwen/Qwen3Guard-Gen-0.6B', qg.REVISIONS['Qwen/Qwen3Guard-Gen-0.6B'], device='cpu', dtype='float32')
qg.Qwen3GuardRunner = CpuRunner
torch.cuda.get_device_name = lambda *a: 'cpu-test'
torch.cuda.memory_allocated = lambda *a: 0
from refusal_cues.protocol import depths
ns.update(torch=torch, EXPECTED_DEPTHS={'qwen3guard': depths(28)})
exec(c4, ns)
# shrink: 4 harmful + 2 safe records, 2 conditions, 2 repeats, shards of 3 records
small = [r for r in ns['RECORDS'] if r['mapped_unsafe']][:4] + [r for r in ns['RECORDS'] if not r['mapped_unsafe']][:2]
ns.update(RECORDS=small, RECORDS_BY_ID={r['source_id']: r for r in small}, CONDITIONS=['B', 'paper_Rgreet'], SHARD_SIZE=3)
import activation_tools as at
ns['PLAN'] = at.build_plan(['qwen3guard'], ns['CONDITIONS'], small, 3)
# cell 6 behavioral: interrupt after 5 judgments, then resume
orig = qg.run_behavioral
qg.run_behavioral = lambda *a, **k: orig(*a, **{**k, 'max_new': 5})
exec(src(6), ns)
qg.run_behavioral = orig
bdir = ns['RUN_DIRECTORY'] / 'behavioral/qwen3guard'
assert [len(p.read_text().splitlines()) for p in qg.chunk_paths(bdir)] == [5], 'interrupt must flush the partial batch'
# legacy file from the pre-chunk notebook (as the in-progress run will have): 3 identical copies + 2 new rows + a truncated line
first = [json.loads(l) for l in qg.chunk_paths(bdir)[0].read_text().splitlines()]
extra_rows = []
orig2 = qg.run_behavioral
probe = Path(tempfile.mkdtemp())
orig2(probe, ns['RECORDS'], ns['CUE_TEXT'], ns['get_judge'], 'probe', ns['CONDITIONS'], ns['REPEATS'], log=lambda m: None, max_new=7)
extra_rows = [r for r in qg.read_verdicts(probe / 'behavioral/qwen3guard').values()][5:7]
shutil.rmtree(probe)
legacy = bdir / 'verdicts.jsonl'
legacy.write_text(''.join(json.dumps(r) + '\n' for r in first[:3] + extra_rows) + '{"source_id": "trunc')
exec(src(6), ns)
rep = {}
assert len(qg.read_verdicts(bdir, rep)) == 24, rep
print('behavioral: interrupted after 5 (flushed), legacy verdicts.jsonl merged (3 identical dupes dropped, 2 rows reused, truncated line ignored), resumed to 24/24;', rep)
exec(src(7), ns)                        # cell 7 results + metrics + combined
beh = pandas.read_csv(ns['RUN_DIRECTORY'] / 'behavioral/qwen3guard_verdicts.csv')
print(beh[['condition', 'repeat', 'harmful', 'raw_label', 'verdict_controversial_unsafe', 'verdict_controversial_safe', 'refusal', 'decision_step', 'logprob_safe', 'logprob_unsafe', 'logprob_controversial']].to_string())
assert (beh.decision_step == 2).all() and beh.format_ok.all()
r0 = beh[beh.repeat == 0].set_index(['source_id', 'condition']).raw_label; r1 = beh[beh.repeat == 1].set_index(['source_id', 'condition']).raw_label
assert (r0 == r1).all(), 'greedy repeats differ'
ar = pandas.read_csv(ns['RUN_DIRECTORY'] / 'behavioral/all_results.csv')
assert list(ar.columns) == qg.ALL_RESULTS_FIELDS and len(ar) == 48 and set(ar.mapping) == set(qg.VIEWS)
comb = pandas.read_csv(ns['RUN_DIRECTORY'] / 'combined/all_results.csv')
assert len(comb) == 39480 + 48 and list(comb.columns) == qg.ALL_RESULTS_FIELDS
# cell 9 capture: interrupt after 2 shards, then resume without rewriting
orig_cap = at.run_capture
at.run_capture = lambda *a, **k: orig_cap(*a, **{**k, 'max_units': 2})
exec(src(8), ns)
at.run_capture = orig_cap
mt = {p: p.stat().st_mtime_ns for p in (ns['RUN_DIRECTORY'] / 'activations').rglob('shard-*.json')}
assert len(mt) == 2
exec(src(8), ns)
assert all(p.stat().st_mtime_ns == t for p, t in mt.items()), 'completed shards rewritten'
try:
    at.acquire_lock(ns['RUN_DIRECTORY'], 'other-runtime'); raise SystemExit('lock not enforced')
except RuntimeError:
    print('capture: interrupted after 2 shards, resumed without rewriting them; lock blocks a second runtime')
exec(src(9), ns)                        # cell 10 index + agreement
S = ns['SUMMARY']
assert S['indexed_prompts'] == 12 and S['index_rows'] == 24 and S['missing_shards'] == 0 and S['errors'] == 0
assert S['raw_label_matches'] == S['raw_label_compared'] == 12 and S['matches'] == 24
idx = ns['INDEX']
q = idx[idx.mapping == 'controversial_unsafe']
assert (q.decision_call == 2).all() and q.decision_token_is_candidate.all() and q.decision_found.all()
assert (q.decision_position == q.prompt_tokens + 1).all()
# decision vector == hidden state of a plain forward pass over prompt + 'Safety:' at the last position
j = ns['get_judge'](); rec = small[0]
turns = at.build_turns('qwen3guard', rec['original_prompt'], at.judge_response(rec['original_response'], ns['CUE_TEXT']['paper_Rgreet']), 'native')
inp = j.encode(turns)
prefix = j.tokenizer.encode('Safety:', add_special_tokens=False)
full = torch.tensor([inp['input_ids'][0].tolist() + prefix])
with torch.inference_mode():
    hs = j.model(input_ids=full, output_hidden_states=True).hidden_states
X, rows, layers = at.load_condition(ns['RUN_DIRECTORY'], 'qwen3guard', 'native', 'paper_Rgreet', 'decision')
F, _, _ = at.load_condition(ns['RUN_DIRECTORY'], 'qwen3guard', 'native', 'paper_Rgreet', 'final_token')
L = inp['input_ids'].shape[-1]
d_dec = max(float((X[0, k].float() - hs[l + 1][0, -1].half().float()).abs().max() / hs[l + 1][0, -1].abs().max()) for k, l in enumerate(layers))
d_fin = max(float((F[0, k].float() - hs[l + 1][0, L - 1].half().float()).abs().max() / hs[l + 1][0, L - 1].abs().max()) for k, l in enumerate(layers))
print(f'decision vector vs forward pass at "Safety:" (rel max diff) {d_dec:.2e}; final_token {d_fin:.2e}; shape {tuple(X.shape)}, layers {layers}')
assert d_dec < 1e-2 and d_fin < 1e-2 and X.shape == (6, 6, 1024)
B, _, _ = at.load_condition(ns['RUN_DIRECTORY'], 'qwen3guard', 'native', 'B', 'cue_mean')
assert float(B.abs().sum()) == 0.0
# identity patch reproduces the label and log-probs
pat = at.qwen3guard_verdict_with_residual_patch(j, turns, layers[2], X[0, 2].float(), int(rows[0]['decision_call']))
print('identity patch:', pat, '| captured', rows[0]['verdict'], rows[0]['logprob_safe'], rows[0]['logprob_unsafe'])
assert pat['raw_label'] == rows[0]['verdict'] and abs(pat['logprob_unsafe'] - rows[0]['logprob_unsafe']) < 0.05
exec(src(10), ns)                       # cell 11 verify + README + zips
run = ns['RUN_DIRECTORY']
assert ns['COMPLETE']
import zipfile
for z in sorted((run / 'zips').iterdir()):
    with zipfile.ZipFile(z) as f:
        assert f.testzip() is None
        print('zip', z.name, z.stat().st_size, 'bytes', len(f.namelist()), 'entries')
print(sorted(p.name for p in run.iterdir()))
print((run / 'analysis/qwen3guard/analysis.md').read_text()[:1500])
# rerun of cell 6 / cell 9 after completion: no work, no model load needed
ns['JUDGE'] = None
exec(src(6), ns); exec(src(8), ns)
assert ns['JUDGE'] is None, 'completed run should not reload the model'
shutil.rmtree(root)
print('NOTEBOOK CPU SMOKE PASSED (Qwen3Guard-Gen-0.6B stand-in; not research results)')
