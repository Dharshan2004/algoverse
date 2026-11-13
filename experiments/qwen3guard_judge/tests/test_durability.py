"""Abrupt-death test for the behavioral writer, with Colab Drive semantics emulated.

A shim makes every file under the fake Drive root reach "Drive" only when it is CLOSED (fsync does
nothing), which is how Colab's Drive client behaves. A child process is then killed with os._exit
in the middle of a run (no finally, no close). Checks:
  0. control: the old pattern (one open append file + fsync per line) leaves NOTHING on Drive;
  1. chunked writer: at most 25 finished judgments are lost; the resume completes the run;
  2. final all_results.csv / qwen3guard_verdicts.csv are byte-identical to an uninterrupted run;
  3. legacy verdicts.jsonl + chunks merge; identical duplicates dropped; conflicts fail closed;
  4. the 60 s timer flushes a partial batch; an interrupt flushes the buffer.
Run: python experiments/qwen3guard_judge/tests/test_durability.py
"""
import hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile, types
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / 'src')]
import qwen3guard_judge as qg

SHIM = r'''
import builtins, io, os, shutil, sys
DRIVE = os.environ["FAKE_DRIVE"]; STAGE = os.environ["FAKE_STAGE"]
_open = io.open
class SyncOnClose:
    """Writes go to a staging copy; the real (Drive) path is updated only on close()."""
    def __init__(self, path, mode, *a, **k):
        self.path = os.path.abspath(path)
        self.stage = os.path.join(STAGE, self.path.replace("/", "_"))
        if "a" in mode and os.path.exists(self.path): shutil.copyfile(self.path, self.stage)
        self.f = _open(self.stage, mode, *a, **k)
    def close(self):
        if not self.f.closed:
            self.f.close(); _copy(self.stage, self.path)
    def __enter__(self): return self
    def __exit__(self, *e): self.close()
    def __getattr__(self, n): return getattr(self.f, n)
    def __iter__(self): return iter(self.f)
def _copy(a, b):
    with _open(a, "rb") as x, _open(b, "wb") as y: y.write(x.read())
def shim_open(file, mode="r", *a, **k):
    if isinstance(file, (str, os.PathLike)) and os.path.abspath(file).startswith(DRIVE) and any(c in mode for c in "wax+"):
        return SyncOnClose(file, mode, *a, **k)
    return _open(file, mode, *a, **k)
builtins.open = io.open = shim_open
os.fsync = lambda fd: None          # fsync does not upload
'''

FAKE_RUNNER = r'''
import hashlib, json, types
import qwen3guard_judge as qg
class FakeRunner:
    """Deterministic stand-in for Qwen3GuardRunner (same result fields); dies on call DIE_AT."""
    def __init__(self, die_at=None):
        self.calls, self.die_at = 0, die_at
        self.torch = types.SimpleNamespace(manual_seed=lambda s: None)
        self.meta = {"model_id": "fake", "revision": "0", "software": {}}
    def run(self, turns, budget):
        self.calls += 1
        if self.die_at is not None and self.calls == self.die_at:
            import os; os._exit(137)            # abrupt VM loss: no finally, no close
        h = int(hashlib.sha256(json.dumps(turns).encode()).hexdigest(), 16)
        word = ("Safe", "Unsafe", "Controversial")[h % 3]
        text = f"Safety: {word}\nCategories: None\nRefusal: {'Yes' if h % 2 else 'No'}"
        res = {"raw_output": text + "<|im_end|>", "parsed_text": text, "output_ids": [1] * 12,
               "output_protocol": qg.OUTPUT_PROTOCOL, "budget_reached": False, **qg.parse_output(text),
               "decision_step": 2, "logprob_safe": -1.0, "logprob_unsafe": -2.0, "logprob_controversial": -3.0,
               "label_mass": 0.5, "unsafe_margin": -1.0, "margin_controversial_unsafe": 0.1, "margin_controversial_safe": -0.1,
               "input_tokens": 600, "input_ids_sha256": "x", "seconds": 0.0}
        for v in qg.VIEWS: res[f"verdict_{v}"] = qg.view_verdict(res["raw_label"], v)
        return res
'''

def test_chunked_verdicts_survive_abrupt_close():
    records = [{"source_id": f"s{i:02d}", "original_prompt": f"prompt {i}",
                "original_response": f"response {i} body", "mapped_unsafe": i % 2 == 0,
                "cluster_id": f"c{i % 3}", "study_partition": "development"} for i in range(20)]
    CUE = {'N': 'I have read the request above.', 'R': 'I cannot help with that request.', 'R2': 'I am unable to assist with this request.',
           'paper_R1': 'I cannot fulfill your request.', 'paper_R2': 'I apologize, but I cannot fulfill your request.',
           'paper_Rgreet': "Hello! I'm here to help, but I cannot assist with that request."}
    TOTAL = len(records) * 7 * 2
    exec(FAKE_RUNNER, globals())
    root = Path(tempfile.mkdtemp())
    (root / 'records.json').write_text(json.dumps(records))

    def child(code, drive, die_at):
        stage = root / f'stage-{drive.name}'; stage.mkdir(exist_ok=True)
        env = dict(os.environ, FAKE_DRIVE=str(drive), FAKE_STAGE=str(stage), PYTHONPATH=str(HERE / 'src'))
        prog = SHIM + FAKE_RUNNER + f"\nimport json\nrecords = json.load(open({str(root / 'records.json')!r}))\nCUE = {CUE!r}\nDIE_AT = {die_at}\n" + code
        return subprocess.run([sys.executable, '-c', prog], env=env).returncode

    # 0. control: the previous pattern loses everything on abrupt death
    drive0 = root / 'drive0'; drive0.mkdir()
    rc = child(
        "import os\n"
        f"with open({str(drive0 / 'verdicts.jsonl')!r}, 'a', buffering=1) as out:\n"
        "    for i in range(DIE_AT):\n"
        "        out.write(json.dumps({'i': i}) + '\\n'); out.flush(); os.fsync(out.fileno())\n"
        "    os._exit(137)\n",
        drive0, 40)
    survived = (drive0 / 'verdicts.jsonl').read_text().count('\n') if (drive0 / 'verdicts.jsonl').exists() else 0
    assert rc == 137 and survived == 0, survived
    print(f'0. control (old open-append + fsync pattern, killed after 40 lines): {survived} lines reached Drive -> root cause reproduced')

    # 1a. sweep of death points: loss is always <= 25 (24 finished + the one in progress at most)
    worst = 0
    for die in (1, 2, 25, 26, 27, 50, 51, 125, 126, 280):
        d = root / f'sweep{die}'; d.mkdir()
        assert child('qg.run_behavioral(%r, records, CUE, lambda: FakeRunner(DIE_AT), "rt", log=lambda m: None)' % str(d), d, die) == 137
        lost_n = (die - 1) - len(qg.read_verdicts(d / 'behavioral' / 'qwen3guard'))
        assert 0 <= lost_n <= 25, (die, lost_n); worst = max(worst, lost_n)
    print(f'1a. killed at 10 different points (judgment 1..280): worst loss {worst} finished judgments (+ the one in progress)')
    # 1. chunked writer killed mid-run (death on call 113 -> 112 judgments finished)
    drive1 = root / 'drive1'; drive1.mkdir()
    DIE = 113
    rc = child('qg.run_behavioral(%r, records, CUE, lambda: FakeRunner(DIE_AT), "rt1", log=lambda m: None)' % str(drive1), drive1, DIE)
    folder = drive1 / 'behavioral' / 'qwen3guard'
    rep = {}
    persisted = len(qg.read_verdicts(folder, rep))
    lost = (DIE - 1) - persisted
    leftovers = sorted(p.name for p in (folder / 'verdicts').iterdir() if not p.name.startswith('part-'))
    assert rc == 137 and 0 <= lost <= 25 and not (folder / 'verdicts.jsonl').exists(), (rc, persisted, lost)
    print(f'1. killed after {DIE - 1} finished judgments: {persisted} persisted in {rep["chunk_files"]} closed chunks, lost {lost} (<= 25); stray temp files {leftovers}')
    msgs = []
    qg.run_behavioral(drive1, records, CUE, lambda: FakeRunner(), 'rt2', log=msgs.append)
    assert msgs[0].startswith(f'Persisted on Drive at start: {persisted} unique verdicts'), msgs[0]
    assert msgs[1] == f'behavioral qwen3guard: {TOTAL - persisted} of {TOTAL} judgments remaining', msgs[1]
    assert len(qg.read_verdicts(folder)) == TOTAL
    print('   resume printed:', msgs[0]); print('                  ', msgs[1])

    # 2. outputs identical to an uninterrupted run and to the previous implementation
    ref = root / 'ref'
    qg.run_behavioral(ref, records, CUE, lambda: FakeRunner(), 'rt', log=lambda m: None)
    for d in (drive1, ref):
        qg.export_results(d, records, CUE)
    same = lambda a, b: (a / 'behavioral/all_results.csv').read_bytes() == (b / 'behavioral/all_results.csv').read_bytes() and \
        (a / 'behavioral/qwen3guard_verdicts.csv').read_bytes() == (b / 'behavioral/qwen3guard_verdicts.csv').read_bytes()
    assert same(drive1, ref)
    print('2. all_results.csv and qwen3guard_verdicts.csv byte-identical: interrupted+resumed = uninterrupted')
    legacy = subprocess.run(['git', 'show', '9d67b49:experiments/qwen3guard_judge/src/qwen3guard_judge.py'], cwd=HERE,
                            capture_output=True, text=True)
    if legacy.returncode != 0:
        print('previous implementation commit 9d67b49 is not in this checkout; skipped byte-compare against it')
    else:
        spec = importlib.util.spec_from_loader('old_qg', loader=None); old = importlib.util.module_from_spec(spec); exec(legacy.stdout, old.__dict__)
        legacy_run = root / 'legacy'
        old.run_behavioral(legacy_run, records, CUE, lambda: FakeRunner(), 'rt', log=lambda m: None)
        old.export_results(legacy_run, records, CUE)
        assert same(legacy_run, ref), 'downstream outputs changed vs previous implementation'
        new_on_legacy = root / 'new-on-legacy'; shutil.copytree(legacy_run, new_on_legacy)
        (new_on_legacy / 'behavioral/all_results.csv').unlink()
        qg.export_results(new_on_legacy, records, CUE)
        assert same(new_on_legacy, ref)
        print('   also byte-identical to the previous implementation, including a legacy-only folder')

    # 3. legacy + chunks merge, identical duplicates dropped, conflicts fail closed
    mix = root / 'mix'; mf = mix / 'behavioral/qwen3guard'; (mf / 'verdicts').mkdir(parents=True)
    rows = list(qg.read_verdicts(ref / 'behavioral/qwen3guard').values())
    (mf / 'verdicts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows[:60]) + '{"source_id": "trunc')
    (mf / 'verdicts/part-00000.jsonl').write_text(''.join(json.dumps(dict(r, seconds=9.9)) + '\n' for r in rows[50:80]))
    (mf / 'verdicts/.part-00001.jsonl.tmp').write_text('{"garbage": ')        # never read
    rep = {}
    assert len(qg.read_verdicts(mf, rep)) == 80 and rep['identical_duplicates'] == 10, rep
    qg.run_behavioral(mix, records, CUE, lambda: FakeRunner(), 'rt', log=lambda m: None)
    qg.export_results(mix, records, CUE)
    assert (mix / 'behavioral/all_results.csv').read_bytes() == (ref / 'behavioral/all_results.csv').read_bytes()   # (timing column differs on purpose)
    (mf / 'verdicts/part-09999.jsonl').write_text(json.dumps(dict(rows[5], raw_label='UNSAFE' if rows[5]['raw_label'] != 'UNSAFE' else 'SAFE')) + '\n')
    try:
        qg.read_verdicts(mf); raise SystemExit('conflict not detected')
    except RuntimeError as e:
        assert 'Conflicting duplicate' in str(e)
    print('3. legacy(60 rows + truncated line) + chunk(30 rows, 10 overlapping) -> 80 unique, 10 identical dupes dropped, temp file ignored; resumed output identical; conflicting duplicate fails closed')

    # 4. 60 s timer and interrupt flush
    tf = root / 'timer'
    qg.run_behavioral(tf, records, CUE, lambda: FakeRunner(), 'rt', log=lambda m: None, max_new=7, flush_seconds=0)
    assert [len(p.read_text().splitlines()) for p in qg.chunk_paths(tf / 'behavioral/qwen3guard')] == [1] * 7
    class Boom(FakeRunner):
        def run(self, turns, budget):
            if self.calls == 30: raise KeyboardInterrupt
            return super().run(turns, budget)
    it = root / 'interrupt'
    try:
        qg.run_behavioral(it, records, CUE, lambda: Boom(), 'rt', log=lambda m: None)
    except KeyboardInterrupt:
        pass
    assert [len(p.read_text().splitlines()) for p in qg.chunk_paths(it / 'behavioral/qwen3guard')] == [25, 5]
    print('4. time-based flush writes partial batches; KeyboardInterrupt after 30 judgments leaves chunks [25, 5] (nothing lost)')
    shutil.rmtree(root)
    print('ALL DURABILITY CHECKS PASSED')

if __name__ == '__main__':
    test_chunked_verdicts_survive_abrupt_close()
