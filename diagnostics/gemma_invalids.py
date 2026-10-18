"""Replay saved Gemma parser failures; report alternatives without relabeling a run.

Run from the repository: python diagnostics/gemma_invalids.py --run runs/.../gemma
No GPU calls. Output contains aggregate counts and output-format fragments only.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol import normalize_native_output, parse

def redact_patterns(patterns, allowed):
    redacted = Counter()
    for text, count in patterns.items():
        redacted[text if text in allowed else '[unpublished output]'] += count
    return dict(redacted)

def diagnose(root):
    root=Path(root)
    config=json.loads((root/'run_config.json').read_text())
    data=(root/'results.jsonl').read_bytes()
    rows=[json.loads(line) for line in data.decode().splitlines() if line.strip()]
    bad=[r for r in rows if r['verdict']=='INVALID']
    changes=Counter()
    for r in rows:
        actual=parse(normalize_native_output(r['raw_parsed_text'],config['model_id'])[0],r['mapping'])
        if actual!=r['verdict']:
            raise ValueError('Saved verdict and current parser disagree')
        alternative=parse('A' if r['logp_A']>=r['logp_B'] else 'B',r['mapping'])
        if alternative!=actual:changes[actual+'->'+alternative]+=1
    patterns=Counter(r['raw_parsed_text'].strip() for r in bad)
    # Only publish known harmless formatting fragments. Unknown text stays private.
    allowed={"-<|channel>thought",'- The user is',"The user's"}
    return {
        'mode':'offline_saved_output_diagnostic_no_model_calls',
        'results_sha256':hashlib.sha256(data).hexdigest(),
        'total_rows':len(rows),'invalid_rows':len(bad),
        'affected_source_records':len({r['source_id'] for r in bad}),
        'invalids_on_source_safe_records':sum(not r['mapped_unsafe'] for r in bad),
        'output_cap_hits':sum(r['length_cap'] for r in bad),
        'ended_answers':sum(r['end_of_answer'] for r in bad),
        'invalid_input_token_range':[min(r['input_tokens'] for r in bad),max(r['input_tokens'] for r in bad)] if bad else [],
        'patterns':redact_patterns(patterns, allowed),
        'by_condition':dict(Counter(r['variant']+'/'+r['mapping'] for r in bad)),
        'first_token_ab_mass_range':[min(r['ab_mass'] for r in bad),max(r['ab_mass'] for r in bad)] if bad else [],
        'offline_alternative':{'rule':'argmax(logp_A, logp_B) at first generation position; ties choose A',
            'adopted':False,'verdict_changes':dict(changes),
            'warning':'A different scoring rule, not recovered answers or a validated constrained-generation run.'},
        'parser_fix_available_from_saved_answers':False,
        'original_results_modified':False,
    }

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True)
    ap.add_argument('--require-no-invalid',action='store_true',help='Regression probe: exits nonzero when saved outputs reproduce INVALID')
    args=ap.parse_args();report=diagnose(args.run)
    print(json.dumps(report,indent=2))
    if args.require_no_invalid and report['invalid_rows']:
        sys.exit(1)
