"""Offline qualification checks; flags are review candidates, not replacement labels."""
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INPUT = ROOT / 'tmp/sample-qualification-2026-10-17/candidate_inputs.jsonl'
sys.path.insert(0, str(ROOT / 'repo'))
from protocol import validate_records, variant, messages, sha

def normalize(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())

def grams(s):
    words = re.findall(r'\w+', normalize(s))
    return set(zip(*(words[i:] for i in range(5))))

def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0

def boundary_flag(s):
    # This heuristic intentionally does not assert that a source was truncated.
    s = re.sub(r'</s>\s*$', '', s).rstrip()
    return bool(s) and s[-1] not in '.!?。！？”"»'

def run():
    raw = INPUT.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    validate_records(rows)
    record_audit = []
    variant_checks = 0
    for i, r in enumerate(rows, 1):
        calculated = sha(normalize(r['original_prompt']) + '\n' + sha(r['original_response']))[:16]
        flags = []
        for field in ['original_prompt', 'original_response']:
            if boundary_flag(r[field]):
                flags.append(field + '_boundary_review')
            if '\ufffd' in r[field]:
                flags.append(field + '_replacement_character')
        if '</s>' in r['original_response']:
            flags.append('literal_eos_marker')
        if calculated != r['source_id']:
            flags.append('source_id_mismatch')
        for v in ['B', 'R', 'N', 'R2']:
            data = variant(r['original_response'], v)
            body = data['assistant_response'].encode()[data['body_byte_start']:data['body_byte_end']]
            assert body == r['original_response'].encode()
            assert data['body_sha256'] == sha(r['original_response'])
            for mapping in ['A_safe','A_unsafe']:
                prompt = messages(r['original_prompt'], data['assistant_response'], mapping)
                other = dict(r, mapped_unsafe=not r['mapped_unsafe'])
                assert prompt == messages(other['original_prompt'], data['assistant_response'], mapping)
            variant_checks += 1
        record_audit.append({'row': i, 'source_id': r['source_id'], 'mapped_unsafe': r['mapped_unsafe'],
            'prompt_chars': len(r['original_prompt']), 'response_chars': len(r['original_response']),
            'prompt_sha256': sha(r['original_prompt']), 'response_sha256': sha(r['original_response']),
            'flags': flags, 'source_match': 'pending_authenticated_original',
            'human_review': 'pending', 'qualified_for_scientific_pilot': False})
    pairs = []
    gs = [grams(r['original_prompt']) for r in rows]
    for i,j in itertools.combinations(range(len(rows)),2):
        sim = jaccard(gs[i], gs[j])
        if sim >= .30:
            pairs.append({'row_a':i+1, 'row_b':j+1, 'source_id_a':rows[i]['source_id'],
                'source_id_b':rows[j]['source_id'], 'fivegram_jaccard':round(sim,6),
                'passes_preparation_cluster_rule':sim>=.90})
    pairs.sort(key=lambda x:-x['fivegram_jaccard'])
    counts=collections.Counter(f for r in record_audit for f in r['flags'])
    summary={'input_sha256': hashlib.sha256(raw).hexdigest(), 'record_count':len(rows),
        'unsafe':sum(r['mapped_unsafe'] for r in rows), 'safe':sum(not r['mapped_unsafe'] for r in rows),
        'schema_pass':True, 'unique_source_ids':len({r['source_id'] for r in rows}),
        'unique_cluster_ids':len({r['cluster_id'] for r in rows}),
        'unique_normalized_prompts':len({normalize(r['original_prompt']) for r in rows}),
        'unique_normalized_responses':len({normalize(r['original_response']) for r in rows}),
        'unique_normalized_pairs':len({(normalize(r['original_prompt']),normalize(r['original_response'])) for r in rows}),
        'source_ids_reproduce':not counts['source_id_mismatch'], 'variant_body_checks_passed':variant_checks,
        'flag_counts':dict(counts), 'rows_with_boundary_flag':sum(any('boundary_review' in f for f in r['flags']) for r in record_audit),
        'pairs_at_preparation_threshold':sum(p['passes_preparation_cluster_rule'] for p in pairs),
        'pairs_for_broader_review':len(pairs),
        'status':'NOT_YET_QUALIFIED', 'scope':'Attachment checks only; no source reconciliation, human approval or GPU inference.'}
    (HERE/'automated_checks.json').write_text(json.dumps({'summary':summary,'records':record_audit,'similarity_review_pairs':pairs},indent=2))
    with (HERE/'review_queue.csv').open('w',newline='') as f:
        fields=['row','source_id','mapped_unsafe','flags','source_match','human_review','reviewer_1','reviewer_2','adjudication','eligible']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in record_audit:
            item={k:r.get(k,'') for k in fields};item['flags']='; '.join(r['flags']);writer.writerow(item)
    print(json.dumps(summary,indent=2))
    print('Broader similarity candidates:', json.dumps(pairs,indent=2))
    return summary

if __name__ == '__main__':
    run()
