"""Run the exploratory B/R/N/R2 pilot, one model process at a time."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REVISIONS = {
    'ministral': 'f6fae9795746f63c9be8344932f01275f3c63734',
    'gemma': '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True, help='New private run directory')
    parser.add_argument('--models', nargs='+', choices=REVISIONS, default=list(REVISIONS))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    from protocol import validate_records
    source = Path(args.input).resolve()
    records = validate_records([json.loads(x) for x in source.read_text().splitlines() if x.strip()])
    if any(r['study_partition'] != 'development' for r in records):
        parser.error('Pilot requires development records; never reuse final-test data')
    if len(set(args.models)) != len(args.models):
        parser.error('Models must be unique')
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parent
    manifest = {'purpose': 'exploratory_development_pilot', 'data_qualified': False,
        'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'records': len(records), 'variants': ['B','R','N','R2'],
        'models': args.models, 'revisions': REVISIONS, 'dry_run': args.dry_run,
        'planned_judgments': len(records)*8*len(args.models),
        'identity_calls': 0 if args.dry_run else 2*len(args.models), 'status': 'started',
        'runner_sha256': hashlib.sha256((repo/'inference.py').read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256((repo/'protocol.py').read_bytes()).hexdigest()}
    def save():
        (root/'pilot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    try:
        for model in args.models:
            command = [sys.executable, '-u', str(repo/'inference.py'), '--input', str(source),
                '--model', model, '--revision', REVISIONS[model], '--compute-dtype', 'bfloat16',
                '--variants', 'B','R','N','R2', '--out', str(root/model)]
            if args.dry_run:
                command.append('--dry-run')
            with (root/(model+'.log')).open('w') as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as proc:
                    for line in proc.stdout:
                        print(line, end='', flush=True)
                        log.write(line); log.flush()
                    if proc.wait():
                        raise RuntimeError(f'{model} failed; partial artifacts retained in {root}')
        manifest['status'] = 'planned_only' if args.dry_run else 'inference_complete_audit_pending'
    except BaseException:
        manifest['status'] = 'failed_or_interrupted'
        raise
    finally:
        save()
    print(manifest['status'])

if __name__ == '__main__':
    main()
