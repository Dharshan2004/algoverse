"""Render an audited AB comparison summary into a compact reviewable Markdown report."""
import json,sys
from pathlib import Path

def render(path):
    path=Path(path);r=json.loads(path.read_text());arms=r['arms'];comp=r['comparisons'];lines=[
        '# Gemma constrained-output comparison — 18 October 2026','',
        '**Completed experiment, exploratory classification evidence.** Output protocol `ab_constrained_v1` was run on a Colab NVIDIA L4 with BF16/NF4, based on PR #3 commit `03b32cb2ac6ae6aabf67a6b2da7632c3de852834` plus versioned implementation `20450d8`. Original pilot files and their ten INVALIDs were preserved.','',
        'The experiment ran the original and constrained protocols on all 50 development records, four variants and both mappings, plus eight clean synthetic controls under the same conditions: **928 judgments and eight identity calls**. All four arm audits passed, including every saved activation file.','',
        '## Formatting and classification are different outcomes','',
        'Validity means a parseable code. Accuracy here means agreement with supplied labels, which remain unqualified for the dataset; controls have authored synthetic labels. INVALID counts as incorrect in the all-judgment denominator. Valid-only accuracy has a changing denominator and must not be used alone to claim improvement.','',
        '| Cohort / protocol | Valid output | Correct / all | Accuracy / all | Accuracy / valid only |','|---|---:|---:|---:|---:|']
    data=[('Original pilot',r['original_pilot']['quality'])]+[(n,a['quality']) for n,a in arms.items()]
    for name,q in data:lines.append(f"| {name} | {q['format_valid_count']}/{q['rows']} ({q['format_valid_pct']:.2f}%) | {q['correct_count_including_invalid_denominator']}/{q['rows']} | {q['accuracy_all_pct']:.2f}% | {q['accuracy_valid_only_pct']:.2f}% |")
    lines+=['','**Interpretation:** this protocol fixes formatting in the tested run, but does not establish a classification-quality improvement. The net dataset gain is one correct judgment, while clean-control accuracy falls by three judgments. Keep this as a separately reported experimental arm; do not replace the original pilot or call its ten INVALIDs solved semantically.','',
        'Three previously correct dataset judgments became wrong. On controls, the added errors are false alarms on a privacy refusal (R and N, A=UNSAFE) and fair-hiring advice (R2, A=UNSAFE). The native error on a response encouraging unauthorized medical-record disclosure (R, A=UNSAFE) remains. All four constrained control errors occur under A=UNSAFE; inspect mapping sensitivity before considering a new protocol version.']
    change=comp['original_to_constrained'];drift=comp['original_to_native_rerun']
    lines+=['','## What happened to the original ten INVALIDs','',
        f"Of the ten originally invalid judgments, **{change['original_invalids_now_source_correct']} became source-label-correct and {change['original_invalids_now_source_wrong']} became source-label-wrong**. The constrained protocol also changed **{change['previously_valid_verdict_changes']} previously valid verdicts**. A valid answer therefore cannot be equated with a correct answer.",'',
        'Original → constrained transitions: `'+json.dumps(change['transitions'],sort_keys=True)+'`.','',
        'Original → native rerun transitions: `'+json.dumps(drift['transitions'],sort_keys=True)+'`.','',
        '## Dataset classification by condition','',
        'Rates below average both mappings on the same 25 source-unsafe and 25 source-safe records. Lower false-safe/false-alarm rates are better; higher recall/specificity are better. These are 50 records, not 400 independent examples.','',
        '| Variant | Native false-safe | Constrained false-safe | Native unsafe recall | Constrained unsafe recall | Native safe specificity | Constrained safe specificity | Native safe false alarm | Constrained safe false alarm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    a=arms['pilot_native_v1']['quality']['per_condition'];b=arms['pilot_ab_constrained_v1']['quality']['per_condition']
    for v in ['B','R','N','R2']:
        vals=[]
        for k in ['false_safe_pct','unsafe_recall_pct','safe_specificity_pct','safe_false_alarm_pct']:
            vals.extend([a['mapping_averaged'][v][k],b['mapping_averaged'][v][k]])
        lines.append('| '+v+' | '+' | '.join(f'{x:.1f}%' for x in vals)+' |')
    lines+=['','### Mapping sensitivity','', '| Variant | Native FSR: A=SAFE | Native FSR: A=UNSAFE | Constrained FSR: A=SAFE | Constrained FSR: A=UNSAFE |','|---|---:|---:|---:|---:|']
    for v in ['B','R','N','R2']:
        vals=[s['by_mapping'][m][v]['false_safe_pct'] for s in [a,b] for m in ['A_safe','A_unsafe']]
        lines.append('| '+v+' | '+' | '.join(f'{x:.1f}%' for x in vals)+' |')
    lines+=['',f"R-minus-B false-safe change: native {a['cue_effect_pp']['R']['average']:+.1f} points; constrained {b['cue_effect_pp']['R']['average']:+.1f} points. Robust B→R UNSAFE-to-SAFE flips under both mappings: native {a['robust_flip_count']}, constrained {b['robust_flip_count']}.",'',
        '## Audit and scope','',
        'Both protocols use identical prompts, tokenized input hashes and checkpoint revision. The constrained protocol masks all generated token choices except A/B and stops after one complete answer token. The native protocol retains its four-token budget. The saved probability scores are the original full-vocabulary scores, not normalized A/B probabilities.','']
    for cohort in ['pilot','controls']:
        c=comp[cohort+'_same_session'];lines.append(f"- {cohort}: identical saved prefill activations in {c['identical_prefill_state_rows']}/{c['rows']} pairs; maximum original A/B log-probability difference {c['max_original_logp_absolute_difference']:.8g}.")
    lines+=['','Every constrained raw output was checked to be a single actual generated A/B token equal to the raw-score A/B argmax. All identity checks passed. Token budget completion is explicitly recorded as `fixed_code_complete`; it is not mislabeled as a failed native truncation.','',
        'Clean controls are eight complete, authored policy examples (four safe/four unsafe), not independently annotated benchmark gold. Simple control success cannot establish general accuracy. The development dataset still has source-label, text-boundary and family-grouping review items. No significance, generalization or causal-mechanism claim is made. No original verdict was patched and no failed cases were selectively rerun.','',
        'See [summary.json](summary.json) for all four arm audits, checkpoint/environment metadata, output hashes, complete per-mapping counts, low A/B code-mass counts and comparisons with the preserved original pilot. Source/overlay and artifact preservation checks are in [provenance.json](provenance.json).','',
        'Execution: [sanitized executed notebook](../../notebooks/history/2026-10-18/gemma_ab_v1.executed.sanitized.ipynb). Exact executed notebook, raw rows and tensors remain under ignored `runs/gemma-ab-v1-2026-10-18/`. The clean [launch notebook](../../notebooks/gemma_ab_v1.ipynb) and [reproduction instructions](../../AB_REPRODUCTION.md) are tracked. No remote push is included.']
    (path.parent/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':render(sys.argv[1])
