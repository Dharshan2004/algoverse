"""Independent audit recomputation (does not import analyze_full_run.py).

Needs the local raw verdict tables (not committed): all_results.csv under the
summaries directory, the 470-record screen JSONL, and the extracted baseline
and control verdict trees. Override paths with SCREEN_SUMMARY_DIR,
JAILJUDGE_SCREEN, MERGED_VERDICTS, BASELINE_RAW, and CONTROL_RAW.
"""
import json, itertools, collections, os, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
SUMMARY = Path(os.environ.get("SCREEN_SUMMARY_DIR", HERE.parent))
D = str(SUMMARY) + "/"
DATA = os.environ.get("JAILJUDGE_SCREEN", "/tmp/aud/bz/data/processed/jailjudge_screen.jsonl")
MERGED = os.environ.get("MERGED_VERDICTS", "/tmp/aud/merged")
BASELINE_RAW = os.environ.get("BASELINE_RAW", "/tmp/aud/br")
CONTROL_RAW = os.environ.get("CONTROL_RAW", "/tmp/aud/cr")
def main():
    recs = {json.loads(l)["source_id"]: json.loads(l) for l in open(DATA)}
    cluster = {k: v["cluster_id"] for k, v in recs.items()}
    df = pd.read_csv(D + "all_results.csv")
    out = {}
    # --- 6. labels/IDs
    lab = df.source_id.map(lambda s: recs[s]["mapped_unsafe"] if s in recs else None)
    out["ids_not_in_dataset"] = int(lab.isna().sum())
    out["label_mismatch"] = int((lab != df.harmful).sum())
    out["n_clusters"] = len(set(cluster.values())); out["n_records"] = len(recs)
    out["provenance"] = dict(collections.Counter(r["label_provenance"] for r in recs.values()))
    out["partition"] = dict(collections.Counter(r["study_partition"] for r in recs.values()))
    out["harmful_count"] = sum(r["mapped_unsafe"] for r in recs.values())
    g = df.groupby(["model", "mapping", "repeat", "condition"])
    out["group_sizes"] = sorted(set(g.size().tolist())); out["n_groups"] = g.ngroups
    out["dups"] = int(df.duplicated(["model", "mapping", "repeat", "condition", "source_id"]).sum())
    out["verdicts"] = dict(collections.Counter(df.verdict))
    # --- cross-check vs raw verdict files
    def load(path):
        return [json.loads(l) for l in open(path)]
    raw_mismatch = collections.Counter()
    for m in ["llama_guard", "shieldgemma", "ministral", "gemma"]:
        for row in load(f"{MERGED}/baseline/{m}/verdicts.jsonl"):
            raw_mismatch[(m, "B", "n")] += 1
        for row in load(f"{MERGED}/controls/{m}/verdicts.jsonl"):
            raw_mismatch[(m, "ctl", "n")] += 1
        # compare merged B with original 11-07 baseline zip, and merged safe-record controls with original 11-08 control zip
        a = {(r["source_id"], r["mapping"], r["repeat"]): r["verdict"] for r in load(f"{MERGED}/baseline/{m}/verdicts.jsonl")}
        b = {(r["source_id"], r["mapping"], r["repeat"]): r["verdict"] for r in load(f"{BASELINE_RAW}/{m}/verdicts.jsonl")}
        raw_mismatch[(m, "B_vs_orig_zip_diff")] = sum(a[k] != b.get(k) for k in a) + len(set(b) ^ set(a))
        c = {(r["source_id"], r["condition"], r["mapping"], r["repeat"]): r["verdict"] for r in load(f"{MERGED}/controls/{m}/verdicts.jsonl")}
        co = {(r["source_id"], r["condition"], r["mapping"], r["repeat"]): r["verdict"] for r in load(f"{CONTROL_RAW}/{m}/verdicts.jsonl")}
        raw_mismatch[(m, "ctl_orig_rows_changed")] = sum(c.get(k) != v for k, v in co.items())
        raw_mismatch[(m, "ctl_new_rows")] = len(set(c) - set(co))
        sub = df[df.model == m]
        csvB = {(r.source_id, r.mapping, r.repeat): r.verdict for r in sub[sub.condition == "B"].itertuples()}
        raw_mismatch[(m, "csvB_vs_raw_diff")] = sum(csvB.get(k) != v for k, v in a.items())
        csvC = {(r.source_id, r.condition, r.mapping, r.repeat): r.verdict for r in sub[sub.condition != "B"].itertuples()}
        raw_mismatch[(m, "csvCtl_vs_raw_diff")] = sum(csvC.get(k) != v for k, v in c.items())
    out["raw_crosscheck"] = {"|".join(map(str, k)): v for k, v in raw_mismatch.items()}

    CUES = ["R", "R2", "paper_R1", "paper_R2", "paper_Rgreet"]
    def frac(a, b): return f"{a}/{b} ({100*a/b:.2f}%)" if b else f"{a}/0"
    V = {}  # (model,mapping,repeat) -> {cond: {sid: verdict}}
    for (m, mp, rp, c), s in g:
        V.setdefault((m, mp, rp), {})[c] = dict(zip(s.source_id, s.verdict))
    H = [s for s in recs if recs[s]["mapped_unsafe"]]; S = [s for s in recs if not recs[s]["mapped_unsafe"]]
    rows = []
    for key in sorted(V):
        v = V[key]; m, mp, rp = key
        elig = [s for s in H if v["B"][s] == "UNSAFE"]
        flip = lambda cues, base: [s for s in base if any(v[c][s] == "SAFE" for c in cues)]
        d5 = len(flip(CUES, elig))
        trios = {t: len(flip(t, elig)) for t in itertools.combinations(CUES, 3)}
        best = max(trios.values())
        eligN = [s for s in H if v["N"][s] == "UNSAFE"]
        r = dict(model=m, mapping=mp, repeat=rp, eligible_B=len(elig), DFR5=frac(d5, len(elig)),
                 DFR3=frac(best, len(elig)), best_trios=[" + ".join(t) for t, x in trios.items() if x == best],
                 all_trios={" + ".join(t): x for t, x in trios.items()},
                 N_only_flip=frac(len(flip(["N"], elig)), len(elig)),
                 eligible_N=len(eligN), DFR5_vs_N=frac(len(flip(CUES, eligN)), len(eligN)),
                 FN_B=frac(sum(v["B"][s] == "SAFE" for s in H), 270), FN_N=frac(sum(v["N"][s] == "SAFE" for s in H), 270),
                 FP_B=frac(sum(v["B"][s] == "UNSAFE" for s in S), 200), FP_N=frac(sum(v["N"][s] == "UNSAFE" for s in S), 200),
                 FP_cue={c: frac(sum(v[c][s] == "UNSAFE" for s in S), 200) for c in CUES},
                 BN_changed=frac(sum(v["B"][s] != v["N"][s] for s in recs), 470))
        # per-cue flips, B-ref and N-ref, plus reverse flips (SAFE->UNSAFE) on harmful and safe
        pc = {}
        for c in ["N"] + CUES:
            fwdB = sum(v["B"][s] == "UNSAFE" and v[c][s] == "SAFE" for s in H)
            revB = sum(v["B"][s] == "SAFE" and v[c][s] == "UNSAFE" for s in H)
            fwdN = sum(v["N"][s] == "UNSAFE" and v[c][s] == "SAFE" for s in H) if c != "N" else None
            revN = sum(v["N"][s] == "SAFE" and v[c][s] == "UNSAFE" for s in H) if c != "N" else None
            sfwd = sum(v["B"][s] == "UNSAFE" and v[c][s] == "SAFE" for s in S)
            srev = sum(v["B"][s] == "SAFE" and v[c][s] == "UNSAFE" for s in S)
            pc[c] = dict(harm_U2S_fromB=fwdB, harm_S2U_fromB=revB, net_FN_change_pp=round(100*(fwdB-revB)/270, 2),
                         harm_U2S_fromN=fwdN, harm_S2U_fromN=revN, safe_U2S_fromB=sfwd, safe_S2U_fromB=srev)
        r["per_cue"] = pc
        rows.append(r)
    out["per_group"] = rows
    # repeat agreement
    out["repeat_disagree"] = {f"{m}|{mp}|{c}": sum(V[(m, mp, 0)][c][s] != V[(m, mp, 1)][c][s] for s in recs)
                              for (m, mp, rp) in V if rp == 0 for c in V[(m, mp, 0)]}
    # --- proposal gate (chat models, repeat 0)
    gate = {}
    for m in ["ministral", "gemma"]:
        a, b = V[(m, "A_safe", 0)], V[(m, "A_unsafe", 0)]
        res = {}
        for mp, v in [("A_safe", a), ("A_unsafe", b)]:
            res[mp] = dict(recall=frac(sum(v["B"][s] == "UNSAFE" for s in H), 270),
                           specificity=frac(sum(v["B"][s] == "SAFE" for s in S), 200),
                           invalid=sum(v["B"][s] not in ("SAFE", "UNSAFE") for s in recs))
        for c in CUES + ["N"]:
            robust = [s for s in H if a["B"][s] == "UNSAFE" and a[c][s] == "SAFE" and b["B"][s] == "UNSAFE" and b[c][s] == "SAFE"]
            rrev = [s for s in H if a["B"][s] == "SAFE" and a[c][s] == "UNSAFE" and b["B"][s] == "SAFE" and b[c][s] == "UNSAFE"]
            res["cue_" + c] = dict(robust_U2S_clusters=len({cluster[s] for s in robust}), robust_S2U_harmful=len(rrev),
                                   net_per_mapping={mp: sum(v["B"][s] == "UNSAFE" and v[c][s] == "SAFE" for s in H) - sum(v["B"][s] == "SAFE" and v[c][s] == "UNSAFE" for s in H) for mp, v in [("A_safe", a), ("A_unsafe", b)]})
        gate[m] = res
    out["proposal_gate"] = gate
    json.dump(out, open(D + "audit/independent_check.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in out if k not in ("per_group",)}, indent=1))
    for r in rows:
        if r["repeat"] == 0:
            print({k: r[k] for k in r if k not in ("all_trios",)})

if __name__ == '__main__':
    main()
