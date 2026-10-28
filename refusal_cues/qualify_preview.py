import hashlib
import json
import re
import unicodedata
from pathlib import Path

from refusal_cues.preview_io import read_preview

SEED = "20261004"
INPUT = Path("data/preview/jailjudge_preview.csv")
OUTPUT = Path("data/preview/cluster_audit.json")

df = read_preview(INPUT)

def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    return " ".join(text.casefold().split())

def tokens(text):
    return re.findall(r"\w+", normalize(text))

def fivegrams(text):
    words = tokens(text)
    return {
        tuple(words[i:i + 5])
        for i in range(len(words) - 4)
    }

def similarity(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0

# Stable IDs derived from each original prompt-response pair.
records = []
for _, row in df.iterrows():
    prompt = str(row["user_prompt"])
    response = str(row["model_response"])
    response_hash = hashlib.sha256(
        response.encode("utf-8")
    ).hexdigest()

    fingerprint = (
        normalize(prompt) + "\n" + response_hash
    )
    source_id = hashlib.sha256(
        fingerprint.encode("utf-8")
    ).hexdigest()[:16]

    records.append({
        "source_id": source_id,
        "prompt": prompt,
        "response": response,
        "normalized_prompt": normalize(prompt),
        "tokens": tokens(prompt),
        "fivegrams": fivegrams(prompt),
        "response_sha256": response_hash,
    })

# Connected components: related prompts remain in one cluster.
parent = list(range(len(records)))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)

for i in range(len(records)):
    for j in range(i + 1, len(records)):
        a, b = records[i], records[j]

        if a["normalized_prompt"] == b["normalized_prompt"]:
            union(i, j)
            continue

        # Fewer than five tokens: exact match only.
        if len(a["tokens"]) < 5 or len(b["tokens"]) < 5:
            continue

        if similarity(a["fivegrams"], b["fivegrams"]) >= 0.90:
            union(i, j)

components = {}
for i, record in enumerate(records):
    components.setdefault(find(i), []).append(i)

clusters = []
for indices in components.values():
    member_ids = sorted(records[i]["source_id"] for i in indices)
    cluster_id = "cluster_" + member_ids[0]

    clusters.append({
        "cluster_id": cluster_id,
        "source_ids": member_ids,
        "size": len(member_ids),
    })

report = {
    "source_dataset": "JAILJUDGE_ID",
    "source_records_in_preview": len(records),
    "unique_source_ids": len({r["source_id"] for r in records}),
    "cluster_count": len(clusters),
    "multi_record_clusters": [
        c for c in clusters if c["size"] > 1
    ],
    "clusters": clusters,
    "method": {
        "normalization": "NFKC, casefold, whitespace normalization",
        "near_duplicate_rule": "five-word-token-set Jaccard >= 0.90",
        "short_prompts": "fewer than five tokens: exact normalized match only",
        "grouping": "connected components",
    },
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Records:", report["source_records_in_preview"])
print("Unique IDs:", report["unique_source_ids"])
print("Clusters:", report["cluster_count"])
print("Multi-record clusters:", len(report["multi_record_clusters"]))
print("Audit saved to:", OUTPUT)
print("This is a preview audit, not development-data approval.")
