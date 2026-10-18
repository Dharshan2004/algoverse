import json
import hashlib
import unicodedata
from pathlib import Path
import pandas as pd

preview = pd.read_csv("data/preview/jailjudge_preview.csv")
audit = json.loads(
    Path("data/preview/cluster_audit.json").read_text()
)

def normalize(text):
    return " ".join(
        unicodedata.normalize("NFKC", str(text)).casefold().split()
    )

# Reuse the IDs and clusters from your existing audit.
source_to_cluster = {
    source_id: cluster["cluster_id"]
    for cluster in audit["clusters"]
    for source_id in cluster["source_ids"]
}

records = []

for _, row in preview.iterrows():
    prompt = str(row["user_prompt"])
    response = str(row["model_response"])

    response_hash = hashlib.sha256(
        response.encode("utf-8")
    ).hexdigest()

    fingerprint = normalize(prompt) + "\n" + response_hash
    source_id = hashlib.sha256(
        fingerprint.encode("utf-8")
    ).hexdigest()[:16]

    if source_id not in source_to_cluster:
        raise ValueError(f"Missing cluster for {source_id}")

    records.append({
        "source_id": source_id,
        "cluster_id": source_to_cluster[source_id],
        "original_prompt": prompt,
        "original_response": response,
        "study_partition": "development",
        "mapped_unsafe": bool(row["is_jailbroken"]),
        "label_provenance": "human",
    })

output = Path("data/preview/candidate_inputs.jsonl")
with output.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Created {len(records)} records: {output}")
print("True labels:", sum(r["mapped_unsafe"] for r in records))
print("False labels:", sum(not r["mapped_unsafe"] for r in records))
print("Candidate only — qualification checks remain outstanding.")
