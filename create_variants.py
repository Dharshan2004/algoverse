
from preview_io import read_preview
from protocol import variant
from pathlib import Path
import hashlib
import json

source = Path("data/preview/jailjudge_preview.csv")
output = Path("data/preview/variants.jsonl")

df = read_preview(source)

prefixes = {
    "B": "",
    "R+B": "I cannot help with that request.",
    "N+B": "I have read the request above.",
    "R2+B": "I am unable to assist with this request.",
}

with output.open("w", encoding="utf-8") as f:
    for index, row in df.iterrows():
        original = row["model_response"]

        if not isinstance(original, str) or not original:
            raise ValueError(f"Empty response at row {index}")

        for variant_id, prefix in prefixes.items():
            text = variant(original, variant_id.split("+")[0])["assistant_response"]

            record = {
                "preview_id": index,
                "variant_id": variant_id,
                "user_prompt": row["user_prompt"],
                "original_response": original,
                "variant_response": text,
                "body_sha256": hashlib.sha256(
                    original.encode("utf-8")
                ).hexdigest(),
                "is_jailbroken": bool(row["is_jailbroken"]),
                "hazard_category": row["hazard_cate_llamaguard3"],
                "complexity_category": row["prompt_complexity_cate"],
                "language": row["language"],
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Created {len(df) * len(prefixes)} variants at {output}")
print("Original responses preserved in the original_response field.")
