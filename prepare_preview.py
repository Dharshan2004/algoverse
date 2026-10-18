
import json
import pandas as pd
from pathlib import Path

source = Path("JAILJUDGE_ID.json")
output_dir = Path("data/preview")
output_dir.mkdir(parents=True, exist_ok=True)

# Read the original JSON file
with source.open("r", encoding="utf-8") as f:
    records = json.load(f)

# Convert records into a table
df = pd.DataFrame(records)

# Verify the expected schema
required = [
    "user_prompt",
    "model_response",
    "hazard_cate_llamaguard3",
    "prompt_complexity_cate",
    "language",
    "is_jailbroken",
]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

print("Total records:", len(df))
print("Available columns:", df.columns.tolist())

# Exclude empty prompts/responses and missing labels
df = df[
    df["user_prompt"].map(lambda x: isinstance(x, str) and x.strip() != "")
    & df["model_response"].map(
        lambda x: isinstance(x, str) and x.strip() != ""
    )
    & df["is_jailbroken"].map(lambda x: type(x) is bool)
].copy()

# Select up to 25 examples per source label
samples = []

for label in [True, False]:
    group = df[df["is_jailbroken"] == label]
    n = min(25, len(group))
    samples.append(group.sample(n=n, random_state=20261004))

preview = pd.concat(samples, ignore_index=True)
preview = preview.sample(frac=1, random_state=20261004).reset_index(drop=True)

# Save preview
preview.to_csv(output_dir / "jailjudge_preview.csv", index=False)

# Save category inventory
inventory = {
    "preview_records": len(preview),
    "label_counts": {
        str(k): int(v)
        for k, v in preview["is_jailbroken"].value_counts().items()
    },
    "hazard_categories": sorted(
        preview["hazard_cate_llamaguard3"].dropna().unique().tolist()
    ),
    "complexity_categories": sorted(
        preview["prompt_complexity_cate"].dropna().unique().tolist()
    ),
    "languages": sorted(preview["language"].dropna().unique().tolist()),
}

with open(output_dir / "category_inventory.json", "w", encoding="utf-8") as f:
    json.dump(inventory, f, indent=2, ensure_ascii=False)

print("\nPreview saved:", output_dir / "jailjudge_preview.csv")
print("Category inventory saved:", output_dir / "category_inventory.json")
print("\nLabel counts:")
print(preview["is_jailbroken"].value_counts())
print("\nHazard categories:")
print(preview["hazard_cate_llamaguard3"].value_counts(dropna=False))
print("\nComplexity categories:")
print(preview["prompt_complexity_cate"].value_counts(dropna=False))
