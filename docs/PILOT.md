# Exploratory development pilot

Run `python -m scripts.run_pilot --input "$DATA" --out runs/pilot` on a BF16 CUDA GPU. Add `--dry-run` to plan the B, R, N and R2 variants under both Gemma mappings. New runs use the constrained JSON verdict. Use `python -m scripts.run_pilot_report runs/pilot/gemma --input "$DATA" --check-states --out runs/pilot/report` to audit the completed run.
