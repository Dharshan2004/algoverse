# Refusal directions (placeholder)

Loaders and artifacts for interpretability probes and steering directions used with this study.

The inference runner exposes an edit interface on saved activations (`inference.Runner.run(..., edit=...)`). This package will hold pinned probe definitions, direction vectors, and validation hooks once they are ready for integration. Do not commit large `.pt` weight files; use `artifacts/` locally (gitignored) or external storage documented in the root README.
