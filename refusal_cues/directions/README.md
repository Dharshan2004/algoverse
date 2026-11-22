# Refusal directions (placeholder)

Interpretability probes and steering directions for this study live here.

The planned work covers direction training, validation, and non-identity intervention hooks for `inference.Runner`.

The inference runner already exposes an edit interface on saved activations; this package will hold pinned probe artifacts and loaders once they are ready for integration. Do not commit large `.pt` weight files; use `artifacts/` locally (gitignored) or external storage documented in the root README.
