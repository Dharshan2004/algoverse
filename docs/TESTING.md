# Testing

Install the repository and CPU test dependencies with `python -m pip install -e . pytest torch numpy pandas`. Run `python -m pytest -q` from the checkout. Tests use synthetic prompts and cover JSON decoding, native guard parsing, cue plans, paired metrics and activation capture helpers. Checkpoint loading and GPU inference require the experiment runtime.
