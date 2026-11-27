# Animated code and pilot walkthrough

[Watch the narrated video](pilot-explained.mp4) · [Read the transcript](transcript.md)

Nine animated scenes explain the experiment, four response variants, A/B mappings, inference, activation snapshots, audits, observed results, limitations, and notebook tracking. Narration is a synthetic macOS Samantha voice. Text captions are rendered into the video. No raw dataset examples appear.

Sources: the tracked `protocol.py`, `inference.py`, `pilot.py`, `pilot_report.py`, the [audited result reports](../../reports/pilot-2026-10-17/README.md), and the [notebook history](../../notebooks/README.md). Result percentages average the two mappings on the same 25 source-unsafe records; repeated calls are not independent examples. Nothing here establishes statistical significance or a causal mechanism.

To regenerate on macOS, install Pillow and imageio-ffmpeg in an isolated Python environment and run `python media/pilot-explainer/render.py` from the repository. The macOS speech service must be accessible; the renderer rejects empty narration. Intermediate frames/audio remain under ignored `runs/video-build/`. `video.json` records duration and encoding settings.
