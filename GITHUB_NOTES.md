# Repository scope

This repository contains experiment code, training scripts, data preprocessing, evaluation tools and viewer generators. Downloaded data, feature caches, checkpoints and generated viewers are excluded from Git and remain local.

- Python 3.10; current experiment environment: PyTorch 2.9.0, NumPy, Pillow.
- Upstream MathWriting resources: https://github.com/google-research/google-research/tree/master/mathwriting (pinned provenance in provenance.json).
- Current model: 10D Bezier features, 512 hidden dimensions, 11 Transformer encoder layers, 8 heads, FFN 2048, CTC; no Transformer decoder.
- Later training scripts resume local checkpoints; a fresh clone cannot run those scripts without their source checkpoints and feature caches. See CTC_README.md for preprocessing and initial training.
- inspect_training.py requires local best weights and cached data; produces a one-sample-at-a-time training viewer.
- build_preprocess_viewer.py additionally uses the locally generated error-viewer sample data. After generation copy viewer_templates/preprocess.html into runs/preprocess-viewer/index.html.
- LaTeX syntax classification requires pdflatex and the documented packages.
- tools/try-dynamic-boost.sh is specific to the original laptop/driver setup, not a required training dependency.

The original README documents the early seq2seq pilot; later CTC scripts and CTC_README.md describe the subsequent experiments. Reported validation scores are not independent final test scores.
