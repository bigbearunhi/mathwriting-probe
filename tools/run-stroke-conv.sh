#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=runs/ctc-bezier002-strokeconv-5k
mkdir -p "$out/source"
cp stroke_conv.py ctc_train.py ctc_data.py train_stroke_conv.py "$out/source/"
printf '{"stage":"preprocessing","target":5000,"tolerance":0.02}\n' > "$out/status.json"
trap 'printf "{\"stage\":\"failed\",\"target\":5000}\n" > "$out/status.json"' ERR
python3 -u ctc_data.py --archive data/mathwriting-2024.tgz --out data/ctc-tolerance-002 --workers 8 --tolerance .02 > "$out/preprocess.log" 2>&1
python3 -u train_stroke_conv.py > "$out/train.log" 2>&1
