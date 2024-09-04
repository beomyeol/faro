#!/bin/bash

# BASE_CMD="docker run --rm -v $(pwd)/experiments:/home/ray/experiments -v $(pwd)/pred:/home/ray/pred -v $(pwd)/models_prob_1:/home/ray/results -v $(pwd)/scripts:/home/ray/scripts -v $(pwd)/data:/home/ray/data beomyeol/faro-operator-1600:234b913"
# BASE_CMD+=" python pred/train.py /home/ray/experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented.pkl --tool=darts"

# BASE_CMD="python pred/train.py experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented.pkl --tool=darts"

BASE_CMD="python pred/train.py experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented.pkl --tool=darts"

$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=rnn --layer-width=512 --batch-size=32 --likelihood=gaussian --model-tag=deepar --seed=42 --dropout=0.1 --lr=1e-3 --idx=0 --out-dir=results --gpu=0 &
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=rnn --layer-width=512 --batch-size=32 --likelihood=gaussian --model-tag=deepar --seed=42 --dropout=0.1 --lr=1e-3 --idx=1 --out-dir=results --gpu=1 &
$BASE_CMD --context-len=10 --pred-len=7 --epochs=400 --layers=2 --model-name=rnn --layer-width=512 --batch-size=32 --likelihood=gaussian --model-tag=deepar --seed=42 --dropout=0.1 --lr=1e-3 --idx=2 --out-dir=results --gpu=2 &
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=rnn --layer-width=512 --batch-size=32 --likelihood=gaussian --model-tag=deepar --seed=42 --dropout=0.1 --lr=1e-3 --idx=3 --out-dir=results --gpu=3 &
wait
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=rnn --layer-width=512 --batch-size=32 --likelihood=gaussian --model-tag=deepar --seed=42 --dropout=0.1 --lr=1e-3 --idx=4 --out-dir=results --gpu=0 &
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=rnn --layer-width=512 --batch-size=32 --likelihood=gaussian --model-tag=deepar --seed=42 --dropout=0.1 --lr=1e-3 --idx=5 --out-dir=results --gpu=1 &
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=rnn --layer-width=512 --batch-size=32 --likelihood=gaussian --model-tag=deepar --seed=42 --dropout=0.1 --lr=1e-3 --idx=6 --out-dir=results --gpu=2 &
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=rnn --layer-width=512 --batch-size=32 --likelihood=gaussian --model-tag=deepar --seed=42 --dropout=0.1 --lr=1e-3 --idx=7 --out-dir=results --gpu=3 &
wait
$BASE_CMD --context-len=10 --pred-len=7 --epochs=400 --layers=2 --model-name=rnn --layer-width=512 --batch-size=32 --likelihood=gaussian --model-tag=deepar --seed=42 --dropout=0.1 --lr=1e-3 --idx=8 --out-dir=results --gpu=0 &
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=rnn --layer-width=512 --batch-size=32 --likelihood=gaussian --model-tag=deepar --seed=42 --dropout=0.1 --lr=1e-3 --idx=9 --out-dir=results --gpu=1 