#!/bin/bash

BASE_CMD="docker run --rm -v $(pwd)/experiments:/home/ray/experiments -v $(pwd)/pred:/home/ray/pred -v $(pwd)/models_1600_1:/home/ray/results -v $(pwd)/scripts:/home/ray/scripts -v $(pwd)/data:/home/ray/data beomyeol/faro-operator-1600:234b913"
BASE_CMD+=" python pred/train.py /home/ray/experiments/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented/top9_twitter_1_1600_avgproc_min_int5m_reduced_6hr_augmented.pkl --tool=darts"

$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=nhits --layer-width=512 --stacks=3 --blocks=1 --batch-size=32 --dropout=0.1 --lr=1e-4 --idx=0 --likelihood=gaussian --out-dir=/home/ray/results
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=nhits --layer-width=512 --stacks=3 --blocks=1 --batch-size=32 --dropout=0.1 --lr=1e-4 --idx=1 --likelihood=gaussian --out-dir=/home/ray/results
$BASE_CMD --context-len=10 --pred-len=7 --epochs=400 --layers=2 --model-name=nhits --layer-width=512 --stacks=3 --blocks=1 --batch-size=32 --dropout=0.1 --lr=1e-4 --idx=2 --likelihood=gaussian --out-dir=/home/ray/results
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=nhits --layer-width=512 --stacks=3 --blocks=1 --batch-size=32 --dropout=0.1 --lr=1e-4 --idx=3 --likelihood=gaussian --out-dir=/home/ray/results
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=nhits --layer-width=512 --stacks=3 --blocks=1 --batch-size=32 --dropout=0.1 --lr=1e-4 --idx=4 --likelihood=gaussian --out-dir=/home/ray/results
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=nhits --layer-width=512 --stacks=3 --blocks=1 --batch-size=32 --dropout=0.1 --lr=1e-4 --idx=5 --likelihood=gaussian --out-dir=/home/ray/results
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=nhits --layer-width=512 --stacks=3 --blocks=1 --batch-size=32 --dropout=0.1 --lr=1e-4 --idx=6 --likelihood=gaussian --out-dir=/home/ray/results
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=nhits --layer-width=512 --stacks=3 --blocks=1 --batch-size=32 --dropout=0.1 --lr=1e-4 --idx=7 --likelihood=gaussian --out-dir=/home/ray/results
$BASE_CMD --context-len=10 --pred-len=7 --epochs=400 --layers=2 --model-name=nhits --layer-width=512 --stacks=3 --blocks=1 --batch-size=32 --dropout=0.1 --lr=1e-4 --idx=8 --likelihood=gaussian --out-dir=/home/ray/results
$BASE_CMD --context-len=15 --pred-len=7 --epochs=400 --layers=2 --model-name=nhits --layer-width=512 --stacks=3 --blocks=1 --batch-size=32 --dropout=0.1 --lr=1e-4 --idx=9 --likelihood=gaussian --out-dir=/home/ray/results

