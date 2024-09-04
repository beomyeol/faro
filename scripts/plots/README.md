- [Notebooks to generate figures](#notebooks-to-generate-figures)
- [Probabilistic distribution figures](#probabilistic-distribution-figures)
  - [Generate models](#generate-models)
  - [Generate plots](#generate-plots)
- [Utility timeline and other figures](#utility-timeline-and-other-figures)
- [Breakdown figures](#breakdown-figures)
- [Solver figures](#solver-figures)


# Notebooks to generate figures
- [plot_utility_timeline.ipynb](plot_utility_timeline.ipynb): Figure 1, 2, 4b, 11
- [roadmap.drawio](roadmap.drawio): Figure 3
- [plot_utility.ipynb](plot_utility.ipynb): Figure 4a, 6
- [plot_solver.ipynb](plot_solver.ipynb): Figure 5
- [plot_prob_pred.ipynb](plot_prob_pred.ipynb): Figure 8
- [plot_hierarchical_utility.ipynb](plot_hierarchical_utility.ipynb): Figure 7
- [figure.drawio](figure.drawio): Figure 9
- [plot_stats.ipynb](plot_stats.ipynb): Figure 10, 12, 13, 15 and Table 7
- [plot_stats_mixed.ipynb](plot_stats_mixed.ipynb): Figure 14
- [plot_breakdown.ipynb](plot_breakdown.ipynb): Figure 16
- [plot_stats_large.ipynb](plot_stats_large.ipynb): Table 8 (100 jobs)
- [plot_stats_32vms.ipynb](plot_stats_32vms.ipynb): Table 8 (20 jobs)

# Probabilistic distribution figures

## Generate models
```bash
$ python pred/train.py data/azure/2019/scaled_by_day8_max40_sample8_day8-12.pkl --tool=darts --context-len=60 --pred-len=40 --lr=1e-4 --epochs=200 --blocks=2 --stacks=3 --layers=4 --model-name=nhits --layer-width=256 --batch-size=32 --dropout=0.1
$ python pred/train.py data/azure/2019/scaled_by_day8_max40_sample8_day8-12.pkl --tool=darts --context-len=60 --pred-len=40 --lr=1e-4 --epochs=200 --blocks=2 --stacks=3 --layers=4 --model-name=nhits --layer-width=256 --batch-size=32 --dropout=0.1 --likelihood=gaussian
```

This will output trained models at `results/pred/scaled_by_day8_max40_sample8_day8-12`.

## Generate plots

Run [plot_prob_pred.ipynb](plot_prob_pred.ipynb)


# Utility timeline and other figures

See [plot_utility_timeline.ipynb](plot_utility_timeline.ipynb), [plot_stats.ipynb](plot_stats.ipynb), [plot_stats_mixed.ipynb](plot_stats_mixed.ipynb)


# Breakdown figures

See [plot_breakdown.ipynb](plot_breakdown.ipynb)

# Solver figures

See [plot_solver.ipynb](plot_solver.ipynb)