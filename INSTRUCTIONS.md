# Reproducing the paper

Tested with torch 2.10 (CUDA 12.8 wheels) on Ampere, Ada, and Blackwell GPUs. The sLSTM CUDA kernel is JIT-compiled on the first run with the system `nvcc`; set `CUDA_HOME` if it is not in `/usr/local/cuda`. Without a system toolkit, install one with `conda install -c nvidia cuda-toolkit=12.8` and `export CUDA_HOME=$CONDA_PREFIX CPATH=$CONDA_PREFIX/targets/x86_64-linux/include`. All commands below run from `ProbTS/`, after the setup in the [Quick start](README.md#quick-start).

## Single run

```bash
bash scripts/runner.sh DATASET HORIZON SEED    # e.g. etth1 720 0
```

Dataset names are the folder names in [`configs/ltsf/`](configs/ltsf/) and [`configs/stsf/`](configs/stsf/), e.g. `illness_ltsf` for ILI. Long-term horizons are 96/192/336/720 (ILI: 24/36/48/60) with context 96 (ILI: 36). Short-term settings, such as `solar 24` or `exchange 30`, are picked up from `stsf.yaml` automatically. Any [flag](#paper-tables) after `SEED` is passed on, e.g. `--gpu 1`.

## Paper tables

The paper reports the mean ± std over 5 seeds:

```bash
python scripts/runner.py scripts/LTSF.yaml --seed 0 1 2 3 4 --gpu 0    # long-term: 9 datasets × 4 horizons
python scripts/runner.py scripts/stsf.yaml --seed 0 1 2 3 4 --gpu 0    # short-term: 8 datasets
```

This is 220 training runs, so `--slurm` is the practical route. Useful flags:
- `--slurm` submits each run as a SLURM job. Add `--partition`, `--gpus`, `--qos`, or `--exclude` as your cluster needs.
- `--dataset` and `--horizon` run a subset of the grid.
- `--resume` skips completed runs and retries failed ones.
- `--dry-run` lists the jobs without running them.

## Efficiency

```bash
python scripts/measure_efficiency.py --gpu 0                             # all 36 long-term pairs
python scripts/measure_efficiency.py --only_cells ETTh1-L_96 ILI-L_24    # a subset
```

Weights are randomly initialized, so no data or training is needed. The protocol is FP32, batch size 1, and 100 trajectories per window; latency is the median of 30 timed calls after 10 warm-up calls, over 3 seeds. The paper's numbers are from an RTX 6000 Ada. Results are written to `../logs/efficiency/kgo.yaml`.

## Outputs

Each run writes to `../logs/<experiment>/`, e.g. `../logs/ltsf_etth1_KGOModel_96_720_0/`:
- `resolved.yaml` holds the full config.
- `run.log` holds the training log.
- `horizons_results.csv` holds the test metrics. CRPS is `test_CRPS`, and NMAE is `test_ND`.
- `status.json` holds the run state and the same two metrics under `metrics`.
