#!/usr/bin/env python3
"""Measure KGO inference latency and peak GPU memory (paper Table 11).

Protocol
    FP32, batch size 1, 100 forecast trajectories per window. Latency is the
    median wall-clock time of `timed_runs` forecast calls per seed, taken after
    `n_warmup` discarded calls, with CUDA synchronisation around each call.
    Memory is the peak allocated by PyTorch, in GB (10^9 bytes). Weights are
    randomly initialised: latency and memory do not depend on their values.
    Electricity and Traffic draw their trajectories in chunks of five to bound
    peak memory.

Usage (from ProbTS/)
    python scripts/measure_efficiency.py --gpu 0
    python scripts/measure_efficiency.py --only_cells ETTh1-L_96 ILI-L_24
"""

import argparse
import contextlib
import importlib
import os
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
TIME_FEAT_DIM = {"H": 4, "min": 5, "D": 3, "B": 3, "W": 2}
LTSF_HORIZONS = [96, 192, 336, 720]


class Dataset(NamedTuple):
    config: str  # directory under configs/ltsf/
    variates: int
    freq: str
    context: int
    horizons: list
    chunk: Optional[int]  # num_samples_chunk; None draws all samples at once


DATASETS = {
    "ETTm1-L": Dataset("ettm1", 7, "min", 96, LTSF_HORIZONS, None),
    "ETTm2-L": Dataset("ettm2", 7, "min", 96, LTSF_HORIZONS, None),
    "ETTh1-L": Dataset("etth1", 7, "H", 96, LTSF_HORIZONS, None),
    "ETTh2-L": Dataset("etth2", 7, "H", 96, LTSF_HORIZONS, None),
    "Electricity-L": Dataset("electricity_ltsf", 321, "H", 96, LTSF_HORIZONS, 5),
    "Exchange-L": Dataset("exchange_ltsf", 8, "B", 96, LTSF_HORIZONS, None),
    "Traffic-L": Dataset("traffic_ltsf", 862, "H", 96, LTSF_HORIZONS, 5),
    "Weather-L": Dataset("weather_ltsf", 21, "min", 96, LTSF_HORIZONS, None),
    "ILI-L": Dataset("illness_ltsf", 7, "W", 36, [24, 36, 48, 60], None),
}
CELLS = [f"{name}_{h}" for name, spec in DATASETS.items() for h in spec.horizons]

ROW = "{:<16}{:>8}{:>15}{:>11}{:>14}"
RULE = "-" * 64


def load_kgo(spec, horizon, num_sampling):
    """Build a randomly initialised KGO forecaster from the dataset's config."""
    config_path = ROOT / "configs" / "ltsf" / spec.config / "KGO.yaml"
    forecaster = yaml.safe_load(config_path.read_text())["model"]["forecaster"]
    init_args = dict(forecaster["init_args"])
    init_args.update(
        target_dim=spec.variates,
        context_length=spec.context,
        prediction_length=horizon,
        freq=spec.freq,
        lags_list=[],
        use_scaling=False,
        time_feat_dim=TIME_FEAT_DIM[spec.freq],
        num_sampling=num_sampling,
        num_samples_chunk=spec.chunk,
    )
    module_path, class_name = forecaster["class_path"].rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)(**init_args)


def mock_batch(spec, horizon, device):
    """Build one random input window with the dataset's shapes."""
    from probts.data import ProbTSBatchData

    context, variates = spec.context, spec.variates
    time_feat_dim = TIME_FEAT_DIM[spec.freq]
    data = {
        "past_target_cdf": torch.randn(1, context, variates),
        "past_observed_values": torch.ones(1, context, variates),
        "past_is_pad": torch.zeros(1, context),
        "past_time_feat": torch.randn(1, context, time_feat_dim),
        "future_target_cdf": torch.randn(1, horizon, variates),
        "future_observed_values": torch.ones(1, horizon, variates),
        "future_time_feat": torch.randn(1, horizon, time_feat_dim),
        "target_dimension_indicator": torch.arange(variates).unsqueeze(0),
        "context_length": context,
        "prediction_length": horizon,
        "max_context_length": context,
        "max_prediction_length": horizon,
    }
    batch = ProbTSBatchData(data, device)
    # ProbTSBatchData records the device but leaves the tensors where they are.
    for key, value in vars(batch).items():
        if torch.is_tensor(value):
            setattr(batch, key, value.to(device))
    return batch


@torch.no_grad()
def time_forecast(forecaster, batch, device, n_warmup, timed_runs):
    """Return the wall-clock latency in seconds of each timed forecast call."""
    for _ in range(n_warmup):
        forecaster.forecast(batch)
        torch.cuda.synchronize(device)

    times = []
    for _ in range(timed_runs):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        forecaster.forecast(batch)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    return times


def measure_cell(spec, horizon, args, device):
    """Measure one dataset-horizon pair, pooling timed calls across seeds."""
    times, allocated, reserved = [], 0.0, 0.0
    for seed in args.seeds:
        torch.manual_seed(seed)
        # The sLSTM kernel prints progress messages when it is first built.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            forecaster = load_kgo(spec, horizon, args.num_sampling).to(device).eval()
            batch = mock_batch(spec, horizon, device)

            torch.cuda.reset_peak_memory_stats(device)
            times += time_forecast(
                forecaster, batch, device, args.n_warmup, args.timed_runs
            )
        allocated = max(allocated, torch.cuda.max_memory_allocated(device) / 1e9)
        reserved = max(reserved, torch.cuda.max_memory_reserved(device) / 1e9)

        del forecaster, batch
        torch.cuda.empty_cache()

    return {
        "latency_median_s": round(statistics.median(times), 6),
        "latency_std_s": round(statistics.stdev(times), 6) if len(times) > 1 else 0.0,
        "memory_allocated_gb": round(allocated, 4),
        "memory_reserved_gb": round(reserved, 4),
        "n": len(times),
    }


def measure_dataset(name, args):
    """Measure the requested cells of one dataset and add them to the output."""
    sys.path.insert(0, str(ROOT / "ProbTS"))
    device = torch.device(f"cuda:{args.gpu}")
    spec = DATASETS[name]

    results = yaml.safe_load(args.out.read_text()) if args.out.exists() else {}
    results.setdefault(
        "meta",
        {
            "gpu": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
            "precision": "fp32",
            "batch_size": 1,
            "num_sampling": args.num_sampling,
            "seeds": args.seeds,
            "n_warmup": args.n_warmup,
            "timed_runs": args.timed_runs,
        },
    )
    cells = results.setdefault("KGO", {})

    label = name
    for horizon in spec.horizons:
        cell = f"{name}_{horizon}"
        if cell not in args.only_cells:
            continue
        cells[cell] = stats = measure_cell(spec, horizon, args, device)
        print(
            ROW.format(
                label,
                horizon,
                f"{1000 * stats['latency_median_s']:.2f}",
                f"{1000 * stats['latency_std_s']:.2f}",
                f"{stats['memory_allocated_gb']:.3f}",
            ),
            flush=True,
        )
        label = ""
        args.out.write_text(yaml.safe_dump(results, sort_keys=False))


def measure_all(args):
    """Measure every requested dataset, each in its own process.

    Allocations left over from earlier datasets would otherwise inflate the
    peak memory of later ones.
    """
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.unlink(missing_ok=True)

    header = ROW.format(
        "Dataset", "Horizon", "Latency (ms)", "Std (ms)", "Memory (GB)"
    )
    print(header, RULE, sep="\n", flush=True)
    names = [
        name
        for name in DATASETS
        if any(cell.startswith(name + "_") for cell in args.only_cells)
    ]
    failures = 0
    for index, name in enumerate(names):
        if index:
            print(flush=True)
        command = [sys.executable, __file__, "--dataset", name] + sys.argv[1:]
        failures += subprocess.run(command).returncode != 0
    print(RULE)

    if failures:
        print(f"{failures} dataset(s) failed; see the errors above", file=sys.stderr)
        return 1
    print_summary(args.out)
    return 0


def print_summary(out):
    """Print the average over the measured cells and the run's settings."""
    results = yaml.safe_load(out.read_text())
    cells, meta = results["KGO"].values(), results["meta"]
    latency = 1000 * statistics.mean(c["latency_median_s"] for c in cells)
    memory = statistics.mean(c["memory_allocated_gb"] for c in cells)
    calls = len(meta["seeds"]) * meta["timed_runs"]
    print(
        "{:<24}{:>15}{:>11}{:>14}".format(
            f"Average ({len(cells)} cells)", f"{latency:.2f}", "", f"{memory:.3f}"
        )
    )
    print(
        f"\n{meta['gpu']} | FP32, batch size 1, {meta['num_sampling']} samples, "
        f"{calls} timed calls per cell"
    )
    print(f"Wrote {out}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_sampling", type=int, default=100)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--timed_runs", type=int, default=30)
    parser.add_argument(
        "--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2]
    )
    parser.add_argument(
        "--only_cells",
        nargs="+",
        default=CELLS,
        metavar="CELL",
        help="Cells to measure, e.g. ETTh1-L_96 (default: all 36)",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "logs" / "efficiency" / "kgo.yaml"
    )
    parser.add_argument("--dataset", choices=DATASETS, help=argparse.SUPPRESS)
    args = parser.parse_args()

    unknown = sorted(set(args.only_cells) - set(CELLS))
    if unknown:
        parser.error(f"unknown cells {unknown}; choose from: {', '.join(CELLS)}")
    args.out = args.out.resolve()
    return args


def main():
    args = parse_args()
    if args.dataset:
        return measure_dataset(args.dataset, args)
    return measure_all(args)


if __name__ == "__main__":
    sys.exit(main())
