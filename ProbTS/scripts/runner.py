#!/usr/bin/env python3
"""Run or submit KGO experiments directly into logs/<experiment>/."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[2]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def write_run_status(run_dir, status, submission_response=False):
    """A fast-starting Slurm worker must not be overwritten by its submitter."""
    path = run_dir / "status.json"
    with (run_dir / ".status.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if submission_response and path.exists():
            current = json.loads(path.read_text())
            if (
                current.get("created_at") == status.get("created_at")
                and current["state"] != "submitting"
            ):
                return
        write_json(path, status)


def enumerate_runs(grid_path, datasets=None, horizons=None, seeds=None):
    grid = yaml.safe_load(Path(grid_path).read_text())
    task = grid.get("task_type", "ltsf")
    runs = []
    for item in grid["datasets"]:
        item = {"name": item} if isinstance(item, str) else item
        dataset = item["name"]
        if datasets and dataset not in datasets:
            continue
        for horizon in item.get("prediction_lengths", grid["prediction_lengths"]):
            if horizons and horizon not in horizons:
                continue
            for seed in (
                seeds
                if seeds is not None
                else item.get("seeds", grid.get("seeds", [0]))
            ):
                context = item.get("context_length", grid["context_length"])
                data_name = item.get("data_name", dataset)
                run_id = f"{task}_{data_name}_KGOModel_{context}_{horizon}_{seed}"
                config_path = ROOT / "configs" / task / dataset / "KGO.yaml"
                config = yaml.safe_load(config_path.read_text())
                config["seed_everything"] = seed
                config["data"]["data_manager"]["init_args"].update(
                    context_length=context,
                    prediction_length=horizon,
                    dataset=item.get("data_name", dataset),
                )
                settings = item.get("horizon_settings", {}).get(horizon, {})
                for key in ("batch_size", "test_batch_size"):
                    if key in settings:
                        config["data"][key] = settings[key]
                if "num_sampling" in settings:
                    config["model"]["forecaster"]["init_args"]["num_sampling"] = settings["num_sampling"]
                    config["model"]["num_samples"] = settings["num_sampling"]
                runs.append(
                    dict(
                        run_id=run_id,
                        task=task,
                        dataset=dataset,
                        context_length=context,
                        prediction_length=horizon,
                        seed=seed,
                        config=config,
                    )
                )
    if not runs:
        raise ValueError("No settings match the requested grid/filter")
    return runs


def active_job(status):
    job_id = status.get("slurm_job_id")
    if not job_id or status.get("state") not in ("submitted", "running"):
        return False
    result = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"], text=True, capture_output=True
    )
    return bool(result.stdout.strip())


def execute(job_path):
    job = json.loads(Path(job_path).read_text())
    run_dir = Path(job["run_dir"])
    with (run_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Already running: {job['run_id']}")
            return 1
        status = {key: value for key, value in job.items() if key != "config"}
        status.update(
            state="running",
            started_at=timestamp(),
            host=os.uname().nodename,
            slurm_job_id=os.environ.get("SLURM_JOB_ID", job.get("slurm_job_id")),
        )
        write_run_status(run_dir, status)
        config = job["config"]
        config["trainer"]["default_root_dir"] = str(run_dir.parent)
        config_path = run_dir / "resolved.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [
            sys.executable,
            "-u",
            str(ROOT / "ProbTS/run.py"),
            "--config",
            str(config_path),
        ]
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        env["PROBTS_TASK"] = job["task"]
        env["PROBTS_TAG"] = job["run_id"]
        env["NO_COLOR"] = "1"
        env.setdefault("MAX_JOBS", "2")
        env.setdefault("OMP_NUM_THREADS", "2")
        if job.get("local_gpu") is not None and "SLURM_JOB_ID" not in env:
            env["CUDA_VISIBLE_DEVICES"] = str(job["local_gpu"])
        start = time.monotonic()
        print(f"Starting {job['run_id']}", flush=True)
        print(f"Log: {run_dir / 'run.log'}", flush=True)
        try:
            status["command"] = command
            # A record of the environment only: a missing tool must not fail the run.
            cuda_bin = Path(env.get("CUDA_HOME", "/usr/local/cuda")) / "bin"
            nvcc = shutil.which("nvcc", path=f"{cuda_bin}{os.pathsep}{env['PATH']}")
            with (run_dir / "environment.txt").open("w") as output:
                for tool in (
                    [sys.executable, "-m", "pip", "freeze"],
                    [shutil.which("nvidia-smi")],
                    [nvcc, "--version"],
                ):
                    if tool[0]:
                        subprocess.run(tool, stdout=output, stderr=subprocess.STDOUT)
            with (run_dir / "run.log").open("w", buffering=1) as output:
                with subprocess.Popen(
                    command,
                    cwd=ROOT / "ProbTS",
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ) as process:
                    for line in process.stdout:
                        output.write(line)
                        print(line, end="", flush=True)
                    returncode = process.wait()
            status["returncode"] = returncode
            if returncode:
                raise RuntimeError(
                    f"Training exited with code {returncode}; see {run_dir / 'run.log'}"
                )
            metrics_files = list(run_dir.glob("horizons_results.csv"))
            if len(metrics_files) != 1:
                raise RuntimeError(
                    f"Expected one metrics CSV, found {len(metrics_files)}"
                )
            with metrics_files[0].open() as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1:
                raise RuntimeError("Expected exactly one horizon result")
            metrics = {key: float(rows[0][key]) for key in ("test_CRPS", "test_ND")}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"Non-finite evaluation metrics: {metrics}")
            sidecar = metrics_files[0].parent / "ckpt/best_ckpt.txt"
            checkpoint = sidecar.read_text().strip()
            if not Path(checkpoint).is_file():
                raise RuntimeError(f"Best checkpoint missing: {checkpoint}")
            status.update(
                state="completed",
                metrics=metrics,
                metrics_file=str(metrics_files[0]),
                checkpoint=checkpoint,
            )
        except Exception as error:
            status.update(
                state="failed", error=str(error), traceback=traceback.format_exc()
            )
        finally:
            status.update(
                finished_at=timestamp(), runtime_seconds=time.monotonic() - start
            )
            write_run_status(run_dir, status)
        print(f"{job['run_id']}: {status['state']}", flush=True)
        return 0 if status["state"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", nargs="?", type=Path)
    parser.add_argument("--dataset", nargs="+")
    parser.add_argument("--horizon", nargs="+", type=int)
    parser.add_argument("--seed", nargs="+", type=int)
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Local device; Slurm visibility is inherited",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs",
        help="Experiment directories",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed or active jobs; retry failures",
    )
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--gpus", default="1", help="Slurm GPU request, e.g. rtx_6000:1")
    parser.add_argument("--qos", default=None)
    parser.add_argument("--exclude", default=None, help="Slurm nodes to avoid")
    parser.add_argument("--execute", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute:
        return execute(args.execute)
    if args.grid is None:
        parser.error("grid is required")
    jobs = enumerate_runs(args.grid, args.dataset, args.horizon, args.seed)
    for job in jobs:
        job["config"]["data"]["data_manager"]["init_args"]["path"] = str(
            args.data_dir.resolve()
        )
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    output = args.output_dir.resolve()
    failures = 0
    for job in jobs:
        run_dir = output / job["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        status_path = run_dir / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if active_job(status) or (
                args.resume and status.get("state") == "completed"
            ):
                print(f"Skipping {job['run_id']}: {status['state']}", flush=True)
                continue
        job.update(
            run_dir=str(run_dir),
            local_gpu=args.gpu,
            sample_count=job["config"]["model"]["forecaster"]["init_args"][
                "num_sampling"
            ],
            precision="32-true; sLSTM kernel bfloat16",
            created_at=timestamp(),
        )
        manifest = run_dir / "job.json"
        write_json(manifest, job)
        command = [
            sys.executable,
            str(ROOT / "ProbTS/scripts/runner.py"),
            "--execute",
            str(manifest),
        ]
        if args.slurm:
            write_run_status(
                run_dir,
                {
                    **{k: v for k, v in job.items() if k != "config"},
                    "state": "submitting",
                },
            )
            submit = [
                "sbatch",
                "--parsable",
                "--gpus",
                args.gpus,
                "--cpus-per-task",
                "10",
                "--mem",
                "64G",
                "--time",
                "2-00:00:00",
                "--job-name",
                "kgo_" + job["run_id"],
                "--output",
                str(run_dir / "slurm-%j.log"),
            ]
            for option in ("partition", "qos", "exclude"):
                if getattr(args, option):
                    submit += [f"--{option}", getattr(args, option)]
            submit += ["--wrap", "exec " + shlex.join(command)]
            result = subprocess.run(submit, text=True, capture_output=True)
            if result.returncode:
                status = {**job, "state": "submission_failed", "error": result.stderr}
                failures += 1
            else:
                status = {
                    **job,
                    "state": "submitted",
                    "slurm_job_id": result.stdout.strip().split(";")[0],
                }
            status.pop("config", None)
            write_run_status(run_dir, status, submission_response=True)
            print(
                f"{job['run_id']}: {status['state']} {status.get('slurm_job_id', status.get('error', ''))}",
                flush=True,
            )
        else:
            failures += subprocess.run(command).returncode != 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
