"""Prepare exactly the nine CSV and four GluonTS datasets used by KGO."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

import gdown

CSV_FILES = (
    "ETT-small/ETTh1.csv",
    "ETT-small/ETTh2.csv",
    "ETT-small/ETTm1.csv",
    "ETT-small/ETTm2.csv",
    "electricity/electricity.csv",
    "traffic/traffic.csv",
    "exchange_rate/exchange_rate.csv",
    "weather/weather.csv",
    "illness/national_illness.csv",
)
NIPS_DATASETS = ("electricity_nips", "traffic_nips", "exchange_rate_nips", "solar_nips")
ARCHIVE_ID = "1tSc1WA30CL2aMt5hAW7M-d5_0IBz-lJP"


def extract_csvs(archive, destination):
    """Validate archive paths and stage only the required benchmark CSVs."""
    destination = Path(destination)
    with zipfile.ZipFile(archive) as source:
        members = {}
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError(f"Unsafe archive member: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink is not supported: {info.filename}")
            for relative in CSV_FILES:
                if info.filename == relative or info.filename.endswith("/" + relative):
                    if relative in members:
                        raise ValueError(f"Duplicate dataset in archive: {relative}")
                    members[relative] = info
        missing = set(CSV_FILES) - members.keys()
        if missing:
            raise ValueError(f"Archive is missing datasets: {sorted(missing)}")
        for relative, info in members.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            if not target.stat().st_size:
                raise ValueError(f"Empty dataset: {relative}")


def nips_complete(path):
    return (
        (path / "metadata.json").is_file()
        and any((path / "train").glob("*.json*"))
        and any((path / "test").glob("*.json*"))
    )


def prepare(destination, include_nips=True):
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing = [
            name
            for name in CSV_FILES
            if not (destination / name).is_file()
            or not (destination / name).stat().st_size
        ]
        if missing:
            with tempfile.TemporaryDirectory(
                prefix=".download-", dir=destination, ignore_cleanup_errors=True
            ) as temp:
                stage = Path(temp)
                archive = stage / "all_datasets.zip"
                result = gdown.download(id=ARCHIVE_ID, output=str(archive), quiet=False)
                if result is None or not zipfile.is_zipfile(archive):
                    raise RuntimeError(
                        "Dataset download did not produce a valid ZIP archive"
                    )
                extract_csvs(archive, stage / "extracted")
                for relative in missing:
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(stage / "extracted" / relative, target)
        print("All nine LTSF CSV datasets are present.", flush=True)
        if include_nips:
            from gluonts.dataset.repository.datasets import get_dataset

            for name in NIPS_DATASETS:
                target = destination / name
                if nips_complete(target):
                    print(f"Already prepared: {name}", flush=True)
                    continue
                if target.exists():
                    raise RuntimeError(
                        f"Incomplete dataset at {target}; move it aside before retrying"
                    )
                with tempfile.TemporaryDirectory(
                    prefix=".download-", dir=destination, ignore_cleanup_errors=True
                ) as temp:
                    stage = Path(temp)
                    get_dataset(name, path=stage)
                    if not nips_complete(stage / name):
                        raise RuntimeError(f"Incomplete GluonTS download: {name}")
                    os.replace(stage / name, target)
                print(f"Prepared: {name}", flush=True)
        files = [destination / name for name in CSV_FILES]
        if include_nips:
            for name in NIPS_DATASETS:
                files.extend(p for p in (destination / name).rglob("*") if p.is_file())
        manifest = {}
        for path in sorted(files):
            digest = hashlib.sha256()
            with path.open("rb") as data:
                for block in iter(lambda: data.read(1024 * 1024), b""):
                    digest.update(block)
            manifest[str(path.relative_to(destination))] = {
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        manifest_path = destination / "manifest.json"
        if not manifest_path.exists() or manifest_path.read_text() != manifest_text:
            manifest_path.write_text(manifest_text)
        print(f"Dataset checksums: {manifest_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--ltsf-only", action="store_true")
    args = parser.parse_args()
    prepare(args.data_path, include_nips=not args.ltsf_only)
