"""Import only known development partitions from the organiser ZIP."""
import argparse
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("zip_path")
parser.add_argument("--destination", default="data")
args = parser.parse_args()
destination = Path(args.destination)
count = 0
with ZipFile(args.zip_path) as archive:
    for info in archive.infolist():
        parts = PurePosixPath(info.filename).parts
        if info.is_dir() or len(parts) != 4:
            continue
        root, partition, trip_id, filename = parts
        if root != "Student folder" or partition not in {"train", "student_test"}:
            continue
        if not trip_id.isdigit() or filename not in {
            f"{trip_id}.{suffix}" for suffix in ("track", "nodes", "arcs", "route")
        }:
            continue
        target = destination / partition / trip_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise SystemExit(f"Already exists: {target}. Import into a fresh destination.")
        target.write_bytes(archive.read(info))
        count += 1
if not count:
    raise SystemExit("No recognised development files found; check ZIP structure.")
print(f"Imported {count} files into {destination.resolve()}")
