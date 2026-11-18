"""Create a reproducible file/row inventory without third-party dependencies."""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", default="data")
parser.add_argument("--output", default="results/data_inventory.csv")
args = parser.parse_args()
rows = []
for partition in ("train", "student_test"):
    trips = sorted(p for p in (Path(args.data) / partition).glob("*") if p.is_dir())
    print(f"{partition}: {len(trips)} trips")
    for trip in trips:
        for suffix in ("track", "nodes", "arcs", "route"):
            path = trip / f"{trip.name}.{suffix}"
            exists = path.is_file()
            with path.open() if exists else open('/dev/null') as stream:
                count = sum(bool(line.strip()) for line in stream)
            rows.append([partition, trip.name, suffix, exists,
                         path.stat().st_size if exists else 0, count])
if not rows:
    raise SystemExit("No trips found. Run scripts/import_dataset.py first.")
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["partition", "trip_id", "file_type", "exists", "bytes", "rows"])
    writer.writerows(rows)
print(f"Saved {output}")
