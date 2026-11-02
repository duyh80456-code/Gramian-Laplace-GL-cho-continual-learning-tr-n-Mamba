"""Read measured outputs only; never synthesize experiment numbers."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('root')
a = p.parse_args()
rows = []
for file in sorted(Path(a.root).rglob('summary.json')):
    r = json.loads(file.read_text())
    rows.append({k: r.get(k) for k in ('arm', 'seed', 'stage', 'strength', 'AA', 'AIA', 'forgetting', 'statistics_floats', 'gpu_hours', 'reimplemented')})
path = Path(a.root) / 'comparison.csv'
if rows:
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)
else:
    print('No completed runs. Paper MambaCL reference: AA 81.67%, forgetting 4.07%; not a reproduction gate.')
