"""Read measured outputs and disclose each run's configured trainable set."""
import argparse
import csv
import json
from pathlib import Path

PARAMETER_SET_NOTE = (
    'MambaCL keeps the pinned upstream trainable set: dt_proj.weight and '
    'dt_proj.bias are frozen. GL and MambaCL+GL additionally train these '
    'parameters. Their comparison with MambaCL changes both the method and '
    'the trainable parameter set; it does not isolate the effect of GL alone.'
)

p = argparse.ArgumentParser()
p.add_argument('root')
a = p.parse_args()
root = Path(a.root)
rows = []
for file in sorted(root.rglob('summary.json')):
    r = json.loads(file.read_text())
    row = {k: r.get(k) for k in ('arm', 'seed', 'stage', 'strength', 'AA', 'AIA', 'forgetting', 'statistics_floats', 'gpu_hours', 'reimplemented')}
    identity_path = file.parent / 'identity.json'
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    config = identity.get('config', {})
    row['backbone']=config.get('backbone','large')
    key = 'baseline_training_string' if r.get('arm') == 'MambaCL' else 'training_string'
    patterns = config.get(key)
    # Read the recorded configuration of each run, not today's working config.
    # This follows the upstream substring rule in set_model_mode().
    if patterns is None:
        row.update(training_string='UNKNOWN', dt_proj_weight='UNKNOWN', dt_proj_bias='UNKNOWN')
    else:
        row['training_string'] = ';'.join(patterns)
        for suffix in ('weight', 'bias'):
            name = 'blocks.0.mixer.dt_proj.' + suffix
            row['dt_proj_' + suffix] = 'trainable' if any(pattern in name for pattern in patterns) else 'frozen'
    row['parameter_scope_source'] = 'recorded identity.json config' if patterns is not None else 'missing recorded configuration'
    row['parameter_set_note'] = PARAMETER_SET_NOTE
    rows.append(row)

paper_note=('Base classification backbone: different architecture/pretraining from the paper large21K reference; AA81.67 is not a base reproduction target.'
            if any(row.get('backbone')=='base' for row in rows) else
            'Paper large21K references: AA81.67%, forgetting4.07%; reference numbers, not measured results or a stopping gate.')
if rows:
    path = root / 'comparison.csv'
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# Stage-one results', '', PARAMETER_SET_NOTE, '',
        paper_note, '',
        'Trainable scopes below come from each run\'s recorded identity.json. '
        'UNKNOWN means the original configuration is missing; no scope is inferred.', '',
        '| Arm | Seed | Stage | AA (%) | AIA (%) | Forgetting (%) | dt_proj weight | dt_proj bias | Training patterns |',
        '| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |',
    ]
    for row in rows:
        values = [row[k] for k in ('arm', 'seed', 'stage', 'AA', 'AIA', 'forgetting', 'dt_proj_weight', 'dt_proj_bias', 'training_string')]
        lines.append('| ' + ' | '.join(str(value) if value is not None else 'UNKNOWN' for value in values) + ' |')
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(path)
    print(root / 'comparison.md')
else:
    print('No completed runs; no measured accuracy yet.')
    print(PARAMETER_SET_NOTE)
