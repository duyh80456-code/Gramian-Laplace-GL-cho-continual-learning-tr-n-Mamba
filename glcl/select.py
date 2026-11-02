"""Select independent GL/hybrid lambda using validation AIA of first 3 tasks."""
import argparse
import json
from pathlib import Path
from .io import atomic_json, digest_sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/phase1.json').read_text())
    scores, identities = {}, []
    for arm in ('GL', 'MambaCL+GL'):
        scores[arm] = {}
        for strength in config['lambda_grid']:
            run = Path(args.root) / f'{arm}_lambda{strength:g}'
            summary = json.loads((run / 'summary.json').read_text())
            identity = json.loads((run / 'identity.json').read_text())
            if not (identity['source_sha256'] == digest_sources() and summary['arm'] == arm and summary['seed'] == 0 and summary['stage'] == 'tune' and summary['strength'] == strength and len(summary['accuracy_matrix']) == 3):
                raise RuntimeError('Incomplete or mismatched tuning run: ' + str(run))
            identities.append(identity)
            scores[arm][strength] = summary['AIA']
    for key in ('dataset_manifest', 'pretrained_sha256', 'config'):
        if any(i[key] != identities[0][key] for i in identities):
            raise RuntimeError('Tuning candidates use different ' + key)
    selected = {arm: max(values, key=lambda strength: (values[strength], -strength)) for arm, values in scores.items()}
    atomic_json(dict(source_sha256=digest_sources(), selected=selected, validation_AIA=scores,
                     criterion='max validation AIA after first 3 tasks; ties use lower lambda',
                     dataset_manifest=identities[0]['dataset_manifest'], pretrained_sha256=identities[0]['pretrained_sha256']), args.out)

if __name__ == '__main__':
    main()
