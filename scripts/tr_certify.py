#!/usr/bin/env python3
"""Prepare isolated A/B TR profiles and regenerate local Partial certification.

This orchestrates structural audits, never player progression or gameplay
certification. No IDs or asset files are embedded in or copied by the runtime.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from tes3_release_audit import write


def profiles(base, data, mainland, factions, output):
    for label in ('A', 'B'):
        roots = [data, mainland] + ([factions] if label == 'B' else [])
        plugins = ['Morrowind.esm', 'Tribunal.esm', 'Bloodmoon.esm', 'Tamriel_Data.esm', 'TR_Mainland.esm']
        if label == 'B':
            plugins.append('TR_Factions.esp')
        write(output / f'profile-{label}.json', {
            'version': 1, 'name': 'TR-certification-' + label, 'game': 'morrowind',
            'data_root': str(base.resolve()), 'encoding': 'windows-1252',
            'layers': [{'id': f'layer-{i}', 'path': str(root.resolve())} for i, root in enumerate(roots)],
            'plugins': plugins, 'archives': [{'path': str((base / name).resolve()), 'required': True}
                                            for name in ('Morrowind.bsa', 'Tribunal.bsa', 'Bloodmoon.bsa')]})


def delta(output):
    a, b = [json.loads((output / label / 'winning-record-manifest.json').read_text()) for label in ('A', 'B')]
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(key for key in a.keys() & b.keys() if a[key] != b[key])
    unexpected = [key for key in added + changed if b[key]['plugin'].lower() != 'tr_factions.esp']
    reports = [json.loads((output / label / 'certification.json').read_text()) for label in ('A', 'B')]
    qa, qb = [set(r['verification']['quests']['non_verified_content_ids']) for r in reports]
    result = {'record_changes_match_optional_plugin_provenance': not unexpected and not removed,
              'behavior_verified': False, 'added_records': added, 'removed_records': removed,
              'changed_records': changed, 'unexpected_changes': unexpected,
              'added_quest_candidates': sorted(qb - qa), 'removed_quest_candidates': sorted(qa - qb),
              'profile_counts': {label: r['counts'] for label, r in zip(('A', 'B'), reports)}}
    write(output / 'optional-profile-delta.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--mainland-root', type=Path, required=True)
    parser.add_argument('--factions-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('captures/tr-certification'))
    parser.add_argument('--probe', type=Path, default=Path('build-linux/odai_bethesda_probe'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profiles(args.base_root, args.data_root, args.mainland_root, args.factions_root, args.output)
    for label in ('A', 'B'):
        command = [sys.executable, str(Path(__file__).with_name('tes3_release_audit.py')),
                   '--profile', str(args.output / f'profile-{label}.json'), '--scope-plugin', 'TR_Mainland.esm',
                   '--output', str(args.output / label), '--probe', str(args.probe)]
        if label == 'B':
            command.extend(['--scope-plugin', 'TR_Factions.esp'])
        result = subprocess.run(command)
        if result.returncode != 1:  # 1 means an explicitly uncertified Partial report.
            print('Audit failed before producing the expected Partial report', file=sys.stderr)
            return 2
    result = delta(args.output)
    print(json.dumps({'status': 'Partial', 'profile_counts': result['profile_counts']}))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
