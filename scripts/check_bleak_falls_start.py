#!/usr/bin/env python3
"""Repeat the installed-content bootstrap without any encounter fixtures.

Evidence stays in a caller-selected local directory. This is not a route pass.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(__file__).resolve().parents[1] /
                           'packaging/skyrim-slice.json').read_text())
    reports = []
    for index in (1, 2):
        result = subprocess.run([str(args.probe.resolve()), str(args.data.resolve()),
            '--scenario-start-check', 'skyrim-bleak-falls', '--profile',
            str(args.profile.resolve())], capture_output=True, text=True)
        (args.out / f'start-{index}.json').write_text(result.stdout)
        (args.out / f'start-{index}.log').write_text(result.stderr)
        if result.returncode:
            raise SystemExit(f'Bootstrap check {index} failed; inspect local report/log')
        reports.append(json.loads(result.stdout))
    first, second = reports
    metadata = first['profile']
    enabled_layers = {layer['id']: layer['version'] for layer in metadata['layers']
                      if layer['enabled']}
    required_archives = [(a['file'], a['layer']) for a in manifest['archives']]
    actual_archives = [(a['file'], a['layer']) for a in metadata['archives'] if a['required']]
    checks = {
        'bootstrap_valid': first['ok'] and second['ok'],
        'same_checkpoint': first['checkpoint'] == second['checkpoint'],
        'same_session_hash': first['session_hash'] == second['session_hash'],
        'same_content': first['content_fingerprint'] == second['content_fingerprint'],
        'plugin_order_matches_baseline': metadata['plugins'] == manifest['plugins'],
        'mod_versions_match': all(enabled_layers.get(m['id']) == m['version']
                                  for m in manifest['mods']),
        'required_archives_match': actual_archives == required_archives,
    }
    summary = {'ok': all(checks.values()), 'checks': checks,
               'content_fingerprint': first['content_fingerprint'],
               'release_gate_passed': False,
               'scope': 'Milestone 1 headless bootstrap; runtime player checkpoint still required'}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
