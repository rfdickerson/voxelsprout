#!/usr/bin/env python3
"""Validate an installed engine package without installed game data."""
from pathlib import Path
import subprocess
import shutil
import argparse
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('package', type=Path)
parser.add_argument('--smoke', type=Path, help='Synthetic renderer test executable; never shipped')
args = parser.parse_args()
root = args.package.resolve()
required = ['bin/odai', 'bin/odai-launcher', 'share/odai/skyrim-slice.json',
            'share/odai/assets/fonts/Inter-Regular.ttf', 'share/doc/odai/LICENSE',
            'share/doc/odai/THIRD_PARTY_NOTICES.md']
for name in required:
    if not (root / name).is_file():
        raise SystemExit(f'Missing package resource: {name}')
shaders = sorted((root / 'share/odai/shaders').glob('*.spv'))
if not shaders:
    raise SystemExit('Package has no shaders')
for path in shaders:
    data = path.read_bytes()
    if len(data) % 4 or data[:4] != b'\x03\x02\x23\x07':
        raise SystemExit(f'Invalid SPIR-V: {path.name}')
for path in root.rglob('*'):
    if path.suffix.lower() in {'.bsa', '.esm', '.esp', '.esl', '.nif', '.dds', '.pex', '.swf', '.ess', '.bin'}:
        raise SystemExit(f'Unexpected content asset: {path.relative_to(root)}')
with tempfile.TemporaryDirectory() as cwd:
    subprocess.run([str(root / 'bin/odai'), '--version'], cwd=cwd, check=True)
    subprocess.run([str(root / 'bin/odai'), '--help'], cwd=cwd, check=True, stdout=subprocess.DEVNULL)
print(f'Package layout verified: {len(shaders)} shaders; no proprietary-format assets')

if args.smoke:
    target = root / 'bin/odai-package-smoke'
    if target.exists():
        raise SystemExit('Refusing to replace an existing package smoke executable')
    try:
        shutil.copy2(args.smoke.resolve(), target)
        with tempfile.TemporaryDirectory() as cwd:
            subprocess.run([str(target)], cwd=cwd, check=True, timeout=60)
    finally:
        target.unlink(missing_ok=True)
