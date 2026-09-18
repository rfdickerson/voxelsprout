#!/usr/bin/env python3
"""List reachable asset candidates for local provenance review, without extracting data."""
import json
from pathlib import PurePosixPath
import subprocess
import sys

objects = subprocess.check_output(['git', 'rev-list', '--objects', '--all'], text=True)
candidates = []
extensions = {'.bin', '.bsa', '.ba2', '.esm', '.esp', '.esl', '.dds', '.nif', '.swf', '.pex',
              '.png', '.jpg', '.jpeg', '.ttf', '.otf', '.wav', '.ogg', '.mp3', '.exe', '.dll', '.smap'}
for line in objects.splitlines():
    oid, _, name = line.partition(' ')
    if name and (PurePosixPath(name).suffix.lower() in extensions or name == 'voxel_skeleton'):
        candidates.append({'object': oid, 'path': name})
json.dump({'scope': 'all reachable refs; filename-based candidates only',
           'provenance_cleared': False, 'candidates': candidates}, sys.stdout, indent=2)
print()
