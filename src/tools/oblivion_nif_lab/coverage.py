#!/usr/bin/env python3
"""Which Oblivion NIFs would a sequential reader be able to walk today?

Run from anywhere:
    python3 src/tools/oblivion_nif_lab/coverage.py [<Oblivion Data dir>]

Reports, over every 20.0.0.x mesh in the archive: how many files have a reader
for every block type they contain, which missing types block the most files,
and which SETS of missing types travel together (that last one is what tells you
which four layouts to write next, rather than which one).
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bsa import Bsa  # noqa: E402
import nif20004 as N  # noqa: E402

DEFAULT_DATA = os.path.expanduser(
    "~/.local/share/Steam/steamapps/common/Oblivion/Data")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    archive = Bsa(os.path.join(data_dir, "Oblivion - Meshes.bsa"))
    known = set(N.READERS)
    geometry = {"NiTriShape", "NiTriStrips"}
    geometry_data = {"NiTriShapeData", "NiTriStripsData"}

    total = walkable = 0
    blocking = collections.Counter()
    sets = collections.Counter()
    prefix_stats = collections.Counter()

    for name in sorted(archive.names(".nif")):
        blob = archive.read(name)
        if not blob.startswith(b"Gamebryo File Format, Version 20.0.0"):
            continue
        try:
            header = N.parse_header(N.Cur(blob))
        except Exception:
            blocking["(header parse failed)"] += 1
            continue
        total += 1
        sequence = [header.types[i] for i in header.type_index]
        missing = {t for t in sequence if t not in known}
        if not missing:
            walkable += 1
        else:
            sets[frozenset(missing)] += 1
            for t in missing:
                blocking[t] += 1

        # Is there an escape hatch in stopping at the first unknown block?
        first_unknown = next(
            (i for i, t in enumerate(sequence) if t not in known), len(sequence))
        prefix = sequence[:first_unknown]
        total_geometry = sum(1 for t in sequence if t in geometry)
        prefix_geometry = sum(1 for t in prefix if t in geometry)
        prefix_data = sum(1 for t in prefix if t in geometry_data)
        if total_geometry == 0:
            prefix_stats["no geometry at all"] += 1
        elif prefix_geometry == total_geometry and prefix_data:
            prefix_stats["all geometry before first unknown"] += 1
        elif prefix_geometry and prefix_data:
            prefix_stats["some geometry before first unknown"] += 1
        else:
            prefix_stats["NO geometry before first unknown"] += 1

    print(f"20.0.0.x meshes: {total}")
    print(f"every block type has a reader: {walkable} ({100.0 * walkable / total:.1f}%)")
    print("\nmissing types, by files blocked:")
    for name, count in blocking.most_common(15):
        print(f"  {count:5d}  {name}")
    print("\nmissing-type sets that travel together:")
    for names, count in sets.most_common(6):
        print(f"  {count:5d}  {sorted(names)}")
    print("\nis 'stop at the first unknown block' an escape hatch?")
    for name, count in prefix_stats.most_common():
        print(f"  {count:5d} ({100.0 * count / total:5.1f}%)  {name}")


if __name__ == "__main__":
    main()
