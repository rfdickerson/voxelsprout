#!/usr/bin/env python3
"""Derive Havok block sizes from files whose unknown blocks are a SUFFIX.

No search at all. If every block after the first unknown one is also unknown,
the run runs to the footer, so

    run total = len(file) - footer bytes - run start

and the run start comes from walking the known prefix. Each such file is one
exact linear equation over the types in its run, obtained in O(1).

That is the whole trick: brute-forcing a run's length costs thousands of walks
per file and the archive has 7000 files; this costs one walk and there are
plenty of suffix files, because Bethesda's exporter emits a mesh's collision
after its geometry.

Solving the resulting system is then ordinary linear algebra over the integers,
done here by elimination on equations that differ by one type.

    python3 solve_suffix.py [<Oblivion Data dir>]
"""
import collections
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bsa import Bsa  # noqa: E402
import nif20004 as N  # noqa: E402

DEFAULT_DATA = os.path.expanduser(
    "~/.local/share/Steam/steamapps/common/Oblivion/Data")


def walk_known(data, header, sequence, stop, pos, known):
    for index in range(stop):
        reader = known.get(sequence[index])
        if reader is None:
            return None
        cursor = N.Cur(data, pos)
        try:
            reader(cursor, header)
        except Exception:
            return None
        pos = cursor.p
        if pos > len(data):
            return None
        if not N.plausible_block_start(data, pos):
            return None
    return pos


def footer_bytes(data):
    """Footer is numRoots + roots; its length is fixed by the last 4-byte count
    only if we know where it starts, so instead try every plausible root count."""
    for roots in range(0, 64):
        size = 4 + 4 * roots
        start = len(data) - size
        if start < 0:
            break
        if struct.unpack_from("<I", data, start)[0] == roots:
            return size
    return None


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    archive = Bsa(os.path.join(data_dir, "Oblivion - Meshes.bsa"))
    known = dict(N.READERS)

    equations = collections.defaultdict(collections.Counter)
    total_files = 0
    for name in sorted(archive.names(".nif")):
        blob = archive.read(name)
        if not blob.startswith(b"Gamebryo File Format, Version 20.0.0"):
            continue
        try:
            header = N.parse_header(N.Cur(blob))
        except Exception:
            continue
        sequence = [header.types[i] for i in header.type_index]
        unknown = [i for i, t in enumerate(sequence) if t not in known]
        if not unknown:
            continue
        first = unknown[0]
        # Suffix: everything from `first` on is unknown.
        if set(unknown) != set(range(first, len(sequence))):
            continue
        cursor = N.Cur(blob)
        N.parse_header(cursor)
        start = walk_known(blob, header, sequence, first, cursor.p, known)
        if start is None:
            continue
        fb = footer_bytes(blob)
        if fb is None:
            continue
        total = len(blob) - fb - start
        if total < 0:
            continue
        total_files += 1
        key = tuple(sorted(collections.Counter(sequence[first:]).items()))
        equations[key][total] += 1

    print(f"suffix-run files solved exactly: {total_files}\n")
    print("run composition -> total bytes")
    singles = {}
    for key, totals in sorted(equations.items(), key=lambda kv: -sum(kv[1].values())):
        shape = " + ".join(f"{n}x{t}" for t, n in key)
        n_files = sum(totals.values())
        if len(totals) == 1:
            total, = totals
            mark = "exact"
            if len(key) == 1:
                (t, n), = key
                if total % n == 0:
                    singles[t] = total // n
                    mark = f"=> {t} = {total // n} bytes"
            print(f"  {n_files:5d} files  {total:7d} B  [{mark}]  {shape}")
        else:
            common = ", ".join(f"{v}B x{c}" for v, c in totals.most_common(3))
            print(f"  {n_files:5d} files  VARIABLE ({len(totals)} totals: {common})  {shape}")

    if singles:
        print("\ndirectly determined fixed sizes:")
        for t, s in sorted(singles.items()):
            print(f"  {t:<28} {s} bytes")


if __name__ == "__main__":
    main()
