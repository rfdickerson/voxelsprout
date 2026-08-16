#!/usr/bin/env python3
"""Solve one unknown block type at a time, from files that contain exactly one
unknown block INSTANCE.

The general solver searches every unknown type at once and is exponential. This
one is linear: find a file where all but a single block can already be walked,
and that block's size is the only free variable. Walk forward to it, try each
candidate size, and keep the ones that let the rest of the file land exactly on
the footer.

Bootstrapping this way -- solve the type that appears alone, add it to the
readers, re-run to expose the next one -- is what makes the Havok set tractable.

    python3 solve_one.py [<Oblivion Data dir>]
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
SIZE_STEP = 4
MAX_SIZE = 4096


def walk_from(data, header, sequence, index, pos, known):
    """Walk known blocks from `index`; -> end position, or None on failure."""
    n = len(sequence)
    while index < n:
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
        index += 1
        if index < n and not N.plausible_block_start(data, pos):
            return None
    return pos


def footer_fits(data, pos):
    if pos + 4 > len(data):
        return False
    roots = struct.unpack_from("<I", data, pos)[0]
    return roots < 64 and pos + 4 + 4 * roots == len(data)


def solve(data, header, known):
    """-> (type, size) when exactly one unknown instance has a unique size."""
    sequence = [header.types[i] for i in header.type_index]
    unknown_at = [i for i, t in enumerate(sequence) if t not in known]
    if len(unknown_at) != 1:
        return None
    at = unknown_at[0]

    pos = N.Cur(data, 0)
    N.parse_header(pos)
    start = walk_from(data, header, sequence[:at], 0, pos.p, known)
    if start is None:
        return None

    hits = []
    for size in range(0, MAX_SIZE + 1, SIZE_STEP):
        end = start + size
        if end > len(data):
            break
        if at + 1 < len(sequence) and not N.plausible_block_start(data, end):
            continue
        final = walk_from(data, header, sequence, at + 1, end, known)
        if final is not None and footer_fits(data, final):
            hits.append(size)
            if len(hits) > 1:
                return None  # ambiguous
    if len(hits) == 1:
        return sequence[at], hits[0]
    return None


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    archive = Bsa(os.path.join(data_dir, "Oblivion - Meshes.bsa"))
    known = dict(N.READERS)

    observed = collections.defaultdict(collections.Counter)
    examples = {}
    candidates = 0
    for name in sorted(archive.names(".nif")):
        blob = archive.read(name)
        if not blob.startswith(b"Gamebryo File Format, Version 20.0.0"):
            continue
        try:
            header = N.parse_header(N.Cur(blob))
        except Exception:
            continue
        sequence = [header.types[i] for i in header.type_index]
        if sum(1 for t in sequence if t not in known) != 1:
            continue
        candidates += 1
        got = solve(blob, header, known)
        if got:
            observed[got[0]][got[1]] += 1
            examples.setdefault(got[0], name)

    print(f"files with exactly one unknown block instance: {candidates}")
    if not observed:
        print("none solved uniquely")
        return
    print()
    for t in sorted(observed, key=lambda k: -sum(observed[k].values())):
        counts = observed[t]
        total = sum(counts.values())
        kind = "FIXED" if len(counts) == 1 else f"VARIABLE ({len(counts)} distinct)"
        print(f"{t:<30} {total:4d} solved   {kind}")
        for size, n in counts.most_common(6):
            print(f"       {size:5d} bytes  x{n}")
        print(f"       e.g. {examples[t]}")


if __name__ == "__main__":
    main()
