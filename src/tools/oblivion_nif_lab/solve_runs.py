#!/usr/bin/env python3
"""Derive per-type block sizes by solving for RUNS, then decomposing.

The Havok blocks never appear alone -- bhkCollisionObject, bhkRigidBody,
bhkMoppBvTreeShape and bhkNiTriStripsShape arrive as one consecutive chain --
so searching each type independently finds nothing and searching all of them at
once is exponential (four consecutive unknowns at 256 candidates each).

The way through: a maximal RUN of consecutive unknown blocks has exactly one
free variable, its total length. Find that by walking the rest of the file and
requiring the footer to land. Each file then yields one linear equation

    sum over types in the run of (count * size) = total

and enough files with different run compositions determine the individual
sizes. A type whose equations cannot be satisfied by any single value is
variable-length, which is a finding rather than a failure.

    python3 solve_runs.py [<Oblivion Data dir>]
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
MAX_RUN_BYTES = 65536


def walk_known(data, header, sequence, index, stop, pos, known):
    while index < stop:
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
        if index < len(sequence) and not N.plausible_block_start(data, pos):
            return None
    return pos


def footer_fits(data, pos):
    if pos + 4 > len(data):
        return False
    roots = struct.unpack_from("<I", data, pos)[0]
    return roots < 64 and pos + 4 + 4 * roots == len(data)


def solve_run(data, header, known):
    """-> (Counter of types in the single unknown run, total bytes) or None."""
    sequence = [header.types[i] for i in header.type_index]
    unknown = [i for i, t in enumerate(sequence) if t not in known]
    if not unknown:
        return None
    # One maximal contiguous run only; more than one leaves two free variables.
    first, last = unknown[0], unknown[-1]
    if set(unknown) != set(range(first, last + 1)):
        return None

    cursor = N.Cur(data)
    N.parse_header(cursor)
    start = walk_known(data, header, sequence, 0, first, cursor.p, known)
    if start is None:
        return None

    hits = []
    for total in range(0, MAX_RUN_BYTES + 1, SIZE_STEP):
        end = start + total
        if end > len(data):
            break
        if last + 1 < len(sequence) and not N.plausible_block_start(data, end):
            continue
        final = walk_known(data, header, sequence, last + 1, len(sequence), end, known)
        if final is not None and footer_fits(data, final):
            hits.append(total)
            if len(hits) > 1:
                return None
    if len(hits) == 1:
        return collections.Counter(sequence[first:last + 1]), hits[0]
    return None


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    archive = Bsa(os.path.join(data_dir, "Oblivion - Meshes.bsa"))
    known = dict(N.READERS)

    equations = []
    for name in sorted(archive.names(".nif")):
        blob = archive.read(name)
        if not blob.startswith(b"Gamebryo File Format, Version 20.0.0"):
            continue
        try:
            header = N.parse_header(N.Cur(blob))
        except Exception:
            continue
        got = solve_run(blob, header, known)
        if got:
            equations.append((got[0], got[1], name))

    print(f"solved {len(equations)} single-run files\n")
    if not equations:
        return

    # Group by run composition; a composition seen with one consistent total is
    # a usable equation.
    by_shape = collections.defaultdict(collections.Counter)
    for counts, total, _ in equations:
        key = tuple(sorted(counts.items()))
        by_shape[key][total] += 1

    print("run compositions, by frequency:")
    for key, totals in sorted(by_shape.items(), key=lambda kv: -sum(kv[1].values()))[:14]:
        shape = " + ".join(f"{n}x{t}" for t, n in key)
        n_files = sum(totals.values())
        spread = "one total" if len(totals) == 1 else f"{len(totals)} totals"
        common = ", ".join(f"{v}B x{c}" for v, c in totals.most_common(3))
        print(f"  {n_files:5d} files  [{spread}]  {shape}")
        print(f"            {common}")

    # Any composition that is a single type with a single total gives that
    # type's size outright.
    print("\ndirectly determined:")
    for key, totals in by_shape.items():
        if len(key) == 1 and len(totals) == 1:
            (t, n), = key
            total, = totals
            if total % n == 0:
                print(f"  {t} = {total // n} bytes  (from {n}x runs)")


if __name__ == "__main__":
    main()
