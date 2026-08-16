#!/usr/bin/env python3
"""Derive the byte size of unknown NIF block types from retail data.

A sequential 20.0.0.4 reader only needs to SKIP the Havok blocks, so their
layouts do not have to be understood -- only their lengths. That turns a
reverse-engineering problem into a search problem, and the walker already has
the two oracles a search needs: the next block must open with a plausible
SizedString, and after the last block the remainder must be exactly the footer.

So: guess a size for each unknown type, walk, keep the assignment that lands on
the footer. Run it over many files and a FIXED-size type produces the same
answer every time; a variable-size one does not, which is itself the finding.

    python3 solve_sizes.py [<Oblivion Data dir>] [max-files]
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

# Havok blocks are 4-byte aligned, so stepping by 4 is not an approximation.
SIZE_STEP = 4
MAX_SIZE = 1024


def solve_file(data, header, known):
    """One consistent type->size assignment that walks the whole file, or None.

    Depth-first over blocks, carrying a partial assignment. The plausibility
    check prunes almost every wrong candidate at the very next block, which is
    what keeps this from exploding.
    """
    sequence = [header.types[i] for i in header.type_index]
    n = len(sequence)
    footer_ok = {}

    def remainder_is_footer(pos):
        if pos in footer_ok:
            return footer_ok[pos]
        ok = False
        if pos + 4 <= len(data):
            roots = struct.unpack_from("<I", data, pos)[0]
            ok = roots < 64 and pos + 4 + 4 * roots == len(data)
        footer_ok[pos] = ok
        return ok

    solutions = []

    def walk(index, pos, assignment):
        if len(solutions) > 1:
            return  # ambiguous already; no point continuing
        if index == n:
            if remainder_is_footer(pos):
                solutions.append(dict(assignment))
            return
        name = sequence[index]
        reader = known.get(name)
        if reader is not None:
            cursor = N.Cur(data, pos)
            try:
                reader(cursor, header)
            except Exception:
                return
            end = cursor.p
            if end > len(data):
                return
            if index + 1 < n and not N.plausible_block_start(data, end):
                return
            walk(index + 1, end, assignment)
            return
        already_bound = name in assignment
        candidates = ([assignment[name]] if already_bound
                      else range(0, MAX_SIZE + 1, SIZE_STEP))
        for size in candidates:
            end = pos + size
            if end > len(data):
                break
            if index + 1 < n and not N.plausible_block_start(data, end):
                continue
            if not already_bound:
                assignment[name] = size
            walk(index + 1, end, assignment)
            if not already_bound:
                # Only unbind what THIS frame bound; a size inherited from an
                # earlier occurrence of the same type has to survive backtracking
                # or the assignment stops being consistent across the file.
                assignment.pop(name, None)
            if len(solutions) > 1:
                return
        return

    walk(0, header_end(data), {})
    if len(solutions) == 1:
        return solutions[0]
    return None


def header_end(data):
    c = N.Cur(data)
    N.parse_header(c)
    return c.p


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    archive = Bsa(os.path.join(data_dir, "Oblivion - Meshes.bsa"))
    known = dict(N.READERS)

    observed = collections.defaultdict(collections.Counter)
    solved = attempted = 0
    for name in sorted(archive.names(".nif")):
        if attempted >= limit:
            break
        blob = archive.read(name)
        if not blob.startswith(b"Gamebryo File Format, Version 20.0.0"):
            continue
        try:
            header = N.parse_header(N.Cur(blob))
        except Exception:
            continue
        missing = {t for t in
                   (header.types[i] for i in header.type_index) if t not in known}
        # Bound the search: every extra unknown type multiplies the space.
        if not missing or len(missing) > 3:
            continue
        attempted += 1
        result = solve_file(blob, header, known)
        if result:
            solved += 1
            for t, size in result.items():
                observed[t][size] += 1

    print(f"attempted {attempted} files, uniquely solved {solved}\n")
    for t in sorted(observed, key=lambda k: -sum(observed[k].values())):
        counts = observed[t]
        total = sum(counts.values())
        top = counts.most_common(4)
        verdict = "FIXED" if len(counts) == 1 else f"variable ({len(counts)} sizes)"
        print(f"{t:<28} {total:5d} samples  {verdict}")
        for size, n in top:
            print(f"      {size:5d} bytes  x{n}")


if __name__ == "__main__":
    main()
