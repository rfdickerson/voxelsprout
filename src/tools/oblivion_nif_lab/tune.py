#!/usr/bin/env python3
"""Find the Havok padding sizes by sweeping them against the whole archive.

The Havok layouts in nif20004.py carry their structure but leave a few fixed
"unused" runs as tunables (nif20004.PARAMS), because those are exactly the parts
that shift between versions. This sweeps each one and keeps whatever walks the
most files to their footer -- coordinate descent over four small integers.

The score is the number of files whose EVERY block is consumed exactly and whose
remainder is exactly the footer. That is a hard oracle: a wrong padding does not
score slightly lower, it scores near zero, so the optimum is unmistakable.

    python3 tune.py [<Oblivion Data dir>] [sample-size]
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bsa import Bsa  # noqa: E402
import nif20004 as N  # noqa: E402

DEFAULT_DATA = os.path.expanduser(
    "~/.local/share/Steam/steamapps/common/Oblivion/Data")

SWEEPS = {
    "rigid_body_head": range(0, 289, 4),
    "tristrips_mid": range(0, 97, 4),
    "mopp_mid": range(0, 65, 4),
    "convex_mid": range(0, 65, 4),
    "mopp_tail": range(0, 33, 4),
}


def score(blobs, only_types=None):
    """BLOCKS consumed before desync, summed over the sample.

    Not "files fully walked": with four coupled paddings, no file walks fully
    until all four are right, so that objective is flat at zero everywhere and
    coordinate descent cannot start. Depth is monotone in each parameter
    separately -- fixing the FIRST Havok type a file hits lets the walk reach the
    second, and so on -- which is what makes sweeping them one at a time work.
    """
    total = 0
    for blob, types in blobs:
        if only_types and not (types & only_types):
            continue
        ok, _, _, done = N.walk(blob)
        total += done + (1 if ok else 0)
    return total


def files_walked(blobs):
    return sum(1 for blob, _ in blobs if N.walk(blob)[0])


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    archive = Bsa(os.path.join(data_dir, "Oblivion - Meshes.bsa"))

    names = sorted(archive.names(".nif"))
    random.seed(11)
    random.shuffle(names)
    blobs = []
    for name in names:
        if len(blobs) >= sample_size:
            break
        blob = archive.read(name)
        if not blob.startswith(b"Gamebryo File Format, Version 20.0.0"):
            continue
        try:
            header = N.parse_header(N.Cur(blob))
        except Exception:
            continue
        blobs.append((blob, {header.types[i] for i in header.type_index}))
    print(f"sample: {len(blobs)} files\n")

    # Which files exercise which tunable -- sweeping a padding against files that
    # never contain its block is pure noise.
    gate = {
        "rigid_body_head": {"bhkRigidBody", "bhkRigidBodyT"},
        "tristrips_mid": {"bhkNiTriStripsShape"},
        "mopp_mid": {"bhkMoppBvTreeShape"},
        "convex_mid": {"bhkConvexVerticesShape"},
        "mopp_tail": {"bhkMoppBvTreeShape"},
    }

    print(f"baseline: {files_walked(blobs)} files walk, depth {score(blobs)}\n")
    for round_index in range(3):
        improved = False
        for key, values in SWEEPS.items():
            start = N.PARAMS[key]
            results = []
            for value in values:
                N.PARAMS[key] = value
                results.append((score(blobs, gate[key]), value))
            results.sort(key=lambda r: (-r[0], r[1]))
            top_score, top_value = results[0]
            N.PARAMS[key] = top_value
            gated_start = None
            if top_value != start:
                improved = True
            runners = ", ".join(f"{v}->{s}" for s, v in results[:4])
            print(f"round {round_index} {key:<20} best {top_value:4d} "
                  f"({top_score} files)   top: {runners}")
            del gated_start
        print(f"  -> {files_walked(blobs)} files walk, depth {score(blobs)}\n")
        if not improved:
            break

    print("final:", N.PARAMS)
    print("files fully walked:", files_walked(blobs), "/", len(blobs))


if __name__ == "__main__":
    main()
