#!/usr/bin/env python3
"""Compare deterministic P6 screenshots against reviewed baselines."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def read_ppm(path):
    with Path(path).open("rb") as stream:
        def token():
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError(f"{path}: truncated PPM header")
                if char == b"#":
                    stream.readline()
                elif not char.isspace():
                    break
            result = bytearray(char)
            while True:
                char = stream.read(1)
                if not char or char.isspace():
                    return bytes(result)
                result.extend(char)

        if token() != b"P6":
            raise ValueError(f"{path}: expected binary P6 PPM")
        width, height, maximum = (int(token()) for _ in range(3))
        if width < 1 or height < 1 or width * height > 100_000_000 or maximum != 255:
            raise ValueError(f"{path}: unsupported dimensions or channel maximum")
        pixels = stream.read()
        if len(pixels) != width * height * 3:
            raise ValueError(f"{path}: expected {width * height * 3} pixel bytes, got {len(pixels)}")
        return width, height, pixels


def write_ppm(path, width, height, pixels):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(f"P6\n{width} {height}\n255\n".encode() + pixels)


def compare(actual_path, baseline_path, diff_path, max_changed_fraction, channel_tolerance):
    width, height, actual = read_ppm(actual_path)
    base_width, base_height, baseline = read_ppm(baseline_path)
    if (width, height) != (base_width, base_height):
        raise ValueError(f"dimensions differ: actual {width}x{height}, baseline {base_width}x{base_height}")
    changed = 0
    largest_delta = 0
    diff = bytearray(len(actual))
    for index in range(0, len(actual), 3):
        delta = max(abs(actual[index + channel] - baseline[index + channel]) for channel in range(3))
        largest_delta = max(largest_delta, delta)
        if delta > channel_tolerance:
            changed += 1
        diff[index:index + 3] = bytes((delta, 0, 0))
    if diff_path:
        write_ppm(diff_path, width, height, diff)
    fraction = changed / (width * height)
    return {"width": width, "height": height, "changed_pixels": changed,
            "changed_fraction": fraction, "max_channel_delta": largest_delta,
            "limit": max_changed_fraction, "passed": fraction <= max_changed_fraction}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="versioned visual case JSON")
    parser.add_argument("--actual", type=Path, help="compare an existing capture")
    parser.add_argument("--output", type=Path, help="capture path when running the command")
    parser.add_argument("--diff", type=Path, help="write a red channel difference PPM")
    args = parser.parse_args()
    try:
        case = json.loads(args.manifest.read_text())
        if case.get("version") != 1 or not isinstance(case.get("name"), str):
            raise ValueError("manifest requires version 1 and a name")
        tolerance = case.get("channel_tolerance", 0)
        limit = case.get("max_changed_fraction", 0)
        if not isinstance(tolerance, int) or isinstance(tolerance, bool) or not 0 <= tolerance <= 255:
            raise ValueError("channel_tolerance must be an integer in [0, 255]")
        if not isinstance(limit, (int, float)) or isinstance(limit, bool) or not math.isfinite(limit) or not 0 <= limit <= 1:
            raise ValueError("max_changed_fraction must be in [0, 1]")
        timeout = case.get("timeout_seconds", 120)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        baseline = (args.manifest.parent / case["baseline"]).resolve()
        if args.actual and args.output:
            raise ValueError("choose --actual or --output")
        actual = args.actual or args.output
        if actual is None:
            raise ValueError("provide --actual or --output")
        if args.output:
            command = case.get("command")
            if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                raise ValueError("capture requires a command array")
            command = [part.replace("{output}", str(actual)) for part in command]
            if not any("{output}" in part for part in case["command"]):
                raise ValueError("capture command must contain {output}")
            actual.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True, timeout=timeout)
        result = compare(actual, baseline, args.diff, limit, tolerance)
        print(json.dumps({"name": case["name"], "actual": str(actual),
                          "baseline": str(baseline), **result}, sort_keys=True))
        return 0 if result["passed"] else 1
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "fail", "error": str(error)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
