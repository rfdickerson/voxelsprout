import json
import subprocess
import sys
import tempfile
from pathlib import Path


runner, fixture_path, visual_tool = sys.argv[1:]
source = json.loads(Path(fixture_path).read_text())


def run_fixture(data):
    path = work / "scenario.json"
    path.write_text(json.dumps(data))
    result = subprocess.run([runner, str(path)], capture_output=True, text=True)
    return result.returncode, json.loads(result.stdout)


with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    for version in (None, 1):
        case = dict(source)
        if version is not None:
            case["version"] = version
        code, result = run_fixture(case)
        assert code == 0 and result["status"] == "pass", result
    for edit, field in [({"version": 2}, "$.version"),
                        ({"unknown": 1}, "$.unknown"),
                        ({"actor": {**source["actor"], "position": [0, 1]}}, "actor.position"),
                        ({"steps": "30"}, "$.steps"),
                        ({"kind": "other"}, "$.kind")]:
        code, result = run_fixture({**source, **edit})
        assert code != 0 and field in result["error"], result

    baseline = work / "baseline.ppm"
    actual = work / "actual.ppm"
    diff = work / "diff.ppm"
    baseline.write_bytes(b"P6\n2 2\n255\n" + bytes([10, 20, 30] * 4))
    actual.write_bytes(baseline.read_bytes())
    manifest = work / "visual.json"
    manifest.write_text(json.dumps({"version": 1, "name": "synthetic",
                                    "baseline": "baseline.ppm", "channel_tolerance": 2,
                                    "max_changed_fraction": 0.25}))

    def visual():
        result = subprocess.run([sys.executable, visual_tool, str(manifest),
                                 "--actual", str(actual), "--diff", str(diff)],
                                capture_output=True, text=True)
        return result.returncode, json.loads(result.stdout)

    code, report = visual()
    assert code == 0 and report["changed_pixels"] == 0 and diff.is_file(), report
    pixels = bytearray(actual.read_bytes())
    pixels[-1] += 1
    actual.write_bytes(pixels)
    code, report = visual()
    assert code == 0 and report["changed_pixels"] == 0, report
    pixels[-1] += 3
    actual.write_bytes(pixels)
    code, report = visual()
    assert code == 0 and report["changed_pixels"] == 1, report
    pixels[-4] += 4
    actual.write_bytes(pixels)
    code, report = visual()
    assert code == 1 and report["changed_pixels"] == 2, report
    actual.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
    code, report = visual()
    assert code == 2 and "dimensions differ" in report["error"], report

    manifest.write_text(json.dumps({"version": 1, "name": "capture",
                                    "baseline": "baseline.ppm", "command":
                                    [sys.executable, "-c",
                                     "import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])",
                                     str(baseline), "{output}"]}))
    captured = work / "captured.ppm"
    result = subprocess.run([sys.executable, visual_tool, str(manifest),
                             "--output", str(captured)], capture_output=True, text=True)
    assert result.returncode == 0 and captured.read_bytes() == baseline.read_bytes(), result.stderr
