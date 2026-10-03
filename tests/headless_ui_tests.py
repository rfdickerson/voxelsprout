import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path


runner, fixture_path = sys.argv[1:]
fixture = json.loads(Path(fixture_path).read_text())


def run(value):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scenario.json"
        path.write_text(json.dumps(value))
        process = subprocess.run([runner, str(path)], text=True, capture_output=True)
        return process.returncode, json.loads(process.stdout), process.stderr


code, first, _ = run(fixture)
assert code == 0 and first["status"] == "pass", first
code, second, _ = run(fixture)
assert code == 0 and first == second, "headless UI replay changed between runs"

world = first["final"]["world_items"]
assert len(world) == 2
for item in world:
    assert item["position"] == [80.0, 100.0, 0.0], item

for field, wrong in (
    ("screen", "map"),
    ("selected", "tes3:WEAP:wrong"),
    ("visible_items", {"tes3:WEAP:iron_dagger": 99}),
    ("labels", {"tes3:WEAP:iron_dagger": "Wrong"}),
    ("inventory", {"tes3:WEAP:iron_dagger": 99}),
    ("world_items", ["tes3:WEAP:wrong"]),
):
    broken = copy.deepcopy(fixture)
    target = 4 if field in ("inventory", "world_items") else 0
    broken["frames"][target]["expect"][field] = wrong
    code, output, stderr = run(broken)
    assert code != 0 and field in output["error"] and "expected" in stderr, (field, output)

for keys in (["Unknown"], ["I", "I"]):
    broken = copy.deepcopy(fixture)
    broken["frames"][0]["keys"] = keys
    code, output, _ = run(broken)
    assert code != 0 and "$.frames[0].keys" in output["error"], output

print("headless UI scenario, replay, negative assertions, and input validation pass")
