import json
import subprocess
import sys
import tempfile
from pathlib import Path


runner, actor_path, inventory_path = sys.argv[1:]


def run_fixture(fixture):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.json"
        path.write_text(json.dumps(fixture))
        result = subprocess.run([runner, str(path)], capture_output=True, text=True)
    return result.returncode, json.loads(result.stdout), result.stderr


actor = json.loads(Path(actor_path).read_text())
code, original, error = run_fixture(actor)
assert code == 0 and "snapshots" not in original, error

actor["snapshots"] = {"ticks": [actor["steps"], 0, 5]}
runs = [run_fixture(actor) for _ in range(3)]
assert all(code == 0 for code, _, _ in runs), runs
assert runs[0][1] == runs[1][1] == runs[2][1], "snapshot replay diverged"
result = runs[0][1]
snapshots = result["snapshots"]
assert snapshots["version"] == 1
assert [entry["tick"] for entry in snapshots["checkpoints"]] == [0, 5, actor["steps"]]
initial, middle, end = [entry["state"] for entry in snapshots["checkpoints"]]
assert initial["objects"][0]["transform"]["position"][0] == 0
assert 0 < middle["objects"][0]["transform"]["position"][0] < end["objects"][0]["transform"]["position"][0]
assert end == snapshots["final"]["state"]
assert end["state_hash"] == result["state_hash"]
assert end["random_state"] == result["random_state"]
assert end["player_object"] == end["objects"][0]["id"] == end["characters"][0]["object"]
assert end["characters"][0]["position"] == result["physics"]["position"]
assert end["objects"][0]["space"]["kind"] == "exterior"
assert end["quests"] == []

final_only = dict(actor)
final_only["snapshots"] = {}
code, result, error = run_fixture(final_only)
assert code == 0 and result["snapshots"]["checkpoints"] == [], error
assert result["snapshots"]["final"]["state"]["tick"] == actor["steps"]

inventory = json.loads(Path(inventory_path).read_text())
inventory["snapshots"] = {"ticks": [0, 1, 2, 3]}
code, result, error = run_fixture(inventory)
assert code == 0, error
checkpoints = result["snapshots"]["checkpoints"]
assert [entry["tick"] for entry in checkpoints] == [0, 1, 2, 3]
assert [len(entry["state"]["objects"]) for entry in checkpoints] == [1, 1, 2, 3]
assert [len(entry["state"]["objects"][0]["inventory"]) for entry in checkpoints] == [2, 2, 2, 1]
assert result["snapshots"]["final"]["state"] == checkpoints[-1]["state"]

for ticks in ([0, 0], [-1], [actor["steps"] + 1], [1.5]):
    invalid = dict(actor)
    invalid["snapshots"] = {"ticks": ticks}
    code, result, error = run_fixture(invalid)
    assert code != 0 and result["status"] == "fail" and "snapshot" in result["error"], (ticks, error, result)

wrong = dict(actor)
wrong["expect"] = dict(actor["expect"])
wrong["expect"]["x_min"] = 1000
wrong["expect"]["x_max"] = 1001
code, result, error = run_fixture(wrong)
assert code != 0 and result["status"] == "fail", error
assert result["snapshots"]["final"]["state"]["tick"] == actor["steps"]
