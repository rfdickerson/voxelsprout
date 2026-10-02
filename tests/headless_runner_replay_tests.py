import json
import subprocess
import sys
import tempfile
from pathlib import Path


runner, fixture = sys.argv[1:]
fixture_data = json.loads(Path(fixture).read_text())
states = []
for _ in range(3):
    result = subprocess.run([runner, fixture], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"headless fixture failed: {result.stderr}\n{result.stdout}")
    state = json.loads(result.stdout)
    if state["status"] != "pass" or state["ticks"] != fixture_data["steps"]:
        raise SystemExit(f"unexpected headless state: {state}")
    states.append(state)
if states[1:] != states[:-1]:
    raise SystemExit(f"headless replay diverged: {states}")

with tempfile.TemporaryDirectory() as directory:
    inspect_fixture = dict(fixture_data)
    inspect_fixture.pop("expect")
    inspect_path = Path(directory) / "inspect.json"
    inspect_path.write_text(json.dumps(inspect_fixture))
    inspected = subprocess.run([runner, str(inspect_path)], capture_output=True, text=True)
    inspection = json.loads(inspected.stdout)
    if inspected.returncode != 0 or inspection["status"] != "completed":
        raise SystemExit(f"inspection mode failed: {inspected.stderr}\n{inspected.stdout}")
    inspection.pop("status")
    expected = dict(states[0])
    expected.pop("status")
    if inspection != expected:
        raise SystemExit(f"inspection changed simulation state: {inspection}")

    wrong_fixture = dict(fixture_data)
    wrong_fixture["expect"] = dict(fixture_data["expect"])
    wrong_fixture["expect"]["x_min"] = states[0]["actor"]["position"][0] + 1000
    wrong_fixture["expect"]["x_max"] = wrong_fixture["expect"]["x_min"] + 1
    wrong_path = Path(directory) / "wrong.json"
    wrong_path.write_text(json.dumps(wrong_fixture))
    wrong = subprocess.run([runner, str(wrong_path)], capture_output=True, text=True)
    failure = json.loads(wrong.stdout)
    if wrong.returncode == 0 or failure["status"] != "fail" or "outside expected" not in failure["error"]:
        raise SystemExit(f"wrong expectation did not fail clearly: {wrong.stderr}\n{wrong.stdout}")

    stopped_fixture = dict(fixture_data)
    stopped_fixture["actor"] = dict(fixture_data["actor"])
    stopped_fixture["actor"]["desired_velocity"] = [0, 0, 0]
    stopped_path = Path(directory) / "stopped.json"
    stopped_path.write_text(json.dumps(stopped_fixture))
    stopped = subprocess.run([runner, str(stopped_path)], capture_output=True, text=True)
    if stopped.returncode == 0 or json.loads(stopped.stdout)["status"] != "fail":
        raise SystemExit("movement expectation passed with zero movement intent")

    seeded_fixture = dict(inspect_fixture)
    seeded_fixture["seed"] = fixture_data["seed"] + 1
    seeded_path = Path(directory) / "seeded.json"
    seeded_path.write_text(json.dumps(seeded_fixture))
    seeded = json.loads(subprocess.check_output([runner, str(seeded_path)], text=True))
    if seeded["random_state"] == inspection["random_state"]:
        raise SystemExit("changing the seed did not change the simulation random state")

    shorter_fixture = dict(inspect_fixture)
    shorter_fixture["steps"] = fixture_data["steps"] // 2
    shorter_path = Path(directory) / "shorter.json"
    shorter_path.write_text(json.dumps(shorter_fixture))
    shorter = json.loads(subprocess.check_output([runner, str(shorter_path)], text=True))
    if shorter["ticks"] != shorter_fixture["steps"] or shorter["actor"]["position"][0] >= inspection["actor"]["position"][0]:
        raise SystemExit("step count did not control simulation progress")

    faster_fixture = dict(inspect_fixture)
    faster_fixture["step_seconds"] = fixture_data["step_seconds"] / 2
    faster_path = Path(directory) / "smaller_timestep.json"
    faster_path.write_text(json.dumps(faster_fixture))
    faster = json.loads(subprocess.check_output([runner, str(faster_path)], text=True))
    if faster["ticks"] != inspection["ticks"] or faster["actor"]["position"][0] >= inspection["actor"]["position"][0]:
        raise SystemExit("timestep did not control simulated movement")
