#!/usr/bin/env python3
"""Local RENDER-GI-001 evidence. Never packages game assets or declares visual acceptance.

Runs serially on the reference display, preserving native DPI. GPU stage sums
exclude receiver shading and report shared screen depth separately. Paired full
GPU times include those costs; neither is a substitute for reviewing the images.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCENES = {
    'mages': ['--interior', 'Balmora, Guild of Mages'],
    'fighters': ['--interior', 'Balmora, Guild of Fighters'],
    'exterior': ['--worldspace', 'Vvardenfell'],
    'streaming': ['--worldspace', 'Vvardenfell', '--tour-file',
                  str(ROOT / 'benchmarks/routes/balmora.tour'), '--flythrough', '20'],
}


def distribution(values):
    if not values or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError('missing or invalid timing samples')
    values = sorted(values)
    def percentile(p):
        index = (len(values) - 1) * p
        low = int(index)
        return values[low] + (values[min(low + 1, len(values)-1)] - values[low]) * (index-low)
    return dict(count=len(values), median=statistics.median(values), p95=percentile(.95),
                p99=percentile(.99), maximum=values[-1], stdev=statistics.pstdev(values))


def summarize(folder, warmup, frames):
    rows = list(csv.DictReader((folder / 'frames.csv').open()))
    rows = [r for r in rows if warmup <= int(r['frame']) < warmup + frames]
    if len(rows) != frames or any(not r['interval_ms'] or r['present_accepted'] != '1' for r in rows):
        raise ValueError('incomplete frame/presentation measurements')
    gpu = {int(r['submission']): r for r in csv.DictReader((folder / 'gi.csv').open())}
    samples = [gpu[int(r['submission_id'])] for r in rows]
    stages = ['occupancy_ms', 'surface_ms', 'inject_ms', 'propagate_ms', 'ssgi_ms']
    gi = [float(r['voxel_span_ms']) + float(r['ssgi_ms']) for r in samples]
    intervals = [float(r['interval_ms']) for r in rows]
    result = {k: distribution([float(r[k]) for r in samples]) for k in ['frame_ms', 'screen_depth_ms', 'voxel_span_ms'] + stages}
    result['gi_dispatch_span_ms'] = distribution(gi)
    result['interval_ms'] = distribution(intervals)
    result['frames_over_16_67_ms'] = sum(x > 16.67 for x in intervals)
    result['updates'] = sum(float(r['surface_ms']) > .01 for r in samples)
    transient = [r for sid, r in gpu.items() if sid < int(rows[0]['submission_id'])]
    result['warmup_gi_dispatch_span_ms'] = distribution([float(r['voxel_span_ms']) + float(r['ssgi_ms']) for r in transient])
    result['gi_stage_gate'] = result['gi_dispatch_span_ms']['p95'] <= 2 and result['gi_dispatch_span_ms']['p99'] <= 3
    ft = result['interval_ms']
    result['perf_001_gate'] = (ft['median'] <= 8.33 and ft['p95'] <= 8.33 and ft['p99'] <= 10 and
                               ft['p99'] - ft['median'] <= 2 and not result['frames_over_16_67_ms'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build-linux-relwithdebinfo/odai')
    parser.add_argument('--data', type=Path, default=Path.home() / '.local/share/Steam/steamapps/common/Morrowind/Data Files')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', choices=['all'] + list(SCENES), default='all')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=180)
    parser.add_argument('--frames', type=int, default=1200)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 1 or args.frames < 2:
        parser.error('runs/warmup must be positive and frames at least two')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(binary=str(args.binary), binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  shader_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (args.binary.parent / 'share/odai/shaders').glob('*.spv')},
                  warmup=args.warmup, frames=args.frames, runs=[], visual_acceptance='requires review',
                  power_mode=Path('/sys/firmware/acpi/platform_profile').read_text().strip())
    for scene in SCENES if args.scene == 'all' else [args.scene]:
        for repetition in range(args.runs):
            for mode in ['on', 'off'] + (['indirect'] if repetition == 0 else []):
                folder = args.output / f'{scene}-{repetition}-{mode}'
                folder.mkdir(exist_ok=False)
                env = {k:v for k,v in os.environ.items() if not k.startswith(('ODAI_', 'VK_LAYER')) and k != 'VK_INSTANCE_LAYERS'}
                env.update(ODAI_WINDOW_SIZE='768x432', ODAI_RENDER_SCALE='1', ODAI_FNV_NOHUD='1',
                           ODAI_FNV_HOUR='14', ODAI_FNV_EXPOSURE_RANGE='8,8' if scene in ('mages','fighters') else '1,1',
                           ODAI_FNV_WEATHER='clear', ODAI_FNV_FLY='1',
                           ODAI_GI='off' if mode == 'off' else 'on',
                           ODAI_GI_STATS_CSV=str(folder / 'gi.csv'),
                           ODAI_FRAME_STATS_CSV=str(folder / 'frames.csv'), ODAI_BENCHMARK_CSV='1')
                if scene == 'streaming':
                    env.update(ODAI_TOUR_STREAMING='1', ODAI_TOUR_FIXED_DT='1', ODAI_TOUR_WARMUP_FRAMES=str(args.warmup))
                else:
                    env.update(ODAI_FNV_BENCH='1', ODAI_FNV_BENCH_SPEED='0', ODAI_FNV_BENCH_TURN='0', ODAI_FNV_BENCH_FIXED_DT='1')
                if scene in ('mages', 'fighters'):
                    env.update(ODAI_FNV_SPAWN_POS=('-35.032,-575.067,828.033' if scene == 'mages'
                                                  else '639.329,-382.082,-127.586'),
                               ODAI_FNV_YAW='0', ODAI_FNV_PITCH='-8')
                if scene == 'exterior':
                    env.update(ODAI_FNV_SPAWN_POS='-19920,300,12960', ODAI_FNV_YAW='-90', ODAI_FNV_PITCH='-8')
                if mode == 'indirect':
                    env['ODAI_GI_VOXEL_DEBUG'] = '6'
                command = [str(args.binary.resolve()), '--stream', str(args.data), '--plugin', 'Morrowind.esm',
                           '--no-resume'] + SCENES[scene] + ['--screenshot', str(folder / 'capture.ppm'), str(args.warmup+args.frames+3)]
                (folder / 'configuration.json').write_text(json.dumps(dict(command=command,
                    environment={k:v for k,v in env.items() if k.startswith('ODAI_')}), indent=2))
                print(folder.name, flush=True)
                with (folder / 'runtime.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
                log = (folder / 'runtime.log').read_text()
                if 'Validation Error' in log or not (folder / 'capture.ppm').is_file():
                    raise ValueError('invalid renderer run')
                if scene != 'streaming':
                    poses = re.findall(r'(?:capture|screenshot) camera: (.*)', log)
                    if len(poses) != 2 or poses[0] != poses[1] or 'mode=fly' not in poses[0]:
                        raise ValueError(f'comparison camera moved: {poses}')
                metadata = [line for line in log.splitlines() if any(key in line for key in
                    ['window: logical', 'selected GPU:', 'GI allocated resource bytes=', 'voxel GI surface mode:',
                     'spawn from', 'capture camera:', 'screenshot camera:', 'GI occupancy voxelized', 'swapchain ready:'])]
                run = dict(scene=scene, repetition=repetition, mode=mode, metadata=metadata)
                if mode != 'indirect':
                    run['measurements'] = summarize(folder, args.warmup, args.frames)
                report['runs'].append(run)
                (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    # Gate every repetition; aggregation must not hide a failing run. Visual
    # evidence, transitions and total incremental cost still need separate review.
    report['timing_gates_pass'] = all(r['measurements']['gi_stage_gate'] and r['measurements']['perf_001_gate']
                                    for r in report['runs'] if r['mode'] == 'on')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    return 0 if report['timing_gates_pass'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
