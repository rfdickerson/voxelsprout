#!/usr/bin/env python3
"""Repeatable local Vulkan traversal benchmark and checked budget reporter."""
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
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / 'benchmarks/routes/balmora.tour'
BUDGET = ROOT / 'benchmarks/balmora.json'
WARMUP = 240
FRAMES = 1200
EXTENT = '768x432'
PHASES = ('tick_ms', 'render_ms', 'cpu_work_ms', 'render_work_ms',
          'wait_frame_slot_ms', 'wait_acquire_ms', 'wait_present_ms', 'wait_transfer_ms',
          'tick_session_ms', 'tick_actors_ms', 'tick_streaming_ms',
          'streamer_update_ms', 'streamer_apply_ms', 'streamer_upload_ms',
          'streamer_callbacks_ms', 'streamer_eviction_ms',
          'streamer_renderer_remove_ms', 'streamer_eviction_callbacks_ms',
          'eviction_collision_ms', 'eviction_navigation_ms', 'eviction_doors_ms',
          'cell_registration_ms',
          'runtime_objects_ms', 'gameplay_cell_ms', 'gameplay_io_ms',
          'gameplay_compile_ms', 'gameplay_publish_ms', 'gameplay_anchor_ms',
          'gameplay_upsert_ms', 'physics_install_ms',
          'collision_world_ms', 'navigation_ms', 'broad_phase_ms')


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered or not all(math.isfinite(x) and x >= 0 for x in ordered):
        raise ValueError('missing or invalid metric samples')
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    return ordered[low] + (ordered[math.ceil(index)] - ordered[low]) * (index - low)


def measured_rows(path, first_frame, minimum):
    with path.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['frame']) >= first_frame]
    if len(rows) < minimum:
        raise ValueError(f'{path}: {len(rows)} measured frames, expected at least {minimum}')
    return rows[:minimum]


def measured_timing_rows(path, first_frame, count):
    """Join delayed GPU events before selecting the measured CPU frame window."""
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row.get('schema_version') != '3' for row in rows):
        raise ValueError('timing CSV requires schema_version 3; rerun with the updated runtime')
    frames = [int(row['frame']) for row in rows]
    if frames != list(range(frames[0], frames[0] + len(frames))):
        raise ValueError('timing CSV has duplicate or nonconsecutive frames')
    submissions = {}
    for row in rows:
        submission = int(row['submission_id'])
        if submission < 0 or any(row[key] not in ('0', '1')
                                 for key in ('render_attempted', 'present_accepted')):
            raise ValueError('invalid submission outcome')
        if (submission and row['render_attempted'] != '1') or (
                row['present_accepted'] == '1' and not submission):
            raise ValueError('inconsistent submission outcome')
        if submission:
            if submission in submissions:
                raise ValueError('duplicate submission ID')
            submissions[submission] = int(row['frame'])
    gpu_samples = {}
    for row in rows:
        sample_id = row.get('gpu_submission_id', '')
        if bool(sample_id) != bool(row.get('gpu_ms')):
            raise ValueError('GPU timing must have an originating submission ID')
        if sample_id:
            sample_id = int(sample_id)
            if sample_id not in submissions or submissions[sample_id] >= int(row['frame']):
                raise ValueError('GPU timing has an unknown or nonpreceding submission')
            if sample_id in gpu_samples:
                raise ValueError('duplicate GPU timing event')
            percentile([float(row['gpu_ms'])], .5)  # reject NaN/negative samples
            gpu_samples[sample_id] = (row['gpu_ms'], row['frame'])
    selected = [dict(row) for row in rows
                if first_frame <= int(row['frame']) < first_frame + count]
    if [int(row['frame']) for row in selected] != list(range(first_frame, first_frame + count)):
        raise ValueError(f'{path}: incomplete measured frame window')
    for row in selected:
        for key in ('interval_ms', 'cpu_ms', 'draw_calls', 'triangles',
                    'resident_cells', *PHASES):
            if not row.get(key):
                raise ValueError(f'frame {row["frame"]}: missing {key}')
            percentile([float(row[key])], .5)
        submission = int(row['submission_id'])
        sample, readback_frame = gpu_samples.get(submission, ('', ''))
        row['gpu_ms'] = sample
        row['gpu_submission_id'] = str(submission) if sample else ''
        row['gpu_readback_frame'] = readback_frame
    return selected


def distribution(values):
    return {'median': percentile(values, .5), 'p95': percentile(values, .95),
            'p99': percentile(values, .99), 'max': max(values),
            'stddev': statistics.pstdev(values)}


def summarize_timing(rows):
    def numbers(key, coverage=1):
        data = [float(r[key]) for r in rows if r.get(key)]
        if len(data) < math.ceil(len(rows) * coverage):
            raise ValueError(f'{key}: {len(data)}/{len(rows)} samples; need {coverage:.0%}')
        percentile(data, .5)
        return data
    cpu = numbers('cpu_ms')
    gpu = numbers('gpu_ms', .95)
    interval = numbers('interval_ms')
    draws = numbers('draw_calls')
    triangles = numbers('triangles')
    interval_stats = {
        'median': percentile(interval, .5), 'p95': percentile(interval, .95),
        'p99': percentile(interval, .99), 'max': max(interval),
        'stddev': statistics.pstdev(interval),
        'long_frames': sum(value > 16.67 for value in interval),
    }
    result = {
        'cpu_frame_ms': {'median': percentile(cpu, .5), 'p95': percentile(cpu, .95)},
        'gpu_frame_ms': {'median': percentile(gpu, .5), 'p95': percentile(gpu, .95)},
        'frame_interval_ms': interval_stats,
        'draw_calls': round(percentile(draws, .5)),
        'triangles': round(percentile(triangles, .5)),
        'gpu_coverage': len(gpu) / len(rows),
    }
    if any('schema_version' in row for row in rows):
        result['phases_ms'] = {key: distribution(numbers(key)) for key in PHASES}
        result['submissions'] = {
            'skipped': sum(int(row['submission_id']) == 0 for row in rows),
            'present_not_accepted': sum(int(row['submission_id']) != 0 and
                                        row['present_accepted'] != '1' for row in rows),
        }
        result['queued_frames'] = distribution(numbers('queued_frames'))
        stages = ('tick_session_ms', 'tick_actors_ms', 'tick_streaming_ms',
                  'streamer_update_ms', 'streamer_apply_ms', 'streamer_upload_ms',
                  'streamer_callbacks_ms', 'streamer_eviction_ms',
                  'streamer_renderer_remove_ms', 'streamer_eviction_callbacks_ms',
                  'eviction_collision_ms', 'eviction_navigation_ms',
                  'eviction_doors_ms',
                  'cell_registration_ms', 'runtime_objects_ms', 'gameplay_cell_ms',
                  'gameplay_io_ms', 'gameplay_compile_ms', 'gameplay_publish_ms',
                  'gameplay_anchor_ms', 'gameplay_upsert_ms',
                  'physics_install_ms', 'collision_world_ms', 'navigation_ms',
                  'broad_phase_ms', 'resident_cells')
        result['tick_outliers'] = [
            {'frame': int(row['frame']), 'tick_ms': float(row['tick_ms']),
             **{key: float(row[key]) for key in stages}}
            for row in sorted(rows, key=lambda row: float(row['tick_ms']), reverse=True)[:10]]
    return result


def summarize_allocations(rows):
    return percentile([float(r['allocations']) for r in rows], .95)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def hardware(log):
    cpuinfo = Path('/proc/cpuinfo').read_text()
    cpu = re.search(r'^model name\s*:\s*(.+)$', cpuinfo, re.MULTILINE)
    gpu = re.search(r'selected GPU:\s*(.+)', log)
    window = re.search(r'window: logical (\d+x\d+), framebuffer (\d+x\d+)', log)
    swapchain = re.search(r'swapchain ready:.*extent=(\d+x\d+), presentMode=(\w+)', log)
    if not cpu or not gpu or not window or not swapchain:
        raise ValueError('cannot identify CPU, GPU, framebuffer or present mode')
    if window.group(2) != EXTENT or swapchain.group(1) != EXTENT:
        raise ValueError(f'benchmark framebuffer is not {EXTENT}')
    return {'cpu': cpu.group(1).strip(), 'gpu': gpu.group(1).strip(),
            'extent': EXTENT, 'present_mode': swapchain.group(2)}


def validate_budget(result, budget):
    if result['hardware'] != budget['hardware']:
        raise ValueError('uncalibrated hardware or resolution: run --calibrate on this host')
    failures = []
    for name, limit in budget['limits'].items():
        value = result['metrics'][name]
        if value > limit:
            base = budget['baseline'][name]
            change = (value / base - 1) * 100 if base else float('inf')
            failures.append(f'{name}: baseline {base:.3f}, current {value:.3f}, '
                            f'budget {limit:.3f}, regression {change:+.1f}%')
    return failures


def validate_120hz(result):
    intervals = result['summary']['frame_interval_ms']
    limits = {'median': 8.33, 'p95': 8.33, 'p99': 10.0,
              'max': 16.67, 'p99_minus_median': 2.0}
    actual = dict(intervals, p99_minus_median=intervals['p99'] - intervals['median'])
    failures = [f'frame {key}: {actual[key]:.3f} ms > {limit:.3f} ms'
                for key, limit in limits.items() if actual[key] > limit]
    if intervals['long_frames']:
        failures.append(f"routine frames >16.67 ms: {intervals['long_frames']}")
    for key, count in result['summary'].get('submissions', {}).items():
        if count:
            failures.append(f'{key}: {count} frames')
    return failures


def report_verdict(result, budget_path, target_120hz=False):
    budget = json.loads(budget_path.read_text())
    if budget['route_sha256'] != result['route_sha256']:
        raise ValueError('route changed since calibration')
    failures = validate_budget(result, budget)
    if target_120hz:
        failures.extend(validate_120hz(result))
    if failures:
        print('PERFORMANCE: FAIL\n' + '\n'.join(failures), file=sys.stderr)
        return 3
    print('PERFORMANCE: PASS')
    return 0


def environment(csv_path, heap_path=None, interposer=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ODAI_', 'VK_LAYER')) and k != 'VK_INSTANCE_LAYERS'}
    env.update({'ODAI_WINDOW_SIZE': EXTENT, 'ODAI_WINDOW_HIDPI': '0',
                'ODAI_RENDER_SIZE': EXTENT, 'ODAI_RENDER_SCALE': '1',
                'ODAI_PRESENT_MODE': 'immediate', 'ODAI_FNV_NOHUD': '1',
                'ODAI_TOUR_FIXED_DT': '1', 'ODAI_TOUR_WARMUP_FRAMES': str(WARMUP),
                'ODAI_TOUR_STREAMING': '1', 'ODAI_BENCHMARK_CSV': '1',
                'ODAI_FRAME_STATS_CSV': str(csv_path)})
    if heap_path:
        env['ODAI_HEAP_ALLOC_CSV'] = str(heap_path)
        env['LD_PRELOAD'] = str(interposer)
    return env


def run_once(binary, data, out, allocation=False):
    csv_path = out / 'frames.csv'
    heap_path = out / 'allocations.csv' if allocation else None
    interposer = binary.parent / 'libodai_bench_heap.so'
    if allocation and not interposer.is_file():
        raise ValueError(f'missing allocation interposer: {interposer}')
    env = environment(csv_path, heap_path, interposer)
    command = [str(binary), '--stream', str(data), '--plugin', 'Morrowind.esm',
               '--worldspace', 'Vvardenfell', '--no-resume',
               '--tour-file', str(ROUTE), '--flythrough', str(FRAMES / 60),
               # Three trailing frames close the final interval and retire the
               # two GPU frame slots before screenshot readback/exit work.
               '--screenshot', str(out / 'endpoint.ppm'), str(WARMUP + FRAMES + 3)]
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'runtime.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                proc.returncode = os.waitstatus_to_exitcode(status)
                break
            if time.monotonic() - started > 900:
                proc.kill()
                os.waitpid(proc.pid, 0)
                raise TimeoutError('runtime exceeded 900 seconds')
            time.sleep(.1)
    log_text = (out / 'runtime.log').read_text(errors='replace')
    if proc.returncode:
        raise RuntimeError(f'odai exited {proc.returncode}; see {out / "runtime.log"}')
    screenshot = out / 'endpoint.ppm'
    if not screenshot.is_file() or not screenshot.open('rb').readline().startswith(b'P6'):
        raise ValueError('runtime did not reach the benchmark endpoint')
    screenshot.unlink()
    rows = measured_timing_rows(csv_path, WARMUP, FRAMES)
    with (out / 'attributed-frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {'peak_rss_mb': usage.ru_maxrss / 1024, 'hardware': hardware(log_text),
              'command': command, 'allocation_instrumented': allocation}
    if allocation:
        alloc_rows = measured_rows(heap_path, WARMUP, FRAMES)
        record['allocations_p95'] = summarize_allocations(alloc_rows)
    else:
        record['timing'] = summarize_timing(rows)
    (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def aggregate(timings, allocations):
    keymap = {'cpu_p95_ms': ('cpu_frame_ms', 'p95'),
              'gpu_p95_ms': ('gpu_frame_ms', 'p95'),
              'interval_p95_ms': ('frame_interval_ms', 'p95')}
    metrics = {name: statistics.median(run['timing'][field][stat] for run in timings)
               for name, (field, stat) in keymap.items()}
    metrics['memory_mb'] = statistics.median(run['peak_rss_mb'] for run in timings)
    metrics['allocations_per_frame'] = statistics.median(run['allocations_p95'] for run in allocations)
    summary = {}
    for field in ('cpu_frame_ms', 'gpu_frame_ms', 'frame_interval_ms'):
        statistics_to_keep = ('median', 'p95') if field != 'frame_interval_ms' else (
            'median', 'p95', 'p99', 'stddev')
        summary[field] = {stat: statistics.median(run['timing'][field][stat] for run in timings)
                          for stat in statistics_to_keep}
    summary['frame_interval_ms']['max'] = max(
        run['timing']['frame_interval_ms']['max'] for run in timings)
    summary['frame_interval_ms']['long_frames'] = sum(
        run['timing']['frame_interval_ms']['long_frames'] for run in timings)
    summary['frame_interval_ms']['p99_minus_median'] = (
        summary['frame_interval_ms']['p99'] - summary['frame_interval_ms']['median'])
    for field in ('draw_calls', 'triangles'):
        summary[field] = round(statistics.median(run['timing'][field] for run in timings))
    if all('phases_ms' in run['timing'] for run in timings):
        summary['phases_ms'] = {
            phase: {stat: (max if stat == 'max' else statistics.median)(
                run['timing']['phases_ms'][phase][stat] for run in timings)
                    for stat in ('median', 'p95', 'p99', 'max', 'stddev')}
            for phase in PHASES}
        summary['submissions'] = {
            key: sum(run['timing']['submissions'][key] for run in timings)
            for key in ('skipped', 'present_not_accepted')}
        summary['tick_outliers'] = sorted(
            (dict(outlier, run=run_index)
             for run_index, run in enumerate(timings)
             for outlier in run['timing']['tick_outliers']),
            key=lambda row: row['tick_ms'], reverse=True)[:10]
    summary['memory_mb'] = metrics['memory_mb']
    summary['allocations_per_frame'] = metrics['allocations_per_frame']
    return metrics, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', choices=['balmora'])
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--data', type=Path, default=Path.home() / '.local/share/Steam/steamapps/common/Morrowind/Data Files')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build-linux-relwithdebinfo/odai')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--budget', type=Path, default=BUDGET)
    parser.add_argument('--report', type=Path, help='check a saved report without rerunning the engine')
    parser.add_argument('--target-120hz', action='store_true', help='also enforce PERF-001 frame pacing limits')
    args = parser.parse_args()
    if args.report:
        try:
            result = json.loads(args.report.read_text())
            if result['scenario'] != args.scenario:
                raise ValueError('report scenario mismatch')
            return report_verdict(result, args.budget, args.target_120hz)
        except (OSError, ValueError, KeyError) as exc:
            print(f'PERFORMANCE: UNCALIBRATED/INVALID: {exc}', file=sys.stderr)
            return 2
    binary = args.binary.resolve()
    data = args.data.resolve()
    if not binary.is_file() or not (data / 'Morrowind.esm').is_file():
        parser.error('optimized odai binary or local Morrowind.esm is missing')
    if 'relwithdebinfo' not in str(binary.parent).lower() and 'release' not in str(binary.parent).lower():
        parser.error('performance measurements require an optimized build')
    destination = args.output or Path(tempfile.mkdtemp(prefix='iridius-bench-balmora-'))
    destination.mkdir(parents=True, exist_ok=True)
    print(f'benchmark evidence: {destination}', flush=True)
    try:
        timings = [run_once(binary, data, destination / f'timing-{n}') for n in range(3)]
        allocations = [run_once(binary, data, destination / f'allocations-{n}', True) for n in range(3)]
        fingerprints = [run['hardware'] for run in timings + allocations]
        if any(item != fingerprints[0] for item in fingerprints):
            raise ValueError('hardware changed between runs')
        metrics, summary = aggregate(timings, allocations)
        result = {'scenario': args.scenario, 'hardware': fingerprints[0],
                  'route_sha256': digest(ROUTE), 'binary_sha256': digest(binary),
                  'build_configuration': 'RelWithDebInfo',
                  'warmup_frames': WARMUP, 'measured_frames': FRAMES,
                  'repetitions': 3, 'timing_schema_version': 3,
                  'metrics': metrics, 'summary': summary}
        (destination / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        if args.calibrate:
            factors = {'cpu_p95_ms': 1.25, 'gpu_p95_ms': 1.25,
                       'interval_p95_ms': 1.25, 'memory_mb': 1.15,
                       'allocations_per_frame': 1.25}
            budget = {'scenario': args.scenario, 'hardware': fingerprints[0],
                      'route_sha256': result['route_sha256'], 'baseline': metrics,
                      'limits': {key: round(value * factors[key], 3) for key, value in metrics.items()}}
            args.budget.write_text(json.dumps(budget, indent=2) + '\n')
            print(f'calibrated budget: {args.budget}')
            return 0
        return report_verdict(result, args.budget, args.target_120hz)
    except (OSError, ValueError, RuntimeError, TimeoutError, KeyError) as exc:
        print(f'PERFORMANCE: UNCALIBRATED/INVALID: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
