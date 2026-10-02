#!/usr/bin/env python3
"""Synthetic performance reporter checks; no GPU or retail content required."""
import importlib.util
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/iridius_bench.py'
spec = importlib.util.spec_from_file_location('iridius_bench', SOURCE)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class PerformanceReporterTests(unittest.TestCase):
    def timing_rows(self):
        rows = []
        for frame in range(6):
            row = dict(frame=str(frame), schema_version='3', interval_ms='8',
                       cpu_ms='7', gpu_ms='', gpu_submission_id='',
                       submission_id=str(10 + frame), render_attempted='1',
                       present_accepted='1', queued_frames='2', draw_calls='10', triangles='100')
            row.update({key: '0' for key in bench.PHASES})
            row['resident_cells'] = '0'
            row.update(tick_ms='1', render_ms='6', wait_acquire_ms='4',
                       cpu_work_ms='3', render_work_ms='2')
            if frame >= 2:
                row.update(gpu_submission_id=str(10 + frame - 2), gpu_ms=str(frame + 1))
            rows.append(row)
        rows[-1]['interval_ms'] = ''
        return rows

    def read_timing(self, rows, start=1, count=3):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frames.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            return bench.measured_timing_rows(path, start, count)

    def test_gpu_join_and_warmup_endpoint(self):
        rows = self.timing_rows()
        rows[0]['interval_ms'] = '90'  # warmup stall must not leak into frame 1
        rows[1]['interval_ms'] = '26'
        rows[1]['tick_ms'] = '25'
        result = self.read_timing(rows)
        self.assertEqual([r['frame'] for r in result], ['1', '2', '3'])
        self.assertEqual([r['gpu_ms'] for r in result], ['4', '5', '6'])
        self.assertEqual([r['gpu_readback_frame'] for r in result], ['3', '4', '5'])
        summary = bench.summarize_timing(result)
        self.assertEqual(summary['frame_interval_ms']['max'], 26)
        self.assertEqual(summary['phases_ms']['tick_ms']['max'], 25)
        self.assertEqual(summary['phases_ms']['wait_acquire_ms']['median'], 4)
        self.assertEqual(summary['tick_outliers'][0]['frame'], 1)
        self.assertEqual(summary['tick_outliers'][0]['tick_ms'], 25)
        self.assertEqual(summary['submissions']['skipped'], 0)
        self.assertEqual(summary['gpu_coverage'], 1)
        runs = [{'timing': summary, 'peak_rss_mb': 500}] * 3
        _, combined = bench.aggregate(runs, [{'allocations_p95': 3}] * 3)
        self.assertEqual(combined['phases_ms']['tick_ms']['max'], 25)
        self.assertEqual(combined['phases_ms']['wait_acquire_ms']['p95'], 4)
        self.assertEqual(combined['submissions']['skipped'], 0)

    def test_timing_csv_rejects_incomplete_or_ambiguous_samples(self):
        for mutation, message in (
            (lambda r: r[1].update(interval_ms=''), 'missing interval_ms'),
            (lambda r: r[1].update(wait_acquire_ms='nan'), 'invalid metric'),
            (lambda r: r[2].update(frame='1'), 'nonconsecutive'),
            (lambda r: r[3].update(gpu_submission_id='10'), 'duplicate GPU'),
            (lambda r: r[3].update(gpu_submission_id='99'), 'unknown'),
            (lambda r: r[3].update(gpu_submission_id=''), 'originating submission'),
            (lambda r: r[1].update(schema_version='1'), 'schema_version 3'),
            (lambda r: r[1].update(submission_id='10'), 'duplicate submission'),
        ):
            with self.subTest(message=message):
                rows = self.timing_rows()
                mutation(rows)
                with self.assertRaisesRegex(ValueError, message):
                    self.read_timing(rows)
        with self.assertRaisesRegex(ValueError, 'missing interval_ms'):
            self.read_timing(self.timing_rows(), start=5, count=1)

    def test_skipped_submission_keeps_wait_and_has_no_gpu_sample(self):
        rows = self.timing_rows()
        rows[1].update(submission_id='0', present_accepted='0', wait_acquire_ms='20')
        rows[3].update(gpu_submission_id='', gpu_ms='')
        result = self.read_timing(rows)
        self.assertEqual(result[0]['gpu_ms'], '')
        self.assertEqual(result[0]['wait_acquire_ms'], '20')
        # The strict gate rejects skipped frames even with smooth intervals.
        summary = {'frame_interval_ms': {'median': 8, 'p95': 8, 'p99': 8,
                   'max': 8, 'long_frames': 0}, 'submissions': {'skipped': 1}}
        self.assertIn('skipped: 1 frames', bench.validate_120hz({'summary': summary}))

    def test_percentiles_and_missing_samples(self):
        self.assertEqual(bench.percentile([1, 2, 3, 4, 5], .95), 4.8)
        with self.assertRaises(ValueError):
            bench.percentile([], .95)

    def test_frame_coverage_and_summary(self):
        row = {'cpu_ms': '4', 'gpu_ms': '5', 'interval_ms': '7',
               'draw_calls': '742', 'triangles': '2184000'}
        rows = [dict(row) for _ in range(100)]
        result = bench.summarize_timing(rows)
        self.assertEqual(result['cpu_frame_ms']['p95'], 4)
        self.assertEqual(result['triangles'], 2184000)
        self.assertEqual(result['frame_interval_ms']['p99'], 7)
        self.assertEqual(result['frame_interval_ms']['long_frames'], 0)
        for item in rows[:6]:
            item['gpu_ms'] = ''
        with self.assertRaisesRegex(ValueError, 'gpu_ms'):
            bench.summarize_timing(rows)

    def test_budget_pass_fail_and_hardware(self):
        hardware = {'cpu': 'a', 'gpu': 'b', 'extent': '768x432'}
        result = {'hardware': hardware, 'metrics': {'cpu_p95_ms': 9.7}}
        budget = {'hardware': hardware, 'baseline': {'cpu_p95_ms': 4.8},
                  'limits': {'cpu_p95_ms': 6}}
        self.assertIn('regression +102.1%', bench.validate_budget(result, budget)[0])
        result['metrics']['cpu_p95_ms'] = 5.5
        self.assertEqual(bench.validate_budget(result, budget), [])
        budget['hardware'] = dict(hardware, gpu='different')
        with self.assertRaisesRegex(ValueError, 'uncalibrated'):
            bench.validate_budget(result, budget)

    def test_csv_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'samples.csv'
            path.write_text('frame,allocations\n0,9\n1,7\n2,11\n')
            rows = bench.measured_rows(path, 1, 2)
            self.assertEqual(bench.summarize_allocations(rows), 10.8)
            with self.assertRaisesRegex(ValueError, 'measured frames'):
                bench.measured_rows(path, 2, 2)

    def test_saved_report_cli_fail_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / 'report.json'
            budget = root / 'budget.json'
            hardware = {'cpu': 'fixture', 'gpu': 'fixture', 'extent': '768x432'}
            report.write_text(json.dumps({'scenario': 'balmora', 'hardware': hardware,
                                          'route_sha256': 'fixture',
                                          'metrics': {'cpu_p95_ms': 9.7}}))
            budget.write_text(json.dumps({'hardware': hardware, 'route_sha256': 'fixture',
                                          'baseline': {'cpu_p95_ms': 4.8},
                                          'limits': {'cpu_p95_ms': 6}}))
            process = subprocess.run([sys.executable, str(SOURCE), 'balmora',
                                      '--report', str(report), '--budget', str(budget)],
                                     text=True, capture_output=True)
            self.assertEqual(process.returncode, 3)
            self.assertIn('regression +102.1%', process.stderr)

    def test_intentional_stall_fails_120hz(self):
        rows = [{'cpu_ms': '2', 'gpu_ms': '4', 'interval_ms': '8',
                 'draw_calls': '10', 'triangles': '100'} for _ in range(200)]
        smooth = bench.summarize_timing(rows)
        self.assertEqual(bench.validate_120hz({'summary': smooth}), [])
        for index in (30, 60, 90, 120, 150):
            rows[index]['interval_ms'] = '25'  # injected recurring stall
        stalled = bench.summarize_timing(rows)
        failures = bench.validate_120hz({'summary': stalled})
        self.assertTrue(any('p99' in failure for failure in failures))
        self.assertTrue(any('routine frames' in failure for failure in failures))

    def test_aggregate_keeps_worst_maximum(self):
        runs = []
        for spike in (20, 30, 40):
            rows = [{'cpu_ms': '2', 'gpu_ms': '4', 'interval_ms': '8',
                     'draw_calls': '10', 'triangles': '100'} for _ in range(100)]
            rows[50]['interval_ms'] = str(spike)
            runs.append({'timing': bench.summarize_timing(rows), 'peak_rss_mb': 500})
        _, summary = bench.aggregate(runs, [{'allocations_p95': 3}] * 3)
        self.assertEqual(summary['frame_interval_ms']['max'], 40)
        self.assertEqual(summary['frame_interval_ms']['long_frames'], 3)


if __name__ == '__main__':
    unittest.main()
