#!/usr/bin/env python3
"""Validate navigation maps and query behavior against the configured build."""
import json
from pathlib import Path
import subprocess
import unittest
import nav

class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.repo = json.loads((nav.OUT / 'repo-map.json').read_text())
        self.deps = json.loads((nav.OUT / 'dependency-map.json').read_text())
        self.tests = json.loads((nav.OUT / 'test-map.json').read_text())

    def query(self, *args):
        return subprocess.run(['python3', str(nav.OUT / 'nav.py'), *args], cwd=nav.ROOT, text=True, capture_output=True, check=True).stdout

    def test_ownership_exceptions(self):
        owners = self.repo['source_owners']
        self.assertIn('odai_bethesda_import', owners['src/bethesda/condition.cc'])
        self.assertIn('odai', owners['src/import/bethesda/cell_streamer.cc'])
        self.assertIn('odai_renderer', owners['src/render/upscale/temporal_upscaler.cc'])
        self.assertIn('odai_upscale', owners['src/render/upscale/upscale_policy.cc'])

    def test_target_graph_is_acyclic_and_sources_exist(self):
        targets = self.repo['targets']
        visited, active = set(), set()
        def visit(n):
            self.assertNotIn(n, active, f'dependency cycle at {n}')
            if n in visited:
                return
            active.add(n)
            for dep in targets[n]['dependencies']:
                if dep in targets:
                    visit(dep)
            active.remove(n)
            visited.add(n)
        for n, t in targets.items():
            visit(n)
            for s in t['sources']:
                self.assertTrue((nav.ROOT / s).is_file(), s)

    def test_test_registration_matches_ctest(self):
        live = json.loads(nav.run(['ctest', '--test-dir', self.repo['build'], '--show-only=json-v1']))
        self.assertEqual({t['name'] for t in live['tests']}, {t['name'] for t in self.tests['tests']})

    def test_portable_cpu_checks_do_not_reach_presentation(self):
        targets = self.repo['targets']
        for name in ('odai_bindless_slot_table_tests', 'odai_gpu_arena_allocator_tests',
                     'odai_imported_render_policy_tests', 'odai_frame_graph_tests',
                     'odai_upscaler_tests'):
            self.assertIn(name, targets)
            pending, reached = [name], set()
            while pending:
                current = pending.pop()
                if current in reached:
                    continue
                reached.add(current)
                pending.extend(targets.get(current, {}).get('dependencies', []))
            self.assertFalse(reached & {'odai_renderer', 'odai_ui', 'slang_shaders'}, name)
            for current in reached:
                for fragment in targets.get(current, {}).get('link_fragments', []):
                    self.assertNotRegex(fragment.lower(), r'vulkan|glfw|imgui', name)

    def test_focused_queries(self):
        for file, expected in [('src/core/job_system.h', 'odai_job_system_tests'), ('src/render/taa_depth_policy.h', 'odai_taa_depth_tests'), ('src/import/cell_residency_planner.h', 'odai_cell_residency_planner_tests')]:
            selected = json.loads(self.query('tests', file))
            self.assertTrue(any(t['name'] == expected and t['evidence'] == 'direct test include' for t in selected))
        self.assertIn('JobSystem::waitIdle', self.query('find', 'waitIdle', '--scope', 'src/core/job_system.cc'))
        self.assertEqual(json.loads(self.query('owner', 'src/core/job_system.cc'))['compiled_by'], ['odai_core'])

    def test_unknown_test_scope_fails(self):
        result = subprocess.run(['python3', str(nav.OUT / 'nav.py'), 'run-tests', 'missing/file.h'], cwd=nav.ROOT, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_includers_and_executable_reachability(self):
        included = json.loads(self.query('includers', 'src/core/job_system.h'))
        self.assertIn('src/core/job_system.cc', included['direct_includers'])
        reached = json.loads(self.query('reaches', 'odai_bethesda_runtime'))
        self.assertIn('odai_headless', reached['executables'])
        self.assertNotIn('odai_texture_pack', reached['executables'])
        header = json.loads(self.query('reaches', 'src/core/job_system.h'))
        self.assertIn('odai_job_system_tests', header['executables'])

    def test_provenance_agrees(self):
        for mapped in (self.deps, self.tests):
            for key in ('schema_version', 'build', 'configuration', 'cmake_sha256'):
                self.assertEqual(self.repo[key], mapped[key])

if __name__ == '__main__':
    unittest.main()
