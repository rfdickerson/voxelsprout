#!/usr/bin/env python3
"""Game-data-free record/override and certification evidence fault tests."""
import copy
import importlib.util
from pathlib import Path
import struct
import tempfile
import sys
import unittest

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'scripts/tes3_release_audit.py')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import tr_certify

audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sub(tag, data):
    if isinstance(data, str):
        data = data.encode() + b'\0'
    return tag.encode() + struct.pack('<I', len(data)) + data


def record(tag, *subs):
    body = b''.join(subs)
    return tag.encode() + struct.pack('<III', len(body), 0, 0) + body


class Inventory(unittest.TestCase):
    def test_winning_profiles_shared_exterior_and_quest_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, mod, optional = [root / name for name in ('base.esm', 'expansion.esm', 'integration.esp')]
            base.write_bytes(record('CELL', sub('NAME', 'shared'), sub('DATA', struct.pack('<Iii', 0, 1, 2))) +
                             record('DIAL', sub('NAME', 'old_quest'), sub('DATA', b'\x04')) +
                             record('INFO', sub('INAM', 'entry'), sub('QSTN', b'\x01')))
            mod.write_bytes(record('CELL', sub('NAME', 'renamed'), sub('DATA', struct.pack('<Iii', 0, 1, 2)),
                                   sub('FRMR', struct.pack('<I', 1)), sub('NAME', 'door'), sub('DELE', b'\0')) +
                            record('CELL', sub('NAME', 'room'), sub('DATA', struct.pack('<Iii', 1, 0, 0))) +
                            record('DIAL', sub('NAME', 'old_quest'), sub('DATA', b'\x04')) +
                            record('INFO', sub('INAM', 'entry'), sub('QSTF', b'\x01')) +
                            record('SCPT', sub('SCHD', b'test\0' + bytes(47)), sub('SCTX', 'journal old_quest 10\nJournal unknown 20')) +
                            record('ZZZZ', sub('UNKN', b'\xff')))
            optional.write_bytes(record('DIAL', sub('NAME', 'old_quest'), sub('DATA', b'\x04')) +
                                 record('INFO', sub('INAM', 'entry'), sub('BNAM', 'Journal old_quest 30')))
            a = audit.inventory([base, mod], [mod.name])
            b = audit.inventory([base, mod, optional], [mod.name, optional.name])
            self.assertEqual(a['counts']['exterior_cells'], 1)
            self.assertEqual(a['counts']['interior_cells'], 1)
            self.assertEqual(a['cells'][0]['name'], 'renamed')
            self.assertEqual(len(a['cells'][0]['provenance']), 2)
            self.assertEqual(len(a['cell_reference_contribution_ledger']), 3)
            self.assertEqual(a['cell_reference_contribution_ledger'][1]['references'][0]['NAME'], 'door')
            self.assertTrue(next(q for q in a['quests'] if q['id'] == 'old_quest')['modified_earlier_master'])
            self.assertTrue(any(q['id'] == 'unknown' for q in a['quests']))
            self.assertEqual(len(a['unclassified_records']), 1)
            info = next(r for r in b['winning_records'] if r['type'] == 'INFO')
            self.assertEqual(info['plugin'], optional.name)
            self.assertTrue(all(c['status'] == 'UNVERIFIED' for c in a['cells']))
            self.assertFalse(a['release_gate_passed'])

    def test_truncated_record_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'broken.esm'
            p.write_bytes(b'CELL' + struct.pack('<III', 12, 0, 0) + b'x')
            with self.assertRaisesRegex(ValueError, 'truncated record body'):
                list(audit.records(p))

    def test_optional_delta_requires_winning_optional_record_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label in ('A', 'B'):
                (root / label).mkdir()
                audit.write(root / label / 'certification.json', {
                    'counts': {}, 'verification': {'quests': {'non_verified_content_ids': []}}})
            audit.write(root / 'A/winning-record-manifest.json', {'DIAL:q': {'sha256': 'before', 'plugin': 'base.esm'}})
            audit.write(root / 'B/winning-record-manifest.json', {'DIAL:q': {'sha256': 'after', 'plugin': 'TR_Factions.esp'}})
            result = tr_certify.delta(root)
            self.assertTrue(result['record_changes_match_optional_plugin_provenance'])
            self.assertFalse(result['behavior_verified'])
            audit.write(root / 'B/winning-record-manifest.json', {'DIAL:q': {'sha256': 'after', 'plugin': 'base.esm'}})
            self.assertFalse(tr_certify.delta(root)['record_changes_match_optional_plugin_provenance'])

    def test_journal_comments_not_candidates(self):
        self.assertEqual(audit.journal_ids('; Journal fake 10\nJournal "real" 20'), ['real'])
        self.assertEqual(audit.journal_ids('MessageBox "journal fake 20; not a command"\nJournal "real;name" 20'), ['real;name'])
        self.assertEqual(audit.journal_ids('if ( GetJournalIndex "real" >= 10 )'), ['real'])


class CertificationFaults(unittest.TestCase):
    def setUp(self):
        self.inv = {'inventory_complete': True, 'limitations': [],
                    'cells': [{'identity': 'CELL:room'}], 'quests': [{'id': 'quest', 'in_scope': True}],
                    'coverage_manifest': {'cells': ['CELL:room'], 'quests': ['quest']}}
        self.scenarios = []
        for kind, identity in [('cell', 'CELL:room'), ('quest', 'quest')]:
            self.scenarios.append({'kind': kind, 'id': identity, 'status': 'VERIFIED', 'outcome': 'passed',
                'actions': [{'input_path': 'ordinary_gameplay', 'before': {}, 'player_input': 'activate',
                             'runtime_events': [], 'after': {}, 'trace_id': 'synthetic', 'provenance': []}],
                'prerequisites_verified': True, 'significant_gates_verified': True, 'required_checks_passed': True,
                'one_time_effects_verified': True, 'unique_reward_count': 1,
                'reference_identity_verified': True, 'transitions_verified': True,
                'restart': {'terminated_process': True, 'uninterrupted_inputs': ['activate'],
                            'resumed_inputs': ['activate'], 'uninterrupted_state': {'local': 1}, 'resumed_state': {'local': 1}}})
        self.assertEqual(audit.certification_errors(self.inv, self.scenarios), [])

    def test_omitted_inventory_entry(self):
        self.inv['cells'] = []
        self.scenarios.pop(0)
        self.assertTrue(audit.certification_errors(self.inv, self.scenarios))

    def test_prerequisite_bypass(self):
        self.scenarios[1]['prerequisites_verified'] = False
        self.assertTrue(audit.certification_errors(self.inv, self.scenarios))

    def test_duplicated_unique_reward(self):
        self.scenarios[1]['unique_reward_count'] = 2
        self.assertTrue(audit.certification_errors(self.inv, self.scenarios))

    def test_persistent_local_disappears(self):
        self.scenarios[1]['restart']['resumed_state'] = {'local': 0}
        self.assertTrue(audit.certification_errors(self.inv, self.scenarios))

    def test_wrong_reference(self):
        self.scenarios[0]['reference_identity_verified'] = False
        self.assertTrue(audit.certification_errors(self.inv, self.scenarios))

    def test_missing_transition(self):
        self.scenarios[0]['transitions_verified'] = False
        self.assertTrue(audit.certification_errors(self.inv, self.scenarios))

    def test_skipped_timeout_crashed(self):
        for outcome in ['skipped', 'timeout', 'crashed']:
            scenarios = copy.deepcopy(self.scenarios)
            scenarios[0]['outcome'] = outcome
            self.assertTrue(audit.certification_errors(self.inv, scenarios))
        self.assertTrue(audit.certification_errors(self.inv, self.scenarios[:1]))

    def test_in_memory_reload_is_insufficient(self):
        self.scenarios[1]['restart']['terminated_process'] = False
        self.assertTrue(audit.certification_errors(self.inv, self.scenarios))

    def test_direct_mutation_input_rejected(self):
        self.scenarios[1]['actions'][0]['input_path'] = 'journal_setter'
        self.assertTrue(audit.certification_errors(self.inv, self.scenarios))


if __name__ == '__main__':
    unittest.main()
