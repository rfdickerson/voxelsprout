#!/usr/bin/env python3
"""Fail-closed TES3 release inventory. Metadata stays local; never certifies gameplay.

The binary-record audit is independent of the engine's structural probes. It
retains every record occurrence and unknown identity instead of trusting that a
runtime parser enumerated everything. Quest inference is conservative and is
explicitly not a complete authored route classifier.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text(data):
    return data.split(b'\0', 1)[0].decode('cp1252', errors='replace')


def records(path):
    """Strict bounds checking; offsets make malformed input reproducible."""
    with Path(path).open('rb') as f:
        offset = 0
        while head := f.read(16):
            if len(head) != 16:
                raise ValueError(f'{path}:{offset}: truncated record header')
            kind, size, _, flags = struct.unpack('<4sIII', head)
            body = f.read(size)
            if len(body) != size:
                raise ValueError(f'{path}:{offset}: truncated record body')
            subs, cursor = [], 0
            while cursor < size:
                if cursor + 8 > size:
                    raise ValueError(f'{path}:{offset + 16 + cursor}: truncated subrecord header')
                tag, length = struct.unpack_from('<4sI', body, cursor)
                start = cursor + 8
                if start + length > size:
                    raise ValueError(f'{path}:{offset + 16 + cursor}: truncated subrecord body')
                subs.append((tag.decode('ascii'), body[start:start + length]))
                cursor = start + length
            yield kind.decode('ascii'), flags, subs, offset, hashlib.sha256(body).hexdigest()
            offset += 16 + size


def first(subs, tag):
    return next((value for name, value in subs if name == tag), b'')


def cell_identity(subs):
    # Reference DATA/NAME must never replace the CELL's own header.
    header = []
    for pair in subs:
        if pair[0] in ('FRMR', 'MVRF'):
            break
        header.append(pair)
    data = first(header, 'DATA')
    if len(data) != 12:
        return None, {'classification': 'MALFORMED', 'reason': 'CELL DATA is not 12 bytes'}
    flags, x, y = struct.unpack('<Iii', data)
    name = text(first(header, 'NAME'))
    if flags & 1:
        return ('CELL', 'interior:' + name.lower()) if name else None, {
            'classification': 'interior', 'name': name}
    return ('CELL', f'exterior:{x},{y}'), {'classification': 'exterior', 'name': name, 'grid': [x, y]}


JOURNAL = re.compile(r'\b(?:journal|setjournalindex|getjournalindex)\s*,?\s*(?:"([^"]+)"|([\w\-]+))', re.I)


def journal_ids(source):
    # A keyword in a comment or quoted MessageBox string is not an operation.
    # Keep quoted operands, but only recognize keyword positions outside quotes.
    found = set()
    for line in source.splitlines():
        quoted = False
        for i, char in enumerate(line):
            if char == '"':
                quoted = not quoted
            elif char == ';' and not quoted:
                line = line[:i]
                break
        for match in JOURNAL.finditer(line):
            if line[:match.start()].count('"') % 2 == 0:
                found.add((match.group(1) or match.group(2)).lower())
    return sorted(found)


def inventory(plugin_paths, scope_plugins):
    scope = {name.lower() for name in scope_plugins}
    winners, occurrences, anomalies, touched, history = {}, [], [], set(), collections.defaultdict(list)
    topic_types = {}
    cell_contributions = []
    for path in plugin_paths:
        plugin = Path(path).name
        topic = None
        for kind, flags, subs, offset, sha in records(path):
            metadata = {'plugin': plugin, 'offset': offset, 'type': kind, 'sha256': sha,
                        'subrecords': dict(collections.Counter(tag for tag, _ in subs))}
            key, detail = None, {}
            if kind == 'TES3':
                continue
            if kind == 'CELL':
                key, detail = cell_identity(subs)
            elif kind == 'SCPT':
                name = text(first(subs, 'SCHD')[:32])
                key = (kind, name.lower()) if name else None
            elif kind == 'INFO':
                name = text(first(subs, 'INAM'))
                key = (kind, topic + ':' + name.lower()) if topic and name else None
                detail['topic'] = topic
            elif kind == 'LAND':
                data = first(subs, 'INTV')
                if len(data) == 8:
                    key = (kind, '%d,%d' % struct.unpack('<ii', data))
            elif kind == 'LTEX':
                data = first(subs, 'INTV')
                if len(data) == 4:
                    key = (kind, plugin.lower() + ':' + str(struct.unpack('<I', data)[0]))
            elif kind == 'PGRD':
                # PGRD has mixed interior/exterior identity; retain all rather
                # than collapse exterior records sharing an empty NAME.
                name = text(first(subs, 'NAME'))
                data = first(subs, 'DATA')
                if len(data) >= 8:
                    key = (kind, name.lower() or ('%d,%d' % struct.unpack('<ii', data[:8])))
            else:
                name = text(first(subs, 'NAME'))
                key = (kind, name.lower()) if name else None
            if kind == 'DIAL':
                topic = text(first(subs, 'NAME')).lower() or None
                data = first(subs, 'DATA')
                if topic and len(data) == 1:
                    topic_types[topic] = data[0]
            # DELE in a CELL reference is not a deleted CELL.
            header = []
            for pair in subs:
                if kind == 'CELL' and pair[0] in ('FRMR', 'MVRF'):
                    break
                header.append(pair)
            deleted = any(tag == 'DELE' for tag, _ in header)
            metadata.update(detail)
            metadata['deleted'] = deleted
            metadata['identity'] = ':'.join(key) if key else None
            occurrences.append(metadata.copy())
            if key is None:
                anomalies.append({**metadata, 'reason': 'unclassified or malformed record identity'})
                continue
            history[key].append(metadata.copy())
            if plugin.lower() in scope:
                touched.add(key)
            # Keep deleted winning records visible as tombstones.
            winner = {**metadata, 'status': 'UNVERIFIED'}
            if kind in ('SCPT', 'INFO'):
                source = text(first(subs, 'SCTX' if kind == 'SCPT' else 'BNAM'))
                winner['journal_candidates'] = journal_ids(source)
                winner['source_available'] = bool(source)
                winner['behavior_classification'] = 'UNCLASSIFIED'
                winner['lexical_command_candidates'] = sorted({line.split()[0].lower()
                    for line in source.splitlines() if line.strip() and not line.lstrip().startswith(';')})
            if kind == 'INFO':
                rules = [text(value) for tag, value in subs if tag == 'SCVR']
                winner['conditions'] = rules
                winner['journal_candidates'] = sorted(set(winner['journal_candidates']) |
                    {rule[5:].lower() for rule in rules if len(rule) >= 5 and rule[1:2] == '4'})
                winner['previous'] = text(first(subs, 'PNAM'))
                winner['next'] = text(first(subs, 'NNAM'))
                winner['journal_flags'] = [tag for tag, _ in subs if tag in ('QSTN', 'QSTF', 'QSTR')]
            if kind == 'DIAL':
                winner['dialogue_type'] = topic_types.get(topic)
            if kind == 'CELL':
                # Embedded reference overrides require engine resolution. Keep
                # every occurrence, including moved references, for audit.
                refs, current = [], None
                for tag, value in subs:
                    if tag in ('FRMR', 'MVRF'):
                        current = {'kind': tag, 'raw_identity': value.hex(), 'subrecords': []}
                        refs.append(current)
                    elif current is not None:
                        if tag in ('NAME', 'DNAM'):
                            current[tag] = text(value)
                        if tag in ('DODT', 'CNDT', 'DELE'):
                            current[tag] = value.hex()
                        current['subrecords'].append(tag)
                winner['reference_occurrences'] = refs
                cell_contributions.append({'cell': winner['identity'], 'plugin': plugin, 'offset': offset,
                                           'references': refs})
            if kind in ('NPC_', 'CREA'):
                winner['travel_links'] = []
                for tag, value in subs:
                    if tag == 'DODT':
                        winner['travel_links'].append({'destination_transform': value.hex(), 'cell': ''})
                    elif tag == 'DNAM' and winner['travel_links']:
                        winner['travel_links'][-1]['cell'] = text(value)
            winner['attached_script'] = text(first(subs, 'SCRI'))
            winners[key] = winner
    for key, winner in winners.items():
        winner['contributed_by_scope'] = key in touched
        winner['provenance'] = history[key]
    # Retain ALL active journal records, labeling scope and lexical dependencies.
    # Scripted quests without journals remain in unclassified quest-bearing data.
    quest_candidates = collections.defaultdict(list)
    for key, winner in winners.items():
        if winner['deleted']:
            continue
        for quest in winner.get('journal_candidates', []):
            quest_candidates[quest].append(winner['identity'])
    infos_by_topic = collections.defaultdict(list)
    for k, v in winners.items():
        if k[0] == "INFO" and not v["deleted"]:
            infos_by_topic[v.get("topic")].append(v)
    scoped_ids = {v["identity"] for k, v in winners.items() if k in touched}
    quests = []
    for key, winner in sorted(winners.items()):
        if key[0] != 'DIAL' or winner.get('dialogue_type') != 4 or winner['deleted']:
            continue
        infos = infos_by_topic[key[1]]
        candidates = quest_candidates.pop(key[1], [])
        in_scope = key in touched or any(v['contributed_by_scope'] for v in infos) or bool(scoped_ids.intersection(candidates))
        quests.append({'id': key[1], 'status': 'UNVERIFIED', 'in_scope': in_scope,
                       'modified_earlier_master': in_scope and history[key][0]['plugin'].lower() not in scope,
                       'record': winner['identity'], 'source': winner['plugin'],
                       'infos': [v['identity'] for v in infos], 'candidate_dependencies': candidates})
    for quest, candidates in sorted(quest_candidates.items()):
        quests.append({'id': quest, 'status': 'UNVERIFIED', 'in_scope': bool(scoped_ids.intersection(candidates)),
            'classification': 'UNRESOLVED_JOURNAL_TARGET', 'candidate_dependencies': candidates})
    cells = [v for k, v in sorted(winners.items()) if k[0] == 'CELL' and k in touched and not v['deleted']]
    return {'schema_version': 1, 'inventory_complete': False, 'release_gate_passed': False,
            'limitations': ['quest dependency/branch classifier incomplete',
                             'embedded/moved reference occurrence ledger is not a winning-reference resolver',
                             'authored ordinary-input traversal and quest scenarios not generated or executed'],
            'scope_plugins': scope_plugins, 'cells': cells, 'quests': quests,
            'cell_reference_contribution_ledger': [c for c in cell_contributions if
                ('CELL', c['cell'][5:]) in touched],
            'coverage_manifest': {'cells': sorted(v['identity'] for v in cells),
                                  'quests': sorted(q['id'] for q in quests if q['in_scope'])},
            'winning_records': list(winners.values()), 'record_occurrences': occurrences,
            'unclassified_records': anomalies,
            'unclassified_quest_bearing_records': [v['identity'] for k, v in sorted(winners.items())
                if k in touched and not v['deleted'] and k[0] in ('SCPT', 'INFO', 'NPC_', 'CREA', 'CELL')],
            'counts': {'exterior_cells': sum(v['classification'] == 'exterior' for v in cells),
                       'interior_cells': sum(v['classification'] == 'interior' for v in cells),
                       'scope_quests': sum(q['in_scope'] for q in quests),
                       'active_journal_and_candidate_quests': len(quests)}}


def certification_errors(inv, scenarios):
    """Reject incomplete inventories and uncertified execution evidence.

    This is an evidence schema guard, not an ordinary-input runner. Producers
    must still prove the supplied causal states originated in runtime gameplay.
    """
    errors = list(inv.get('limitations', []))
    expected = {('cell', v['identity']) for v in inv['cells']} | {
        ('quest', q['id']) for q in inv['quests'] if q['in_scope']}
    manifest = inv.get('coverage_manifest', {})
    if (sorted(v['identity'] for v in inv['cells']) != manifest.get('cells') or
        sorted(q['id'] for q in inv['quests'] if q['in_scope']) != manifest.get('quests')):
        errors.append('inventory entry omitted or coverage manifest absent')
    supplied = [(s.get('kind'), s.get('id')) for s in scenarios]
    if set(supplied) != expected or len(supplied) != len(set(supplied)):
        errors.append('scenario coverage differs from inventory')
    if not inv.get('inventory_complete', False):
        errors.append('inventory classifier incomplete')
    for s in scenarios:
        label = str((s.get('kind'), s.get('id')))
        if s.get('status') != 'VERIFIED' or s.get('outcome') != 'passed':
            errors.append(label + ': skipped, failed, crashed, timed out, or unverified')
        actions = s.get('actions', [])
        if not actions or any(a.get('input_path') != 'ordinary_gameplay' or not all(
                field in a for field in ('before', 'player_input', 'runtime_events', 'after', 'trace_id', 'provenance'))
                for a in actions):
            errors.append(label + ': missing causal ordinary-input evidence')
        if not s.get('prerequisites_verified', False):
            errors.append(label + ': prerequisite bypass/unverified checkpoint')
        if not s.get('significant_gates_verified', False) or not s.get('required_checks_passed', False):
            errors.append(label + ': authored gates or required checks incomplete')
        if not s.get('one_time_effects_verified', False) or s.get('unique_reward_count', 0) > 1:
            errors.append(label + ': one-time reward violation/unverified')
        if not s.get('reference_identity_verified', False):
            errors.append(label + ': wrong/unverified reference identity')
        if not s.get('transitions_verified', False):
            errors.append(label + ': expected transition missing/unverified')
        restart = s.get('restart', {})
        if (restart.get('terminated_process') is not True or
            restart.get('uninterrupted_inputs') != restart.get('resumed_inputs') or
            not restart.get('uninterrupted_inputs') or
            restart.get('uninterrupted_state') != restart.get('resumed_state') or
            not restart.get('uninterrupted_state')):
            errors.append(label + ': process restart semantic comparison missing/mismatch')
    return errors


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=None if Path(path).name == 'inventory.json' else 2, sort_keys=True) + '\n')


def run(profile, scope, output, probe):
    output.mkdir(parents=True, exist_ok=True)
    profile = profile.resolve()
    raw = json.loads(profile.read_text())
    roots = [Path(raw['data_root'])] + [Path(layer['path']) for layer in raw.get('layers', []) if layer.get('enabled', True)]
    paths = []
    for name in raw['plugins']:
        found = next((p for root in reversed(roots) for p in root.iterdir() if p.name.lower() == name.lower()), None)
        if found is None:
            raise ValueError('required plugin unavailable: ' + name)
        paths.append(found)
    identity = {'load_order': [{'filename': p.name, 'path': str(p), 'sha256': digest(p)} for p in paths],
                'resource_roots': [str(p) for p in roots], 'profile_sha256': digest(profile),
                'loose_asset_verification': 'UNVERIFIED; roots pinned, required asset resolution not executed',
                'archives': [{'path': a['path'] if isinstance(a, dict) else a,
                              'sha256': digest(a['path'] if isinstance(a, dict) else a)} for a in raw.get('archives', [])],
                'engine_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'working_tree_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(),
                'harness_sha256': digest(__file__), 'probe_sha256': digest(probe)}
    write(output / 'content-identity.json', identity)
    inv = inventory(paths, scope)
    write(output / 'inventory.json', inv)
    write(output / 'winning-record-manifest.json', {v['identity']: {'sha256': v['sha256'], 'plugin': v['plugin'], 'deleted': v['deleted']}
                                                  for v in inv['winning_records']})
    results = []
    for mode, flags in [('--profilecheck', []), ('--tes3-scriptcheck', ['--strict']), ('--tes3-quest-suite', [])]:
        command = [str(probe.resolve()), mode, str(profile), *flags]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            code, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            code, stdout, stderr = None, (e.stdout or b'').decode(), (e.stderr or b'').decode()
        stem = mode[2:]
        (output / (stem + '.stdout')).write_text(stdout)
        (output / (stem + '.stderr')).write_text(stderr)
        results.append({'command': command, 'exit_code': code, 'stdout': stem + '.stdout', 'stderr': stem + '.stderr'})
        write(output / 'probe-results.json', results)
    try:
        runtime_quests = json.loads((output / 'tes3-quest-suite.stdout').read_text())
        actual = {q['id'].lower() for q in runtime_quests.get('quests', [])}
        discovered = {q['id'] for q in inv['quests'] if q.get('classification') != 'UNRESOLVED_JOURNAL_TARGET'}
        reconciliation = {'runtime_only': sorted(actual - discovered), 'audit_only': sorted(discovered - actual),
                          'matched': actual == discovered, 'runtime_count': len(actual), 'audit_count': len(discovered)}
    except (ValueError, OSError):
        reconciliation = {'matched': False, 'error': 'runtime quest suite unavailable or malformed'}
    write(output / 'journal-reconciliation.json', reconciliation)
    blockers = [{'id': 'gameplay-cell-coverage', 'subsystem': 'harness defects',
                 'affected_content_ids': [v['identity'] for v in inv['cells']],
                 'observed': 'No ordinary-input traversal scenario execution evidence',
                 'expected': 'All applicable cell checks and authored return/onward route pass'},
                {'id': 'gameplay-quest-coverage', 'subsystem': 'harness defects',
                 'affected_content_ids': [q['id'] for q in inv['quests'] if q['in_scope']],
                 'observed': 'Structural quest suite explicitly lacks transition explorer; no executed ordinary-gameplay routes',
                 'expected': 'Authored gates, resolution, causal trace, one-time effects and process restart pass'},
                {'id': 'inventory-classifier', 'subsystem': 'harness defects',
                 'affected_content_ids': inv['unclassified_quest_bearing_records'],
                 'observed': inv['limitations'], 'expected': 'Complete branch/dependency classification and winning reference linkage'}]
    for r in results:
        try:
            parsed = json.loads((output / r['stdout']).read_text())
        except ValueError:
            parsed = {'error': 'probe output is not JSON'}
        if r['exit_code'] != 0 or parsed.get('strict_pass') is False:
            blockers.append({'id': Path(r['stdout']).stem, 'subsystem': 'MWScript' if 'scriptcheck' in r['stdout'] else 'TES3 record loading / overrides',
                             'affected_content_ids': [q['id'] for q in inv['quests'] if q['in_scope']],
                             'affected_ids_precision': 'conservative scope; per-quest impact not established',
                             'reproduction': r['command'], 'observed_evidence': r['stdout'],
                             'observed': parsed, 'expected': 'Profile load and supported script closure without errors'})
    if not reconciliation['matched']:
        blockers.append({'id': 'journal-inventory-mismatch', 'subsystem': 'TES3 record loading / overrides',
                         'affected_content_ids': reconciliation.get('runtime_only', []) + reconciliation.get('audit_only', []),
                         'observed': reconciliation, 'expected': 'Independent and runtime winning journal enumeration match'})
    blockers.extend([
        {'id': 'persistence-process-restart', 'subsystem': 'save/load',
         'affected_content_ids': [v['identity'] for v in inv['cells']] + [q['id'] for q in inv['quests'] if q['in_scope']],
         'observed': 'No installed-profile save/terminate/reload/subsequent-input semantic comparison executed',
         'expected': 'Journal, locals/globals, inventory, factions, actor/reference state and unique rewards survive process restart'},
        {'id': 'required-assets-and-playability', 'subsystem': 'asset resolution / terrain / collision / actors / streaming',
         'affected_content_ids': [v['identity'] for v in inv['cells']],
         'observed': 'Profile resolution does not verify assets or movement, collision, terrain, water, actors, doors and travel',
         'expected': 'Execute all applicable authored cell checks through ordinary player input'},
        {'id': 'validator-runtime-fault-tests', 'subsystem': 'harness defects',
         'affected_content_ids': [q['id'] for q in inv['quests'] if q['in_scope']],
         'observed': 'Synthetic evidence-schema faults pass; installed-content gameplay runner and runtime fault injection absent',
         'expected': 'Fault tests must demonstrate actual gameplay runner rejects each required semantic fault'}])
    for b in blockers:
        b.setdefault('reproduction', ['python3', 'scripts/tes3_release_audit.py', '--profile', str(profile), *[arg for name in scope for arg in ('--scope-plugin', name)], '--output', str(output), '--probe', str(probe)])
    report = {'capability': 'MOD-TES3-001', 'status': 'Partial', 'content_identity': identity,
              'counts': inv['counts'], 'inventory_complete': False, 'release_gate_passed': False,
              'verification': {kind: {'VERIFIED': 0, 'BLOCKED': 0, 'UNVERIFIED': len(ids), 'non_verified_content_ids': ids}
                               for kind, ids in [('cells', [v['identity'] for v in inv['cells']]),
                                                 ('quests', [q['id'] for q in inv['quests'] if q['in_scope']])]},
              'blockers': blockers, 'engine_defects_fixed': [], 'test_results': results,
              'unclassified_content_ids': [v['plugin'] + ':' + str(v['offset']) for v in inv['unclassified_records']],
              'certification_errors': certification_errors(inv, [])}
    write(output / 'certification.json', report)
    print(json.dumps({'output': str(output), 'status': 'Partial', 'counts': inv['counts']}))
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--scope-plugin', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', type=Path, default=Path('build-linux/odai_bethesda_probe'))
    args = parser.parse_args()
    try:
        return run(args.profile, args.scope_plugin, args.output, args.probe)
    except (OSError, ValueError, struct.error) as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output / 'fatal.json', {'status': 'Partial', 'release_gate_passed': False, 'error': str(error)})
        print(str(error))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
