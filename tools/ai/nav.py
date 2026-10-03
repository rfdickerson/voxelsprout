#!/usr/bin/env python3
"""Configured CMake/CTest navigation. No third-party source traversal."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tools/ai'

def emit(value):
    print(json.dumps(value, indent=2, sort_keys=True))

def run(args):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=True).stdout

def generate(build):
    reply = build / '.cmake/api/v1/reply'
    indexes = sorted(reply.glob('index-*.json'))
    if not indexes:
        raise SystemExit('Request codemodel-v2 then configure; see tools/ai/README.md')
    if indexes[-1].stat().st_mtime < (ROOT / 'CMakeLists.txt').stat().st_mtime:
        raise SystemExit('CMakeLists.txt is newer than configured File API; configure first.')
    index = json.loads(indexes[-1].read_text())
    model = json.loads((reply / next(o['jsonFile'] for o in index['objects'] if o['kind'] == 'codemodel')).read_text())
    config = model['configurations'][0]
    ids = {t['id']: t['name'] for t in config['targets']}
    targets = {}
    for ref in config['targets']:
        if not (ref['name'].startswith('odai') or ref['name'] == 'slang_shaders'):
            continue
        t = json.loads((reply / ref['jsonFile']).read_text())
        targets[t['name']] = {'type': t['type'], 'sources': sorted(s['path'] for s in t.get('sources', []) if not s.get('isGenerated')), 'generated': sorted(s['path'] for s in t.get('sources', []) if s.get('isGenerated')), 'dependencies': sorted(ids[d['id']] for d in t.get('dependencies', [])), 'link_fragments': [f['fragment'].replace(str(ROOT), '<repo>') for f in t.get('link', {}).get('commandFragments', []) if f['role'] == 'libraries']}
    owners = {}
    for name, t in targets.items():
        for source in t['sources']:
            owners.setdefault(source, []).append(name)
    tests = json.loads(run(['ctest', '--test-dir', str(build), '--show-only=json-v1']))['tests']
    includes = {}
    # Only direct include lines in target-listed translation units and their local
    # header closure; retain edges without loading full source into agent context.
    pending = list(owners)
    while pending:
        source = pending.pop()
        if source in includes or not (ROOT / source).is_file():
            continue
        local = []
        for inc in re.findall(r'^\s*#\s*include\s*"([^\"]+)"', (ROOT / source).read_text(errors='replace'), re.M):
            candidates = [ROOT / 'src' / inc, (ROOT / source).parent / inc]
            for p in candidates:
                if p.is_file() and p.is_relative_to(ROOT / 'src'):
                    rel = str(p.relative_to(ROOT))
                    local.append(rel)
                    pending.append(rel)
                    break
        includes[source] = sorted(set(local))
    def closure(source):
        seen, todo = set(), [source]
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            todo.extend(includes.get(p, []))
        return seen
    mapped = []
    for test in tests:
        command = [x.replace(str(ROOT), '<repo>') for x in test.get('command', [])]
        reached = sorted(n for n in targets if any(Path(x).name == n for x in test.get('command', [])))
        source_set = set()
        for n in reached:
            for s in targets[n]['sources']:
                source_set.update(closure(s))
        for x in test.get('command', []):
            p = Path(x)
            if p.is_file() and p.is_relative_to(ROOT) and p.suffix in ('.py', '.json'):
                source_set.add(str(p.relative_to(ROOT)))
        mapped.append({'name': test['name'], 'targets': reached, 'command': command, 'direct_test_includes': sorted({i for n in reached for s in targets[n]['sources'] if s.startswith(('tests/', 'src/render/tests/')) for i in includes.get(s, [])}), 'source_evidence': sorted(source_set), 'properties': test.get('properties', [])})
    provenance = {'schema_version': 1, 'build': str(build.relative_to(ROOT)), 'configuration': config['name'], 'cmake_sha256': hashlib.sha256((ROOT / 'CMakeLists.txt').read_bytes()).hexdigest(), 'scope': 'Configured targets only; include closure is textual, test association is not coverage.'}
    for filename, data in [('repo-map.json', {'targets': targets, 'source_owners': owners}), ('dependency-map.json', {'target_edges': {n:t['dependencies'] for n,t in targets.items()}, 'local_includes': includes}), ('test-map.json', {'tests': mapped})]:
        (OUT / filename).write_text(json.dumps({**provenance, **data}, indent=2, sort_keys=True).replace(str(ROOT), '<repo>') + '\n')
    emit({'targets': len(targets), 'tests': len(mapped), 'include_files': len(includes)})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', default='build-linux')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('generate')
    for c in ('owner', 'neighbors', 'includers', 'reaches', 'tests', 'check', 'run-tests'):
        q = sub.add_parser(c)
        q.add_argument('item', help='exact target or repository-relative file')
    q = sub.add_parser('find')
    q.add_argument('symbol')
    q.add_argument('--scope', required=True, help='target or file; literal references/declarations, not semantic callers')
    a = p.parse_args()
    build = ROOT / a.build
    if a.command == 'generate':
        generate(build)
        return
    repo = json.loads((OUT / 'repo-map.json').read_text())
    if repo['cmake_sha256'] != hashlib.sha256((ROOT / 'CMakeLists.txt').read_bytes()).hexdigest() or repo['build'] != a.build:
        raise SystemExit('Map differs from requested build/current CMakeLists.txt; regenerate.')
    targets = repo['targets']
    deps = json.loads((OUT / 'dependency-map.json').read_text())
    test_map = json.loads((OUT / 'test-map.json').read_text())
    for mapped in (deps, test_map):
        if any(mapped.get(k) != repo.get(k) for k in ('schema_version', 'build', 'configuration', 'cmake_sha256')):
            raise SystemExit('Navigation maps have inconsistent provenance; regenerate.')
    if a.command == 'find':
        files = set(targets[a.scope]['sources'] if a.scope in targets else [a.scope])
        todo = list(files)
        while todo:
            for f in deps['local_includes'].get(todo.pop(), []):
                if f not in files:
                    files.add(f)
                    todo.append(f)
        result = subprocess.run(['rg', '-n', '-F', '--', a.symbol, *sorted(files)], cwd=ROOT, text=True, capture_output=True)
        if result.returncode not in (0, 1):
            raise SystemExit(result.stderr)
        print(result.stdout, end='')
        return
    owned = [a.item] if a.item in targets else repo['source_owners'].get(a.item, [])
    if a.command == 'includers':
        emit({'file': a.item, 'direct_includers': [s for s,v in deps['local_includes'].items() if a.item in v], 'note': 'Textual quoted includes, not semantic callers.'})
    elif a.command == 'reaches':
        def reaches(n):
            seen, pending = set(), [n]
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                if current in targets:
                    if current == a.item or a.item in targets[current]['sources']:
                        return True
                    files = list(targets[current]['sources'])
                    included = set()
                    while files:
                        f = files.pop()
                        if f in included:
                            continue
                        included.add(f)
                        files.extend(deps['local_includes'].get(f, []))
                    if a.item in included:
                        return True
                    pending.extend(targets[current]['dependencies'])
            return False
        emit({'item': a.item, 'executables': [n for n,t in targets.items() if t['type'] == 'EXECUTABLE' and reaches(n)], 'note': 'Configured build/include reachability, not proof of runtime execution.'})
    elif a.command == 'owner':
        emit({'file': a.item, 'compiled_by': owned, 'direct_includers': [s for s,v in deps['local_includes'].items() if a.item in v], 'note': 'Headers can be shared; no compiled owner does not mean unused.'})
    elif a.command == 'neighbors':
        emit({n: {'dependencies': targets[n]['dependencies'], 'dependents': [k for k,v in targets.items() if n in v['dependencies']]} for n in owned})
    elif a.command == 'check':
        if not owned:
            raise SystemExit('Use owner to identify a compile target for this header.')
        subprocess.run(['cmake', '--build', str(build), '--target', *owned, '-j', '2'], cwd=ROOT, check=True)
    else:
        tests = test_map['tests']
        chosen = [t for t in tests if a.item in t['source_evidence'] or a.item in t['targets'] or any(a.item in targets[n]['dependencies'] for n in t['targets'])]
        if a.command == 'tests':
            emit([{'name':t['name'], 'targets':t['targets'], 'evidence': 'direct test include' if a.item in t.get('direct_test_includes', []) else 'source/include or configured target dependency'} for t in chosen])
        else:
            if not chosen:
                raise SystemExit('No associated tests; expand to owning target.')
            compile_targets = sorted({n for t in chosen for n in t['targets']})
            if compile_targets:
                subprocess.run(['cmake', '--build', str(build), '--target', *compile_targets, '-j', '2'], cwd=ROOT, check=True)
            pattern = '^(' + '|'.join(re.escape(t['name']) for t in chosen) + ')$'
            subprocess.run(['ctest', '--test-dir', str(build), '-R', pattern, '--no-tests=error', '--output-on-failure'], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
