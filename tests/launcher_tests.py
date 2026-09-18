import importlib.machinery
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader('launcher', str(ROOT / 'scripts/odai-launcher'))
spec = importlib.util.spec_from_loader(loader.name, loader)
launcher = importlib.util.module_from_spec(spec)
loader.exec_module(launcher)

class SetupTests(unittest.TestCase):
    def test_backup_only_slot_is_recoverable(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            backup = directory / 'interrupted.odai.json.previous'
            backup.write_text('{}')
            self.assertEqual(launcher.save_slots(directory), [directory / 'interrupted.odai.json'])
            (directory / 'interrupted.odai.json').write_text('{}')
            self.assertEqual(len(launcher.save_slots(directory)), 1)

    def test_required_content_and_order(self):
        manifest = json.loads((ROOT / 'packaging/skyrim-slice.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            roots = [Path(temp) / name for name in ('data with spaces', 'smim', 'jk')]
            for root in roots:
                root.mkdir()
            with self.assertRaisesRegex(ValueError, 'Skyrim.esm'):
                launcher.make_profile(manifest, *roots)
            for plugin in manifest['plugins']:
                index = 1 if plugin == 'SMIM-SE-Merged-All.esp' else 2 if plugin == 'JKs Skyrim.esp' else 0
                (roots[index] / plugin).touch()
            mapping = {'base-data': roots[0], manifest['mods'][0]['id']: roots[1], manifest['mods'][1]['id']: roots[2]}
            for archive in manifest['archives']:
                (mapping[archive['layer']] / archive['file']).touch()
            for name in ('meshes', 'textures'):
                (roots[1] / name).mkdir()
            profile = launcher.make_profile(manifest, *roots)
            self.assertEqual(profile['plugins'][-2:], ['SMIM-SE-Merged-All.esp', 'JKs Skyrim.esp'])
            self.assertTrue(all(a['required'] for a in profile['archives']))
            path = Path(temp) / 'config/profile.json'
            launcher.atomic_json(path, profile)
            self.assertEqual(json.loads(path.read_text()), profile)
            (roots[0] / '_ResourcePack.esl').unlink()
            with self.assertRaisesRegex(ValueError, '_ResourcePack.esl'):
                launcher.make_profile(manifest, *roots)

if __name__ == '__main__':
    unittest.main()
