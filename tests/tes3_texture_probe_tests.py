#!/usr/bin/env python3
"""Strict texture audit with synthetic TES3 NIF/LAND/LTEX assets only."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

PROBE = str(Path(sys.argv.pop(1)).resolve())

def sub(tag, value):
    if isinstance(value, str):
        value = value.encode("cp1252") + b"\0"
    return tag.encode() + struct.pack("<I", len(value)) + value


def record(tag, *parts):
    body = b"".join(parts)
    return tag.encode() + struct.pack("<III", len(body), 0, 0) + body


def header(master=None):
    parts = [sub("HEDR", struct.pack("<fI", 1.3, master is None) + bytes(288) + struct.pack("<I", 0))]
    if master:
        parts += [sub("MAST", master), sub("DATA", bytes(8))]
    return record("TES3", *parts)


def nif_string(value):
    value = value.encode()
    return struct.pack("<I", len(value)) + value


def nif_file(blocks):
    return b"NetImmerse File Format, Version 4.0.0.2\n" + struct.pack("<II", 0x04000002, len(blocks)) + \
        b"".join(nif_string(name) + data for name, data in blocks) + struct.pack("<Ii", 1, 0)


def av_object(name, controller=-1):
    return nif_string(name) + struct.pack("<iiH", -1, controller, 0) + struct.pack("<3f", 0, 0, 0) + \
        struct.pack("<9f", 1, 0, 0, 0, 1, 0, 0, 0, 1) + struct.pack("<f3fII", 1, 0, 0, 0, 0, 0)


class TextureChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = self.root / 'profile.json'

    def test_strict_object_and_terrain_texture_audit(self):
        # Independent TES3 fixture includes an authored object request and a
        # LAND/LTEX texture. Removing either must fail, even if fallback exists.
        geometry = struct.pack("<HI9fI9f4fIH I6fHI3HH", 3, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 3, 0, 1, 2, 0)
        shape = av_object("textured")
        # Replace the empty property count; bounding flag remains zero.
        shape = shape[:-8] + struct.pack("<IiIii", 1, 2, 0, 1, -1)
        prop = nif_string("") + struct.pack("<iiHIIIiIIIIH", -1, -1, 0, 2, 1, 1, 3, 3, 2, 0, 0, 0)
        source = nif_string("") + struct.pack("<iiB", -1, -1, 1) + nif_string("object.tga") + struct.pack("<IIIB", 0, 0, 0, 1)
        mesh = nif_file([("NiTriShape", shape), ("NiTriShapeData", geometry),
                         ("NiTexturingProperty", prop), ("NiSourceTexture", source)])
        (self.root / 'meshes').mkdir()
        (self.root / 'meshes/texture.nif').write_bytes(mesh)
        (self.root / 'textures').mkdir()
        # Real top-origin RGB TGA, no mips; normal loader must generate them.
        tga = bytearray(18); tga[2] = 2; tga[12] = 4; tga[14] = 4; tga[16] = 24; tga[17] = 32
        tga += bytes([0, 128, 255]) * 16
        for name in ('object.tga', 'ground.tga', '_land_default.tga'):
            (self.root / 'textures' / name).write_bytes(tga)
        plugin = header() + record('STAT', sub('NAME', 'texture_stat'), sub('MODL', 'texture.nif'))
        plugin += record('LTEX', sub('NAME', 'ground'), sub('INTV', struct.pack('<i', 0)), sub('DATA', 'ground.tga'))
        for name, flags in [('texture room', 1), ('', 0)]:
            plugin += record('CELL', sub('NAME', name), sub('DATA', struct.pack('<Iii', flags, 0, 0)),
                sub('FRMR', struct.pack('<I', 1 if flags else 2)), sub('NAME', 'texture_stat'), sub('DATA', bytes(24)))
        plugin += record('LAND', sub('INTV', bytes(8)), sub('DATA', struct.pack('<I', 3)),
            sub('VHGT', struct.pack('<f', 0) + bytes(65 * 65 + 3)), sub('VTEX', struct.pack('<256H', *([1] * 256))))
        (self.root / 'textures.esm').write_bytes(plugin)
        self.profile.write_text(json.dumps({'version': 1, 'game': 'Morrowind', 'data_root': str(self.root),
            'plugins': ['textures.esm'], 'archives': []}))
        plan = {'version': 1, 'checks': [
            {'kind': 'scene-textures', 'id': 'texture room', 'measure_cached': True, 'expected': {'parsed': True, 'missing_textures': 0, 'mesh_failures': 0, 'texture_count': 1, 'cached_bindings_valid': True, 'cache_misses': 1, 'cache_hits': 1}},
            {'kind': 'scene-textures', 'id': 'terrain', 'grid': [0, 0], 'expected': {'parsed': True, 'missing_textures': 0, 'terrain_cells': 1}}]}
        result, report = self.run_probe(plan)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        for name in ('object.tga', 'ground.tga'):
            path = self.root / 'textures' / name
            path.unlink()
            result, report = self.run_probe(plan)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(report['ok'])
            path.write_bytes(tga)

    def run_probe(self, plan):
        expectations = self.root / 'expectations.json'
        expectations.write_text(json.dumps(plan))
        result = subprocess.run([PROBE, '--tes3-texturecheck', str(self.profile), str(expectations)],
                                text=True, capture_output=True, timeout=30)
        return result, json.loads(result.stdout)


if __name__ == '__main__':
    unittest.main()
