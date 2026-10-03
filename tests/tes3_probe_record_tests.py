#!/usr/bin/env python3
"""Check the production TES3 importer through probe JSON, without retail data."""
import copy
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


def global_record(name, kind, value):
    return record("GLOB", sub("NAME", name), sub("FNAM", kind.encode()), sub("FLTV", struct.pack("<f", value)))


def info(name, value, previous="", next_id=""):
    return record("INFO", sub("INAM", name), sub("PNAM", previous), sub("NNAM", next_id),
                  sub("DATA", struct.pack("<iibbbb", 0, value, -1, -1, -1, 0)),
                  sub("SCVR", "02000tr_global"), sub("INTV", struct.pack("<i", 427)),
                  sub("NAME", "Synthetic response."))


def nif_string(value):
    value = value.encode()
    return struct.pack("<I", len(value)) + value


def nif_file(blocks):
    return b"NetImmerse File Format, Version 4.0.0.2\n" + struct.pack("<II", 0x04000002, len(blocks)) + \
        b"".join(nif_string(name) + data for name, data in blocks) + struct.pack("<Ii", 1, 0)


def av_object(name, controller=-1):
    return nif_string(name) + struct.pack("<iiH", -1, controller, 0) + struct.pack("<3f", 0, 0, 0) + \
        struct.pack("<9f", 1, 0, 0, 0, 1, 0, 0, 0, 1) + struct.pack("<f3fII", 1, 0, 0, 0, 0, 0)


def dds_file(width=4, height=4):
    # One 4x4 BC1 block, one mip. The production decoder retains BC bytes.
    payload = ((width + 3) // 4) * ((height + 3) // 4) * 8
    words = [124, 0x81007, height, width, payload, 0, 1] + [0] * 11
    words += [32, 4, int.from_bytes(b"DXT1", "little"), 0, 0, 0, 0, 0]
    words += [0x1000, 0, 0, 0, 0]
    return b"DDS " + struct.pack("<31I", *words) + bytes(payload)


def bsa_file(entries):
    names, name_offsets, data, table = b"", [], b"", b""
    for name, content in entries:
        name_offsets.append(len(names))
        names += name.encode() + b"\0"
        table += struct.pack("<II", len(content), len(data))
        data += content
    return struct.pack("<III", 0x100, len(entries) * 12 + len(names), len(entries)) + table + \
        struct.pack("<" + "I" * len(entries), *name_offsets) + names + bytes(8 * len(entries)) + data


class RecordChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        stats = bytearray(52)
        struct.pack_into("<h", stats, 0, 8)
        stats[2] = 55
        stats[10] = 42
        struct.pack_into("<hhh", stats, 38, 75, 60, 90)
        stats[44:47] = bytes([65, 12, 3])
        faction_data = struct.pack("<2i", 0, 1) + struct.pack("<5i", 30, 35, 40, 30, 10) * 10
        faction_data += struct.pack("<8i", 0, 1, 2, -1, -1, -1, -1, 0)
        script_source = "begin tr_script\nshort state\nend"
        base = header() + global_record("tr_global", "s", 427.9)
        base += global_record("bad_short", "s", float("nan")) + global_record("long_value", "l", -123.9)
        base += record("NPC_", sub("NAME", "tr_actor"), sub("NPDT", stats), sub("FLAG", struct.pack("<I", 1)),
                       sub("RNAM", "dark elf"), sub("CNAM", "priest"), sub("ANAM", "temple"),
                       sub("SCRI", "tr_script"), sub("NPCO", struct.pack("<i32s", 2, b"tr_robe")))
        base += record("FACT", sub("NAME", "temple"), sub("FADT", faction_data),
                       sub("ANAM", "blades"), sub("INTV", struct.pack("<i", -10)))
        base += record("CLOT", sub("NAME", "tr_robe"), sub("CTDT", struct.pack("<ifHH", 4, 1, 200, 0)))
        base += record("SCPT", sub("SCHD", struct.pack("<32s5I", b"tr_script", 1, 0, 0, 3, 6)),
                       sub("SCVR", b"state\0"), sub("SCDT", b"\1\2\3"), sub("SCTX", script_source))
        base += record("SPEL", sub("NAME", "tr_spell"), sub("SPDT", struct.pack("<iii", 2, 7, 1)),
                       sub("ENAM", struct.pack("<hbb5i", 79, -1, 0, 0, 0, 30, 2, 4)))
        base += record("DIAL", sub("NAME", "tr_topic"), sub("DATA", b"\0"))
        base += info("a", 60, next_id="b") + info("b", 20, previous="a")
        base += record("CELL", sub("NAME", "room"), sub("DATA", struct.pack("<Iii", 1, 0, 0)),
                       sub("FRMR", struct.pack("<I", 42)), sub("NAME", "tr_actor"),
                       sub("DATA", struct.pack("<6f", 1, 2, 3, 0, 0, 0)))
        base += record("STAT", sub("NAME", "removed"))
        patch = header("base.esm") + record("DIAL", sub("NAME", "tr_topic"), sub("DATA", b"\0"))
        patch += info("a", 70, next_id="b")
        patch += record("CELL", sub("NAME", "room"), sub("DATA", struct.pack("<Iii", 1, 0, 0)),
                        sub("FRMR", struct.pack("<I", 0x0100002a)), sub("NAME", "tr_actor"),
                        sub("INTV", struct.pack("<i", 37)), sub("DATA", struct.pack("<6f", 4, 5, 6, 0, 0, 0)))
        patch += record("STAT", sub("NAME", "removed"), sub("DELE", bytes(4)))
        (self.root / "base.esm").write_bytes(base)
        (self.root / "patch.esp").write_bytes(patch)
        geometry = struct.pack("<HI9fI9f4fIH I6fHI3HH", 3, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 3, 0, 1, 2, 0)
        mesh = nif_file([("NiTriShape", av_object("triangle") + struct.pack("<ii", 1, -1)),
                         ("NiTriShapeData", geometry)])
        node = av_object("branch", 1) + struct.pack("<II", 0, 0)
        controller = struct.pack("<iH4fii", -1, 8, 1, 0, 0, 1, 0, 2)
        keys = struct.pack("<II10fII", 2, 1, 0, 1, 0, 0, 0, 1, 0.9238795, 0, 0, 0.3826834, 0, 0)
        animation = nif_file([("NiNode", node), ("NiKeyframeController", controller), ("NiKeyframeData", keys)])
        self.stream_blocks = [("NiSequenceStreamHelper", nif_string("testclip") + struct.pack("<ii", 1, 3)),
            ("NiTextKeyExtraData", struct.pack("<iII", 2, 0, 0)),
            ("NiStringExtraData", struct.pack("<iI", -1, 0) + nif_string("branch")),
            ("NiKeyframeController", struct.pack("<iH4fii", -1, 8, 2, 0, 10, 12, -1, 4)),
            ("NiKeyframeData", struct.pack("<II10fII", 2, 1, 10, 1, 0, 0, 0, 12, 0.9238795, 0, 0, 0.3826834, 0, 0))]
        (self.root / "stream.kf").write_bytes(nif_file(self.stream_blocks))
        self.archive = bsa_file([("textures\\test.dds", dds_file()), ("meshes\\test.nif", mesh),
                                 ("meshes\\animated.nif", animation)])
        (self.root / "test.bsa").write_bytes(self.archive)
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps({"version": 1, "name": "synthetic TES3 parsing",
            "game": "Morrowind", "encoding": "windows-1252", "data_root": str(self.root),
            "plugins": ["patch.esp"], "archives": ["test.bsa"]}))
        def check(kind, identity, expected, **fields):
            return dict(kind=kind, id=identity, expected=expected, **fields)
        condition = {"valid": True, "function": 74, "comparison": "0", "variable": "tr_global", "value": 427}
        self.plan = {"version": 1, "checks": [
            check("global", "TR_GLOBAL", {"type": "s", "value": 427, "source": "base.esm"}),
            check("global", "bad_short", {"value": 0}), check("global", "long_value", {"value": -123}),
            check("actor", "tr_actor", {"level": 8, "rank": 3, "gender": 1, "disposition": 65,
                "reputation": 8, "health": 75, "attributes": {"strength": 55}, "skills": {"block": 42},
                "race": "dark elf", "class": "priest", "faction": "temple", "inventory": {"tr_robe": 2}}),
            check("faction", "temple", {"attributes": [0, 1], "reactions": {"blades": -10},
                "ranks": [dict(attribute1=30, attribute2=35, primary_skill=40, favoured_skill=30, reputation=10)] * 10}),
            check("item", "tr_robe", {"worn_value": 200}, type="CLOT"),
            check("script", "tr_script", {"shorts": 1, "longs": 0, "floats": 0,
                "variables": ["state"], "bytecode_bytes": 3, "source_bytes": len(script_source)}),
            check("spell", "tr_spell", {"type": 2, "cost": 7, "effects": [dict(id=79, attribute=0,
                skill=-1, range=0, area=0, duration=30, min=2, max=4)]}),
            check("dialogue", "tr_topic", {"type": 0, "infos": [
                dict(id="a", index=70, source="patch.esp", conditions=[condition]),
                dict(id="b", index=20, source="base.esm", conditions=[condition])]}),
            check("reference", "actor-placement", {"base": "tr_actor", "cell": "room", "interior": True,
                "position": [4, 5, 6], "source": "patch.esp", "condition": 37}, plugin="base.esm", frmr=42),
            check("record", "removed", {"exists": False}, type="STAT"),
            check("archive", "test.bsa", {"parsed": True, "files": 3}),
            check("asset", "textures/test.dds", {"parsed": True, "archive": "test.bsa",
                "width": 4, "height": 4, "mips": 1, "decoded_bytes": 8}, type="dds"),
            check("asset", "meshes/test.nif", {"parsed": True, "version": 0x04000002,
                "shapes": 1, "vertices": 3, "triangles": 1}, type="nif"),
            check("asset", "meshes/animated.nif", {"parsed": True,
                "clips": [dict(name="branch", duration=1, tracks=1, unsupported_interpolators=0)]}, type="animation"),
            check("asset", "stream.kf", {"parsed": True,
                "clips": [dict(name="testclip", duration=1, tracks=1, channels=[dict(node="branch",
                    rotation_keys=2, rotation_times=[0, 1], rotation_first=[0, 0, 0, 1],
                    rotation_last=[0, 0, struct.unpack('<f', struct.pack('<f', 0.3826834))[0],
                        struct.unpack('<f', struct.pack('<f', 0.9238795))[0]])])]}, type="animation")]}

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

    def run_probe(self, plan=None):
        expectations = self.root / "expectations.json"
        expectations.write_text(json.dumps(self.plan if plan is None else plan))
        result = subprocess.run([PROBE, "--tes3-recordcheck", str(self.profile), str(expectations)],
                                text=True, capture_output=True, timeout=30)
        return result, json.loads(result.stdout)

    def test_typed_values_overrides_and_determinism(self):
        result, report = self.run_probe()
        self.assertEqual(result.returncode, 0, report)
        self.assertTrue(report["ok"])
        self.assertEqual(report, self.run_probe()[1])

    def test_each_wrong_expectation_fails_with_field_path(self):
        for index in range(len(self.plan["checks"])):
            with self.subTest(index=index):
                plan = copy.deepcopy(self.plan)
                expected = plan["checks"][index]["expected"]
                field = next(iter(expected))
                expected[field] = "incorrect"
                result, report = self.run_probe(plan)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(report["checks"][index]["passed"])
                self.assertTrue(report["checks"][index]["mismatch_path"].startswith("/" + field))

    def test_invalid_contract_fails(self):
        for plan in ({"version": 2, "checks": []}, {"version": 1, "checks": []},
                     {"version": 1, "checks": [{"kind": "unknown", "id": "x", "expected": {"exists": True}}]},
                     {"version": 1, "checks": [{"kind": "actor", "id": "x", "expected": {}}]}):
            result, report = self.run_probe(plan)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error", report)

    def test_truncated_record_and_subrecord_fail(self):
        original = (self.root / "patch.esp").read_bytes()
        for malformed in (b"STAT" + struct.pack("<III", 12, 0, 0) + b"x",
                          record("STAT", b"NAME" + struct.pack("<I", 100) + b"x")):
            (self.root / "patch.esp").write_bytes(original + malformed)
            result, report = self.run_probe()
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(report["ok"])

    def test_corrupt_archive_and_assets_fail(self):
        unterminated = bytearray(self.archive)
        unterminated[12 + struct.unpack_from('<I', self.archive, 4)[0] - 1] = ord('x')
        for malformed in (self.archive[:-1], self.archive[:20],
                          self.archive[:12] + struct.pack("<II", 0xffffffff, 0) + self.archive[20:], unterminated):
            (self.root / "test.bsa").write_bytes(malformed)
            result, report = self.run_probe()
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(report["checks"][11]["passed"])
        (self.root / "test.bsa").write_bytes(self.archive)
        for kind, name in (("dds", "bad.dds"), ("nif", "bad.nif"), ("animation", "bad-animation.nif")):
            (self.root / name).write_bytes(b"not a valid artifact")
            result, report = self.run_probe({"version": 1, "checks": [dict(kind="asset", id=name,
                type=kind, expected={"exists": True})]})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(report["checks"][0]["passed"])

    def test_animation_stream_rejects_cycles_bad_links_and_timing(self):
        for controller in (struct.pack("<iH4fii", 3, 8, 2, 0, 10, 12, -1, 4),
                           struct.pack("<iH4fii", -1, 8, 2, 0, 10, 12, -1, 99),
                           struct.pack("<iH4fii", -1, 8, 0, 0, 10, 12, -1, 4)):
            blocks = list(self.stream_blocks)
            blocks[3] = ("NiKeyframeController", controller)
            (self.root / "stream.kf").write_bytes(nif_file(blocks))
            result, report = self.run_probe({"version": 1, "checks": [self.plan["checks"][-1]]})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(report["checks"][0]["passed"])

    def test_loose_asset_override_uses_profile_precedence(self):
        (self.root / 'textures').mkdir()
        (self.root / 'textures/test.dds').write_bytes(dds_file(8, 4))
        plan = {"version": 1, "checks": [dict(kind='asset', id='textures/test.dds', type='dds',
            expected=dict(parsed=True, archive='', width=8, height=4, decoded_bytes=16))]}
        result, report = self.run_probe(plan)
        self.assertEqual(result.returncode, 0, report)

    def test_nonfinite_float_global_does_not_pass_as_json_null(self):
        with (self.root / 'base.esm').open('ab') as plugin:
            plugin.write(global_record('nonfinite', 'f', float('nan')))
        result, report = self.run_probe({"version": 1, "checks": [dict(kind='global', id='nonfinite',
            expected=dict(exists=True, value=None))]})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('non-finite', report['checks'][0]['diagnostic'])


if __name__ == "__main__":
    unittest.main()
