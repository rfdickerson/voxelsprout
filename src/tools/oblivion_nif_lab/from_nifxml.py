#!/usr/bin/env python3
"""Resolve nif.xml field lists for Oblivion (20.0.0.4, BSVER 11).

Searching for the Havok block sizes was the wrong instinct: nif.xml is the
authoritative description of the format, and the only reason it was not the
starting point is that it is 565 KB of version-conditioned XML. Resolving it for
one specific version is far more reliable than inferring lengths from data --
and unlike a search, it explains WHY each field is or is not present.

Prints, per type, the fields that exist at this version and their fixed byte
sizes, marking the array/variable ones a reader has to handle by hand.

    curl -sSLo nif.xml https://raw.githubusercontent.com/niftools/nifxml/master/nif.xml
    python3 from_nifxml.py nif.xml bhkRigidBody bhkMoppBvTreeShape ...
"""
import re
import sys
import xml.etree.ElementTree as ET

# Oblivion.
VERSION = (20, 0, 0, 4)
BSVER = 11

FIXED = {
    "byte": 1, "char": 1, "bool": 1, "hkResponseType": 1, "MotionSystem": 1,
    "DeactivatorType": 1, "SolverDeactivation": 1, "MotionQuality": 1,
    "OblivionLayer": 1, "SkyrimLayer": 1,
    "short": 2, "ushort": 2, "hkbool": 2, "hkPackedVector3": 6,
    "int": 4, "uint": 4, "float": 4, "Ref": 4, "Ptr": 4, "StringOffset": 4,
    "FileVersion": 4, "hkHalf": 2, "HavokMaterial": 4, "HavokFilter": 4,
    "HavokColFilter": 4, "OblivionHavokMaterial": 4, "hkResponse": 4,
    "BroadPhaseType": 1,
    "Vector3": 12, "Vector4": 16, "hkQuaternion": 16, "Quaternion": 16,
    "hkMatrix3": 48, "Matrix33": 36, "Matrix44": 64, "hkTriangle": 16,
    "Color4": 16, "Color3": 12, "InertiaMatrix": 48, "QuaternionXYZW": 16,
}


def parse_version(text):
    parts = [int(p) for p in text.strip().split(".")]
    while len(parts) < 4:
        parts.append(0)
    return tuple(parts[:4])


def version_ok(node):
    since = node.get("since")
    until = node.get("until")
    if since and VERSION < parse_version(since):
        return False
    if until and VERSION > parse_version(until):
        return False
    cond = node.get("vercond") or ""
    if cond:
        # Only the handful of tokens that actually gate Havok fields. Anything
        # unrecognised is reported rather than silently assumed -- a wrongly
        # included field is exactly the kind of error this script exists to avoid.
        truth = {
            "#NI_BS_LTE_FO3#": True, "#BS_GTE_SKY#": False, "#BS_FO4#": False,
            "#BSVER# #LT# 76": True, "#BSVER# #GTE# 76": False,
            "#NI_BS_GTE_SKY#": False, "#BSVER# #LT# 16": True,
            "#BSVER# #GTE# 83": False, "#BSVER# #LTE# 34": True,
        }
        for token, value in truth.items():
            if cond.strip() == token or cond.strip() == "(" + token + ")":
                return value
        if cond.strip() == "#BS_GTE_SKY# #AND# (!#BS_FO4#)":
            return False
        return None  # unknown condition
    return True


def main():
    path = sys.argv[1]
    wanted = sys.argv[2:]
    root = ET.parse(path).getroot()
    by_name = {}
    for node in root:
        name = node.get("name")
        if name:
            by_name[name] = node

    def fields(type_name, depth=0):
        node = by_name.get(type_name)
        if node is None:
            return [("  " * depth + f"?? unknown type {type_name}", None)]
        out = []
        parent = node.get("inherit")
        if parent:
            out.extend(fields(parent, depth))
        for field in node.findall("field"):
            ok = version_ok(field)
            fname = field.get("name")
            ftype = field.get("type")
            length = field.get("length")
            if ok is False:
                continue
            label = "  " * depth + f"{fname} : {ftype}"
            if ok is None:
                out.append((label + f"   [UNRESOLVED cond {field.get('vercond')}]", None))
                continue
            if length:
                if ftype == "byte" and length.isdigit():
                    out.append((label + f"[{length}]", int(length)))
                else:
                    out.append((label + f"[{length}]  ARRAY", None))
                continue
            if ftype in FIXED:
                out.append((label, FIXED[ftype]))
            elif ftype in by_name and by_name[ftype].tag in ("enum", "bitflags", "bitfield"):
                storage = by_name[ftype].get("storage", "uint")
                out.append((label + f"  <{by_name[ftype].tag} {storage}>",
                            FIXED.get(storage)))
            elif ftype in by_name:
                out.append((label + "  (expand)", None))
                out.extend(fields(ftype, depth + 1))
            else:
                out.append((label + "  ?? size unknown", None))
        return out

    for name in wanted:
        print(f"===== {name} @ 20.0.0.4 bsver 11 =====")
        total = 0
        exact = True
        for label, size in fields(name):
            if size is None:
                print(f"   {label}")
                if "(expand)" not in label:
                    exact = False
            else:
                print(f"   {label:<58} {size:4d}")
                total += size
        print(f"   ---- fixed part: {total} bytes"
              f"{'  (EXACT: whole block is fixed size)' if exact else '  + variable'}\n")


if __name__ == "__main__":
    main()
