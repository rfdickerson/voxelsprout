"""Sequential NIF 20.0.0.4 block walker, for deriving Oblivion's block layouts.

20.0.0.4 has no block-size table, so a reader has to consume every block type
exactly or it desyncs from the first unknown block onward. This is the harness
for working out those layouts against retail data: each block type gets a reader
that must consume exactly its own bytes, and the walk self-checks.

Two oracles:
  * per-block -- after each block the next one must start with a plausible
    SizedString length (most blocks derive from NiObjectNET, whose first field
    is the name), which localizes a desync to the block that caused it.
  * whole-file -- after numBlocks blocks the remainder must be exactly the
    footer, 4 + 4*numRoots bytes.
"""
import struct


class Cur:
    def __init__(self, data, pos=0):
        self.d = data
        self.p = pos

    def u8(self):
        v = self.d[self.p]
        self.p += 1
        return v

    def u16(self):
        v = struct.unpack_from("<H", self.d, self.p)[0]
        self.p += 2
        return v

    def i32(self):
        v = struct.unpack_from("<i", self.d, self.p)[0]
        self.p += 4
        return v

    def u32(self):
        v = struct.unpack_from("<I", self.d, self.p)[0]
        self.p += 4
        return v

    def f32(self):
        v = struct.unpack_from("<f", self.d, self.p)[0]
        self.p += 4
        return v

    def skip(self, n):
        if n < 0:
            raise ValueError("negative skip")
        self.p += n
        if self.p > len(self.d):
            raise ValueError("overrun")

    def sstr(self):
        n = self.u32()
        if n > 4096:
            raise ValueError(f"implausible string length {n}")
        s = self.d[self.p:self.p + n]
        self.p += n
        return s.decode("latin-1")

    def line(self):
        e = self.d.index(b"\n", self.p)
        s = self.d[self.p:e].decode("latin-1")
        self.p = e + 1
        return s


class Header:
    pass


def parse_header(c):
    h = Header()
    h.magic = c.line()
    h.version = c.u32()
    h.endian = c.u8()
    h.user = c.u32()
    h.num_blocks = c.u32()
    h.user2 = c.u32()
    for _ in range(3):
        n = c.u8()
        c.skip(n)
    n_types = c.u16()
    h.types = [c.sstr() for _ in range(n_types)]
    h.type_index = [c.u16() for _ in range(h.num_blocks)]
    h.num_groups = c.u32()
    c.skip(4 * h.num_groups)
    return h


# --- shared bases -----------------------------------------------------------

def ni_object_net(c):
    name = c.sstr()
    n = c.u32()
    c.skip(4 * n)          # extra data refs
    c.i32()                # controller ref
    return name


def ni_av_object(c):
    name = ni_object_net(c)
    c.u16()                # flags (u16: userVersion2 <= 26)
    c.skip(4 * 3)          # translation
    c.skip(4 * 9)          # rotation
    c.f32()                # scale
    n = c.u32()
    props = [c.i32() for _ in range(n)]
    c.i32()                # collision object ref
    return name, props


def ni_geometry_data(c):
    c.i32()                # groupId
    nv = c.u16()
    c.u8(); c.u8()         # keep / compress flags
    if c.u8():
        c.skip(12 * nv)
    vflags = c.u16()       # stock Gamebryo NiVectorFlags: bits 0-5 = UV count
    if c.u8():
        c.skip(12 * nv)
        if vflags & 0x1000:
            c.skip(24 * nv)
    c.skip(12 + 4)         # bounding sphere
    if c.u8():
        c.skip(16 * nv)    # vertex colours
    c.skip((vflags & 0x3F) * 8 * nv)
    c.u16()                # consistency flags
    c.i32()                # additional data ref
    return nv


# --- block readers ----------------------------------------------------------
# Each consumes exactly one block and returns anything the caller wants.

def r_ninode(c, h):
    ni_av_object(c)
    n = c.u32(); c.skip(4 * n)      # children
    n = c.u32(); c.skip(4 * n)      # effects


def r_trishape(c, h):
    ni_av_object(c)
    c.i32()                          # data
    c.i32()                          # skin instance
    if c.u8():                       # has shader
        c.sstr()
        c.i32()


def r_trishapedata(c, h):
    nt = ni_geometry_data(c)
    ntri = c.u16()
    c.u32()                          # numTrianglePoints
    if c.u8():
        c.skip(6 * ntri)
    ng = c.u16()
    for _ in range(ng):
        n = c.u16()
        c.skip(2 * n)


def r_tristripsdata(c, h):
    ni_geometry_data(c)
    c.u16()                          # numTriangles
    ns = c.u16()
    lens = [c.u16() for _ in range(ns)]
    if c.u8():
        c.skip(2 * sum(lens))


def r_niproperty_only(c, h):
    ni_object_net(c)


def r_alpha(c, h):
    ni_object_net(c)
    c.u16()                          # flags
    c.u8()                           # threshold


def r_material(c, h):
    ni_object_net(c)
    c.skip(12 * 4)                   # ambient, diffuse, specular, emissive
    c.f32()                          # glossiness
    c.f32()                          # alpha


def r_stencil(c, h):
    ni_object_net(c)
    c.u8()                           # stencil enabled
    c.u32(); c.u32()                 # stencil function, ref
    c.u32(); c.u32(); c.u32(); c.u32()   # mask, fail, zfail, pass
    c.u32()                          # draw mode


def r_specular(c, h):
    ni_object_net(c)
    c.u16()


def r_vertexcolor(c, h):
    ni_object_net(c)
    c.u16()                          # flags
    c.u32(); c.u32()                 # vertex mode, lighting mode


def tex_desc(c):
    c.i32()                          # source ref
    c.u32()                          # clamp mode
    c.u32()                          # filter mode
    c.u32()                          # uv set
    if c.u8():                       # has texture transform
        c.skip(2 * 4 + 2 * 4 + 4)    # translation, tiling, w rotation
        c.u32()                      # transform type
        c.skip(2 * 4)                # centre offset


def r_texturing(c, h):
    # Resolved from nif.xml at 20.0.0.4. Two things a count-driven loop gets
    # wrong: the bump-map slot carries three extra fields after its TexDesc, and
    # the block ends with a shader-texture array that exists from 10.0.1.0 --
    # omitting it under-consumes every NiTexturingProperty in the archive and
    # desyncs the NiSourceTexture that follows.
    ni_object_net(c)
    # `Flags` is until=10.0.1.2 and so is absent here.
    c.u32()                          # apply mode (until 20.1.0.1)
    count = c.u32()                  # texture count

    def slot(present):
        if present and c.u8():
            tex_desc(c)
            return True
        return False

    slot(True)                       # base
    slot(True)                       # dark
    slot(True)                       # detail
    slot(True)                       # gloss
    slot(True)                       # glow
    if count > 5 and c.u8():         # bump map
        tex_desc(c)
        c.f32(); c.f32()             # luma scale, luma offset
        c.skip(16)                   # Matrix22
    if count > 6:
        slot(True)                   # decal 0
    if count > 7:
        slot(True)                   # decal 1
    if count > 8:
        slot(True)                   # decal 2
    if count > 9:
        slot(True)                   # decal 3
    n = c.u32()                      # num shader textures (since 10.0.1.0)
    for _ in range(n):
        if c.u8():                   # ShaderTexDesc: has map
            tex_desc(c)
            c.u32()                  # map index


def r_sourcetexture(c, h):
    ni_object_net(c)
    if c.u8():                       # use external
        c.sstr()                     # file name
        c.i32()                      # unknown link (>= 10.1.0.0)
    else:
        c.sstr()                     # internal file name (>= 10.1.0.0)
        c.i32()                      # pixel data ref
    c.u32()                          # pixel layout
    c.u32()                          # use mipmaps
    c.u32()                          # alpha format
    c.u8()                           # is static
    c.u8()                           # direct render
    # persistRenderData arrived at 20.2.0.7 and is NOT present here.


def r_stringextradata(c, h):
    c.sstr()                         # NiExtraData: name only
    c.sstr()


def r_bsxflags(c, h):
    c.sstr()
    c.u32()


def r_bsbound(c, h):
    c.sstr()
    c.skip(12 + 12)                  # centre, dimensions


def r_binaryextradata(c, h):
    # NiExtraData derives from NiObject: name only, no extra-data list and no
    # controller ref. Getting this wrong desyncs immediately.
    c.sstr()
    n = c.u32()
    c.skip(n)


def r_integerextradata(c, h):
    c.sstr()
    c.u32()


READERS = {
    "NiNode": r_ninode,
    "NiTriShape": r_trishape,
    "NiTriStrips": r_trishape,
    "NiTriShapeData": r_trishapedata,
    "NiTriStripsData": r_tristripsdata,
    "NiAlphaProperty": r_alpha,
    "NiMaterialProperty": r_material,
    "NiStencilProperty": r_stencil,
    "NiSpecularProperty": r_specular,
    "NiVertexColorProperty": r_vertexcolor,
    "NiTexturingProperty": r_texturing,
    "NiSourceTexture": r_sourcetexture,
    "NiStringExtraData": r_stringextradata,
    "NiBinaryExtraData": r_binaryextradata,
    "NiIntegerExtraData": r_integerextradata,
    "BSXFlags": r_bsxflags,
    "BSBound": r_bsbound,
}


# --- Havok blocks -----------------------------------------------------------
#
# Pure SKIPPERS: nothing downstream wants collision, it only has to be consumed
# exactly so the next block starts where it should.
#
# Every layout below is RESOLVED FROM nif.xml for version 20.0.0.4 / BSVER 11 by
# from_nifxml.py -- not guessed, and not inferred from data. Guessing was tried
# first and each guess was wrong in a different specific way: bhkMoppBvTreeShape
# has no Material field at all, bhkNiTriStripsShape has a Radius and a 20-byte
# unused run, and bhkRigidBody's body is 236 bytes rather than the ~160 a
# plausible reading of the field list suggests. Searching for those numbers
# against the data found weak local optima that walked a handful of files;
# the spec gives them outright, with the reason each field is or is not present.


def r_bhk_collision_object(c, h):
    c.i32()                          # target Ptr
    c.u16()                          # bhkCOFlags
    c.i32()                          # body Ref


def r_bhk_rigid_body(c, h):
    # bhkWorldObject + bhkEntity + bhkRigidBodyCInfo550_660 = 228 fixed bytes
    # before the constraint list. bhkRigidBodyT is byte-identical -- the T only
    # changes whether translation/rotation are HONOURED, not whether they are
    # stored.
    c.skip(228)
    n = c.u32()                      # numConstraints
    c.skip(4 * n)
    c.u32()                          # body flags


def r_bhk_mopp_bv_tree_shape(c, h):
    c.i32()                          # shape Ref
    c.skip(12)                       # unused
    c.f32()                          # scale
    size = c.u32()                   # MOPP data size
    c.skip(16)                       # offset Vector4
    # hkMoppCode's Build Type is gated on #BS_GT_FO3# and so is absent here.
    c.skip(size)


def r_bhk_nitristrips_shape(c, h):
    c.u32(); c.f32()                 # material, radius
    c.skip(20)                       # unused
    c.u32()                          # grow by
    c.skip(16)                       # scale Vector4
    n = c.u32(); c.skip(4 * n)       # strips data refs
    m = c.u32(); c.skip(4 * m)       # per-strip havok filters


def r_bhk_convex_vertices_shape(c, h):
    c.u32(); c.f32()                 # material, radius
    c.skip(12 + 12)                  # vertices + normals bhkWorldObjCInfoProperty
    nv = c.u32(); c.skip(16 * nv)
    nn = c.u32(); c.skip(16 * nn)


def r_bhk_box_shape(c, h):
    c.u32(); c.f32()                 # material, radius
    c.skip(8 + 12 + 4)               # unused, dimensions, unused float


def r_bhk_capsule_shape(c, h):
    c.u32(); c.f32()
    c.skip(8 + 12 + 4 + 12 + 4)


def r_bhk_sphere_shape(c, h):
    c.u32(); c.f32()


def r_bhk_transform_shape(c, h):
    c.i32()                          # shape Ref
    c.u32(); c.f32()                 # material, radius
    c.skip(8 + 64)                   # unused, transform Matrix44


def r_bhk_list_shape(c, h):
    n = c.u32(); c.skip(4 * n)       # sub-shape refs
    c.u32()                          # material
    c.skip(12 + 12)                  # child shape + child filter properties
    m = c.u32(); c.skip(4 * m)       # filters


HAVOK_READERS = {
    "bhkCollisionObject": r_bhk_collision_object,
    "bhkRigidBody": r_bhk_rigid_body,
    "bhkRigidBodyT": r_bhk_rigid_body,
    "bhkMoppBvTreeShape": r_bhk_mopp_bv_tree_shape,
    "bhkNiTriStripsShape": r_bhk_nitristrips_shape,
    "bhkConvexVerticesShape": r_bhk_convex_vertices_shape,
    "bhkBoxShape": r_bhk_box_shape,
    "bhkCapsuleShape": r_bhk_capsule_shape,
    "bhkSphereShape": r_bhk_sphere_shape,
    "bhkConvexTransformShape": r_bhk_transform_shape,
    "bhkTransformShape": r_bhk_transform_shape,
    "bhkListShape": r_bhk_list_shape,
}
READERS.update(HAVOK_READERS)


# Types whose first field is NOT a SizedString name, so the plausibility check
# below cannot be applied to them. Getting this wrong does not produce a wrong
# parse -- it produces a FALSE ALARM, aborting a walk that was correct. That is
# how the Havok chain looked broken long after its layouts were right: every
# bhk* block starts with a Ref, and NiGeometryData starts with Group ID.
NOT_NAME_FIRST = {
    "NiTriShapeData", "NiTriStripsData", "NiSkinData", "NiSkinPartition",
    "NiSkinInstance", "NiAdditionalGeometryData",
}


def starts_with_name(type_name):
    return not type_name.startswith("bhk") and type_name not in NOT_NAME_FIRST


def r_zbuffer(c, h):
    ni_object_net(c)
    c.u16()                          # flags
    c.u32()                          # z compare function


def r_textkey_extradata(c, h):
    c.sstr()                         # NiExtraData name
    n = c.u32()
    for _ in range(n):               # Key<string>: time + value
        c.f32()
        c.sstr()


def r_billboard_node(c, h):
    r_ninode(c, h)
    c.u16()                          # billboard mode


def r_furniture_marker(c, h):
    c.sstr()                         # NiExtraData name
    n = c.u32()
    c.skip(16 * n)                   # FurniturePosition: offset, orientation, 2 refs


READERS.update({
    "BSFurnitureMarker": r_furniture_marker,
    "NiZBufferProperty": r_zbuffer,
    "NiTextKeyExtraData": r_textkey_extradata,
    "NiBillboardNode": r_billboard_node,
})


def plausible_block_start(data, p):
    """Most blocks open with a SizedString name; a desync almost never does.

    Only meaningful for types in starts_with_name() -- callers must check.
    """
    if p + 4 > len(data):
        return False
    n = struct.unpack_from("<I", data, p)[0]
    if n == 0:
        return True
    if n > 512 or p + 4 + n > len(data):
        return False
    return all(32 <= b < 127 for b in data[p + 4:p + 4 + n])


def walk(data):
    """-> (ok, header, first_bad_type, blocks_done)."""
    c = Cur(data)
    h = parse_header(c)
    for i in range(h.num_blocks):
        t = h.types[h.type_index[i]]
        fn = READERS.get(t)
        if fn is None:
            return False, h, t, i
        start = c.p
        try:
            fn(c, h)
        except Exception:
            return False, h, t + " (raised)", i
        if c.p > len(data):
            return False, h, t + " (overrun)", i
        if (i + 1 < h.num_blocks and
                starts_with_name(h.types[h.type_index[i + 1]]) and
                not plausible_block_start(data, c.p)):
            return False, h, t + " (desync)", i
    # Footer: numRoots + roots, and nothing else.
    if c.p + 4 > len(data):
        return False, h, "(no footer)", h.num_blocks
    n_roots = struct.unpack_from("<I", data, c.p)[0]
    if c.p + 4 + 4 * n_roots != len(data):
        return False, h, "(footer mismatch)", h.num_blocks
    return True, h, None, h.num_blocks
