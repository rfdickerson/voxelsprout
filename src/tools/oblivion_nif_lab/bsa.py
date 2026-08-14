"""Minimal BSA v103/v104 reader, for iterating on NIF layouts outside the C++ build.

Layout transcribed from src/import/fnv/bsa_archive.cc. The one v103 quirk it has
to reproduce is the embed-file-names flag: Oblivion sets 0x100 but writes no
embedded names, so honouring it eats the head of every payload.
"""
import struct
import zlib

FLAG_FOLDER_NAMES = 0x1
FLAG_FILE_NAMES = 0x2
FLAG_COMPRESSED = 0x4
FLAG_EMBED_NAMES = 0x100
FILE_COMPRESSION_TOGGLE = 0x40000000
FILE_SIZE_MASK = 0x3FFFFFFF


class Bsa:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as handle:
            self.data = handle.read()
        magic, version, folder_offset, flags, n_folders, n_files, \
            total_folder_name_len, total_file_name_len, file_flags = \
            struct.unpack_from("<4s8I", self.data, 0)
        if magic != b"BSA\0":
            raise ValueError(f"{path}: not a BSA")
        self.version = version
        if version == 103:
            flags &= ~FLAG_EMBED_NAMES
        self.flags = flags
        self.entries = {}

        pos = folder_offset
        folders = []
        for _ in range(n_folders):
            _hash, count, offset = struct.unpack_from("<QII", self.data, pos)
            pos += 16
            folders.append((count, offset - total_file_name_len))

        names = []
        # The file-name block sits after ALL the folder blocks (each folder's
        # name plus its file records), not after the folder record table.
        name_block_start = pos
        for count, offset in folders:
            p = offset
            folder_name = ""
            if flags & FLAG_FOLDER_NAMES:
                length = self.data[p]
                p += 1
                folder_name = self.data[p:p + length].split(b"\0")[0].decode("latin-1")
                p += length
            for _ in range(count):
                _h, size, off = struct.unpack_from("<QII", self.data, p)
                p += 16
                names.append((folder_name, size, off))
            name_block_start = max(name_block_start, p)

        # One NUL-terminated file name per entry, in the same order.
        p = name_block_start
        for folder_name, size, off in names:
            end = self.data.index(b"\0", p)
            file_name = self.data[p:end].decode("latin-1")
            p = end + 1
            key = (folder_name + "\\" + file_name).lower() if folder_name else file_name.lower()
            self.entries[key] = (size, off)

    def read(self, key):
        size, off = self.entries[key.lower().replace("/", "\\")]
        compressed = bool(self.flags & FLAG_COMPRESSED)
        if size & FILE_COMPRESSION_TOGGLE:
            compressed = not compressed
        size &= FILE_SIZE_MASK
        if self.flags & FLAG_EMBED_NAMES:
            length = self.data[off]
            off += 1 + length
            size -= 1 + length
        if compressed:
            original = struct.unpack_from("<I", self.data, off)[0]
            raw = zlib.decompress(self.data[off + 4:off + size])
            assert len(raw) == original, f"{key}: {len(raw)} != {original}"
            return raw
        return self.data[off:off + size]

    def names(self, suffix=None):
        for key in self.entries:
            if suffix is None or key.endswith(suffix):
                yield key
