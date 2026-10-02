#!/usr/bin/env python3
"""Check the Linux allocation interposer counts calls across threads."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


def child():
    lib = ctypes.CDLL(None)
    lib.odai_bench_frame_begin.argtypes = [ctypes.c_uint64]
    lib.odai_bench_frame_end.argtypes = [ctypes.c_uint64]
    lib.malloc.argtypes = [ctypes.c_size_t]
    lib.malloc.restype = ctypes.c_void_p
    lib.free.argtypes = [ctypes.c_void_p]
    lib.odai_bench_frame_begin(17)
    def allocate():
        for _ in range(100):
            lib.free(lib.malloc(32))
    thread = threading.Thread(target=allocate)
    thread.start()
    thread.join()
    lib.odai_bench_frame_end(17)
    lines = Path(os.environ['ODAI_HEAP_ALLOC_CSV']).read_text().splitlines()
    assert lines[0] == 'frame,allocations'
    frame, count = lines[1].split(',')
    assert frame == '17' and int(count) >= 100, lines


if __name__ == '__main__':
    if len(sys.argv) == 1:
        child()
    else:
        with tempfile.TemporaryDirectory() as folder:
            env = dict(os.environ, LD_PRELOAD=sys.argv[1],
                       ODAI_HEAP_ALLOC_CSV=str(Path(folder) / 'alloc.csv'))
            subprocess.run([sys.executable, __file__], env=env, check=True)
