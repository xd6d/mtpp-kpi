"""
Memory-bound задача: транспонування матриці 10000x10000 (1e8 елементів int32 ≈ 400 MB).
Реалізації:
  - sequential   — повністю поелементне транспонування у Python (на меншому розмірі)
  - numpy        — еталонна швидка реалізація через .T.copy()
  - threads      — паралельне поблочне транспонування (numpy звільняє GIL → потоки прискорюють)
  - processes    — поблочне транспонування через ProcessPoolExecutor (передача shared-memory)

Memory-bound характер: вузьким місцем є пропускна здатність ОЗП, а не CPU.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import shared_memory

import numpy as np

N = 10_000
BLOCK_SIZE = 1_000          # 10x10 = 100 блоків
DTYPE = np.int32


def _make_matrix() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 1000, size=(N, N), dtype=DTYPE)


def transpose_numpy(a: np.ndarray) -> np.ndarray:
    return a.T.copy()


def _transpose_block(a: np.ndarray, out: np.ndarray, br: int, bc: int):
    rs, re = br * BLOCK_SIZE, (br + 1) * BLOCK_SIZE
    cs, ce = bc * BLOCK_SIZE, (bc + 1) * BLOCK_SIZE
    out[cs:ce, rs:re] = a[rs:re, cs:ce].T


def transpose_threads(a: np.ndarray, workers: int) -> np.ndarray:
    out = np.empty_like(a)
    blocks = [(br, bc) for br in range(N // BLOCK_SIZE) for bc in range(N // BLOCK_SIZE)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda b: _transpose_block(a, out, b[0], b[1]), blocks))
    return out


# Для процесів використовуємо shared memory, щоб не серіалізувати 400 МБ.
_SHARED_INFO: dict = {}


def _proc_init(name_in: str, name_out: str, shape: tuple[int, int], dtype_str: str):
    shm_in = shared_memory.SharedMemory(name=name_in)
    shm_out = shared_memory.SharedMemory(name=name_out)
    _SHARED_INFO["a"] = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm_in.buf)
    _SHARED_INFO["out"] = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm_out.buf)
    _SHARED_INFO["shm_in"] = shm_in
    _SHARED_INFO["shm_out"] = shm_out


def _proc_block(args):
    br, bc = args
    a = _SHARED_INFO["a"]
    out = _SHARED_INFO["out"]
    rs, re = br * BLOCK_SIZE, (br + 1) * BLOCK_SIZE
    cs, ce = bc * BLOCK_SIZE, (bc + 1) * BLOCK_SIZE
    out[cs:ce, rs:re] = a[rs:re, cs:ce].T


def transpose_processes(a: np.ndarray, workers: int) -> np.ndarray:
    shm_in = shared_memory.SharedMemory(create=True, size=a.nbytes)
    shm_out = shared_memory.SharedMemory(create=True, size=a.nbytes)
    try:
        shared_a = np.ndarray(a.shape, dtype=a.dtype, buffer=shm_in.buf)
        np.copyto(shared_a, a)
        blocks = [(br, bc) for br in range(N // BLOCK_SIZE) for bc in range(N // BLOCK_SIZE)]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_proc_init,
            initargs=(shm_in.name, shm_out.name, a.shape, a.dtype.str),
        ) as ex:
            list(ex.map(_proc_block, blocks))
        shared_out = np.ndarray(a.shape, dtype=a.dtype, buffer=shm_out.buf)
        return shared_out.copy()
    finally:
        shm_in.close(); shm_in.unlink()
        shm_out.close(); shm_out.unlink()


def run_all():
    a = _make_matrix()
    print(f"Matrix: {a.shape}, {a.nbytes / 1024 / 1024:.0f} MB, CPU={os.cpu_count()}")

    t0 = time.perf_counter()
    b = transpose_numpy(a)
    seq = time.perf_counter() - t0
    print(f"NumPy .T.copy() (1 thread baseline): {seq:.3f}s")
    assert b[0, 1] == a[1, 0]

    results = {"sequential": seq}
    for w in (2, 4, 8):
        t0 = time.perf_counter()
        b = transpose_threads(a, w)
        t = time.perf_counter() - t0
        assert b[0, 1] == a[1, 0]
        results[f"threads_{w}"] = t
        print(f"Threads w={w}: {t:.3f}s   speedup={seq / t:.2f}x")

    for w in (2, 4):
        t0 = time.perf_counter()
        b = transpose_processes(a, w)
        t = time.perf_counter() - t0
        assert b[0, 1] == a[1, 0]
        results[f"processes_{w}"] = t
        print(f"Process w={w}: {t:.3f}s   speedup={seq / t:.2f}x")

    return results


if __name__ == "__main__":
    run_all()
