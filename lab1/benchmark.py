"""
Запускає послідовний/потоковий/процесний варіанти для всіх трьох типів задач
і зберігає зведені результати у JSON для побудови графіків.
"""
import json
import os
import sys
import time

# Локальні модулі
sys.path.insert(0, os.path.dirname(__file__))
import cpu_bound
import memory_bound
import io_bound


THREAD_LEVELS = [1, 2, 4, 8]
PROC_LEVELS = [1, 2, 4, 8]
IO_THREAD_LEVELS = [1, 2, 4, 8, 16, 32, 64]


def bench_cpu():
    print("=== CPU-bound ===")
    out = {"sequential": cpu_bound.run_sequential()[0], "threads": {}, "processes": {}}
    for w in THREAD_LEVELS:
        t = cpu_bound.run_threads(w)[0] if w > 1 else out["sequential"]
        out["threads"][w] = t
        print(f"  threads  w={w}: {t:.3f}s")
    for w in PROC_LEVELS:
        t = cpu_bound.run_processes(w)[0] if w > 1 else out["sequential"]
        out["processes"][w] = t
        print(f"  process  w={w}: {t:.3f}s")
    return out


def bench_memory():
    print("=== Memory-bound ===")
    a = memory_bound._make_matrix()
    t0 = time.perf_counter()
    memory_bound.transpose_numpy(a)
    seq = time.perf_counter() - t0
    out = {"sequential": seq, "threads": {1: seq}, "processes": {1: seq}}
    print(f"  seq (.T.copy()): {seq:.3f}s")
    for w in (2, 4, 8):
        t0 = time.perf_counter()
        memory_bound.transpose_threads(a, w)
        t = time.perf_counter() - t0
        out["threads"][w] = t
        print(f"  threads  w={w}: {t:.3f}s")
    for w in (2, 4):
        t0 = time.perf_counter()
        memory_bound.transpose_processes(a, w)
        t = time.perf_counter() - t0
        out["processes"][w] = t
        print(f"  process  w={w}: {t:.3f}s")
    return out


def bench_io():
    print("=== I/O-bound ===")
    io_bound.generate_dataset()
    files = io_bound.list_files(io_bound.DATASET_ROOT)
    seq_t, _ = io_bound.run_sequential(files)
    out = {"sequential": seq_t, "threads": {1: seq_t}, "processes": {1: seq_t}}
    print(f"  sequential: {seq_t:.3f}s")
    for w in IO_THREAD_LEVELS[1:]:
        t, _ = io_bound.run_threads(files, w)
        out["threads"][w] = t
        print(f"  threads  w={w}: {t:.3f}s")
    for w in (2, 4, 8):
        t, _ = io_bound.run_processes(files, w)
        out["processes"][w] = t
        print(f"  process  w={w}: {t:.3f}s")
    return out


if __name__ == "__main__":
    results = {
        "cpu_count": os.cpu_count(),
        "cpu": bench_cpu(),
        "memory": bench_memory(),
        "io": bench_io(),
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "results.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
