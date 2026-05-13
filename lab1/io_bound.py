"""
I/O-bound task: recursively traverse a directory and count total words
in all .txt files. Test dataset is generated randomly (~1000 small files).
A small per-file sleep (IO_LATENCY) simulates slow disk / network I/O —
without it, files cached in RAM make the work effectively CPU-bound.
"""
import os
import random
import shutil
import string
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

DATASET_ROOT = Path("/tmp/lab1_io_dataset")
NUM_FILES = 1000
FILE_KB = 20
SUBDIRS = 10
IO_LATENCY = 0.002      # 2 ms per file = simulated slow I/O


def generate_dataset():
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    DATASET_ROOT.mkdir(parents=True)
    rng = random.Random(7)
    words_pool = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9)))
                  for _ in range(2000)]
    total = 0
    for i in range(NUM_FILES):
        sub = DATASET_ROOT / f"dir_{i % SUBDIRS:02d}"
        sub.mkdir(exist_ok=True)
        path = sub / f"file_{i:04d}.txt"
        chunks, size = [], 0
        while size < FILE_KB * 1024:
            line = " ".join(rng.choices(words_pool, k=20)) + "\n"
            chunks.append(line)
            size += len(line)
        text = "".join(chunks)
        path.write_text(text, encoding="utf-8")
        total += len(text.split())
    return total


def count_words_in_file(path):
    if IO_LATENCY:
        time.sleep(IO_LATENCY)
    with open(path, encoding="utf-8") as f:
        return len(f.read().split())


def list_files(root):
    return [str(p) for p in root.rglob("*.txt")]


def run_sequential(files):
    t0 = time.perf_counter()
    total = sum(count_words_in_file(p) for p in files)
    return time.perf_counter() - t0, total


def run_threads(files, workers):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        total = sum(ex.map(count_words_in_file, files))
    return time.perf_counter() - t0, total


def run_processes(files, workers):
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        total = sum(ex.map(count_words_in_file, files, chunksize=16))
    return time.perf_counter() - t0, total


if __name__ == "__main__":
    print("Generating dataset...")
    expected = generate_dataset()
    files = list_files(DATASET_ROOT)
    print(f"Files: {len(files)}, expected words: {expected}, CPU={os.cpu_count()}")

    seq_t, total = run_sequential(files)
    print(f"Sequential: {seq_t:.3f}s, words={total}")

    for w in (2, 4, 8, 16, 32):
        t, total = run_threads(files, w)
        print(f"Threads w={w}: {t:.3f}s   speedup={seq_t / t:.2f}x")

    for w in (2, 4, 8):
        t, total = run_processes(files, w)
        print(f"Process w={w}: {t:.3f}s   speedup={seq_t / t:.2f}x")
