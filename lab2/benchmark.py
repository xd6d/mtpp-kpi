"""Запуск усіх сценаріїв та збір результатів у results.json + графіки."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from task1 import generate_data, html_tags, array_stats, matrix_mult
from task2 import generate_transactions, pipeline as t2pipeline

DATA = ROOT.parent / "data" if (ROOT.parent / "data").exists() else ROOT / "data"
RESULTS_JSON = ROOT / "results.json"


def timeit(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


def ensure_data():
    if not (DATA / "html_docs").exists():
        generate_data.generate_html()
    if not (DATA / "array.npy").exists():
        generate_data.generate_array()
    if not (DATA / "matrix_a.npy").exists():
        generate_data.generate_matrices()
    if not (DATA / "transactions.csv").exists():
        generate_transactions.generate()


def bench_task1_html(worker_list):
    files = html_tags.list_html(DATA / "html_docs")
    print(f"[task1.html] {len(files)} файлів")
    seq_t, seq_result = timeit(html_tags.run_sequential, files)
    rows = [{"pattern": "Sequential", "workers": 1, "time": seq_t}]
    for w in worker_list:
        for name, fn in (("Map-Reduce", html_tags.run_mapreduce),
                         ("Fork-Join", html_tags.run_forkjoin),
                         ("Worker Pool", html_tags.run_worker_pool)):
            t, _ = timeit(fn, files, w)
            rows.append({"pattern": name, "workers": w, "time": t})
            print(f"  {name:11s} w={w:2d}  {t:6.3f}s")
    return {"baseline": seq_t, "runs": rows, "top_tags": dict(seq_result.most_common(10))}


def bench_task1_array(worker_list):
    arr = np.load(DATA / "array.npy")
    print(f"[task1.array] {arr.size} чисел")
    seq_t, seq_result = timeit(array_stats.run_sequential, arr)
    rows = [{"pattern": "Sequential", "workers": 1, "time": seq_t}]
    for w in worker_list:
        for name, fn in (("Map-Reduce", array_stats.run_mapreduce),
                         ("Fork-Join", array_stats.run_forkjoin),
                         ("Worker Pool", array_stats.run_worker_pool)):
            t, _ = timeit(fn, arr, w)
            rows.append({"pattern": name, "workers": w, "time": t})
            print(f"  {name:11s} w={w:2d}  {t:6.3f}s")
    return {"baseline": seq_t, "runs": rows,
            "stats": {"min": seq_result.min, "max": seq_result.max,
                       "median": seq_result.median, "mean": seq_result.mean}}


def bench_task1_matrix(worker_list):
    A = np.load(DATA / "matrix_a.npy")
    B = np.load(DATA / "matrix_b.npy")
    print(f"[task1.matrix] A:{A.shape} B:{B.shape}")
    seq_t, _ = timeit(matrix_mult.run_sequential, A, B)
    rows = [{"pattern": "Sequential", "workers": 1, "time": seq_t}]
    for w in worker_list:
        for name, fn in (("Map-Reduce", matrix_mult.run_mapreduce),
                         ("Fork-Join", matrix_mult.run_forkjoin),
                         ("Worker Pool", matrix_mult.run_worker_pool)):
            t, _ = timeit(fn, A, B, w)
            rows.append({"pattern": name, "workers": w, "time": t})
            print(f"  {name:11s} w={w:2d}  {t:6.3f}s")
    return {"baseline": seq_t, "runs": rows}


def bench_task2(consumer_list):
    path = DATA / "transactions.csv"
    print(f"[task2] {path.name}")
    seq_t, agg = timeit(t2pipeline.run_sequential, path)
    rows = [{"pattern": "Sequential", "workers": 1, "time": seq_t}]
    pip_t, _ = timeit(t2pipeline.run_pipeline, path)
    rows.append({"pattern": "Pipeline", "workers": 3, "time": pip_t})
    print(f"  Pipeline(3 stages)  {pip_t:6.3f}s")
    for c in consumer_list:
        t, _ = timeit(t2pipeline.run_producer_consumer, path, c)
        rows.append({"pattern": "Producer-Consumer", "workers": c, "time": t})
        print(f"  Producer-Consumer  c={c:2d}  {t:6.3f}s")
    return {"baseline": seq_t, "runs": rows,
            "aggregate": {"total_usd": round(agg.total_usd, 2),
                           "refund_usd": round(agg.refund_usd, 2),
                           "count": agg.count,
                           "per_category": {k: round(v, 2) for k, v in agg.per_category.items()}}}


def main():
    ensure_data()
    worker_list = [2, 4, 8]
    out = {
        "task1_html": bench_task1_html(worker_list),
        "task1_array": bench_task1_array(worker_list),
        "task1_matrix": bench_task1_matrix(worker_list),
        "task2": bench_task2([2, 4, 8]),
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] зведені дані -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
