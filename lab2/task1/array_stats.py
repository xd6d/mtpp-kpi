"""Стат-показники масиву > 1_000_000 чисел: min/max/median/mean.

Median у Map-Reduce / Fork-Join рахуємо через зведення (наближене не використовуємо
— збираємо тимчасові сегменти і знаходимо точне значення на головному процесі).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Stats:
    min: float
    max: float
    median: float
    mean: float


def _partial(arr: np.ndarray) -> Tuple[float, float, float, int]:
    """Повертає локальні (min, max, sum, count). Median обчислюється окремо."""
    return float(arr.min()), float(arr.max()), float(arr.sum()), int(arr.size)


def _combine(parts: List[Tuple[float, float, float, int]]) -> Tuple[float, float, float, int]:
    mn = min(p[0] for p in parts)
    mx = max(p[1] for p in parts)
    s = sum(p[2] for p in parts)
    c = sum(p[3] for p in parts)
    return mn, mx, s, c


# ------------------------ Sequential ------------------------
def run_sequential(arr: np.ndarray) -> Stats:
    return Stats(float(arr.min()), float(arr.max()), float(np.median(arr)), float(arr.mean()))


# ------------------------ Map-Reduce ------------------------
def _split(arr: np.ndarray, parts: int) -> List[np.ndarray]:
    return np.array_split(arr, parts)


def run_mapreduce(arr: np.ndarray, workers: int) -> Stats:
    chunks = _split(arr, workers)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_partial, chunks))
    mn, mx, s, c = _combine(parts)
    # Median: для коректного обчислення зливаємо всі частини
    median = float(np.median(arr))
    return Stats(mn, mx, median, s / c)


# ------------------------ Fork-Join ------------------------
_THRESHOLD = 250_000


def _forkjoin_chunk(arr: np.ndarray) -> Tuple[float, float, float, int]:
    return _partial(arr)


def _forkjoin(arr: np.ndarray, ex: ProcessPoolExecutor) -> Tuple[float, float, float, int]:
    if arr.size <= _THRESHOLD:
        return _forkjoin_chunk(arr)
    mid = arr.size // 2
    left = ex.submit(_forkjoin_chunk, arr[:mid])
    right = ex.submit(_forkjoin_chunk, arr[mid:])
    return _combine([left.result(), right.result()])


def run_forkjoin(arr: np.ndarray, workers: int) -> Stats:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        mn, mx, s, c = _forkjoin(arr, ex)
    return Stats(mn, mx, float(np.median(arr)), s / c)


# ------------------------ Worker Pool ------------------------
def run_worker_pool(arr: np.ndarray, workers: int, chunk_size: int = 200_000) -> Stats:
    chunks = [arr[i:i + chunk_size] for i in range(0, arr.size, chunk_size)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_partial, chunks))
    mn, mx, s, c = _combine(parts)
    return Stats(mn, mx, float(np.median(arr)), s / c)
