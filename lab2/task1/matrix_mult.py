"""Множення двох великих матриць (1000x1000) у трьох патернах.

Стратегія: A @ B розбивається на рядкові блоки A. Для кожного блоку рядків
обчислюється A_block @ B → блок результату. NumPy всередині працює на BLAS і
звільняє GIL під час обчислень → потоки можуть давати реальне прискорення.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np


# ------------------------ Sequential ------------------------
def run_sequential(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


# ------------------------ Map-Reduce ------------------------
def _block_mul(args: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    a_rows, B = args
    return a_rows @ B


def _split_rows(A: np.ndarray, parts: int) -> List[np.ndarray]:
    return np.array_split(A, parts, axis=0)


def run_mapreduce(A: np.ndarray, B: np.ndarray, workers: int) -> np.ndarray:
    blocks = _split_rows(A, workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        partials = list(ex.map(_block_mul, ((blk, B) for blk in blocks)))
    return np.vstack(partials)


# ------------------------ Fork-Join ------------------------
_THRESHOLD = 64  # рядків


def _forkjoin(A: np.ndarray, B: np.ndarray, ex: ThreadPoolExecutor) -> np.ndarray:
    if A.shape[0] <= _THRESHOLD:
        return A @ B
    mid = A.shape[0] // 2
    left = ex.submit(_chunk_mul, A[:mid], B)
    right = ex.submit(_chunk_mul, A[mid:], B)
    return np.vstack([left.result(), right.result()])


def _chunk_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def run_forkjoin(A: np.ndarray, B: np.ndarray, workers: int) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return _forkjoin(A, B, ex)


# ------------------------ Worker Pool ------------------------
def run_worker_pool(A: np.ndarray, B: np.ndarray, workers: int, block: int = 64) -> np.ndarray:
    blocks = [A[i:i + block] for i in range(0, A.shape[0], block)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        partials = list(ex.map(_chunk_mul, blocks, [B] * len(blocks)))
    return np.vstack(partials)
