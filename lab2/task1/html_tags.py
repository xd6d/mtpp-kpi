"""Підрахунок частоти HTML-тегів трьома патернами + sequential."""
from __future__ import annotations

import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Tuple

TAG_RE = re.compile(r"<\s*([a-zA-Z][a-zA-Z0-9]*)")


def count_tags_in_text(text: str) -> Counter:
    return Counter(m.group(1).lower() for m in TAG_RE.finditer(text))


def _read_file(path: str) -> Counter:
    with open(path, "r", encoding="utf-8") as f:
        return count_tags_in_text(f.read())


def _reduce(parts: Iterable[Counter]) -> Counter:
    total = Counter()
    for p in parts:
        total.update(p)
    return total


# ------------------------ Sequential ------------------------
def run_sequential(files: List[str]) -> Counter:
    return _reduce(_read_file(p) for p in files)


# ------------------------ Map-Reduce ------------------------
def _mapreduce_map(chunk: List[str]) -> Counter:
    """Map: обробити цілий чанк, повернути локальний Counter."""
    return _reduce(_read_file(p) for p in chunk)


def run_mapreduce(files: List[str], workers: int) -> Counter:
    chunks = [files[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        partials = list(ex.map(_mapreduce_map, chunks))
    return _reduce(partials)


# ------------------------ Fork-Join ------------------------
# Рекурсивно ділимо завдання навпіл, виконуємо в пулі і об'єднуємо.
_THRESHOLD = 64


def _forkjoin_recursive(files: List[str], executor: ProcessPoolExecutor) -> Counter:
    if len(files) <= _THRESHOLD:
        return _mapreduce_map(files)
    mid = len(files) // 2
    left = executor.submit(_forkjoin_recursive_chunk, files[:mid])
    right = executor.submit(_forkjoin_recursive_chunk, files[mid:])
    return _reduce([left.result(), right.result()])


def _forkjoin_recursive_chunk(files: List[str]) -> Counter:
    """Викликається у процесі-воркері. Просто прямий map-reduce у чанку."""
    return _mapreduce_map(files)


def run_forkjoin(files: List[str], workers: int) -> Counter:
    # Реалізація fork-join через рекурсивну decomposition + ProcessPoolExecutor.
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return _forkjoin_recursive(files, ex)


# ------------------------ Worker Pool ------------------------
def run_worker_pool(files: List[str], workers: int) -> Counter:
    """Worker pool: кожен файл — окрема задача, пул бере її з черги."""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        partials = list(ex.map(_read_file, files, chunksize=8))
    return _reduce(partials)


def list_html(html_dir: Path) -> List[str]:
    return sorted(str(p) for p in html_dir.glob("*.html"))
