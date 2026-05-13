"""Pipeline (FX -> Refund -> Aggregate) та Producer-Consumer на тому самому
наборі транзакцій. Для порівняння є і Sequential baseline.

Бізнес-логіка:
 1) FX-конвертація суми у USD (за фіксованим курсом).
 2) Повернення коштів для користувачів з ID > 25000 — нараховуємо 20%.
 3) Агрегація: підсумок за категоріями {category: total_usd, refund_usd}.
"""
from __future__ import annotations

import csv
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Dict, Iterable, Iterator, List, Tuple

FX = {"USD": 1.0, "EUR": 1.08, "UAH": 0.025, "GBP": 1.27, "PLN": 0.25}
PREMIUM_THRESHOLD = 25_000
REFUND_RATE = 0.20

SENTINEL = None  # маркер кінця для черг


@dataclass
class Aggregate:
    total_usd: float = 0.0
    refund_usd: float = 0.0
    per_category: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    per_category_refund: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    count: int = 0


# ---------- етапи обробки (чисті функції) ----------
def fx_convert(row: dict) -> dict:
    rate = FX.get(row["currency"], 1.0)
    row["amount_usd"] = float(row["amount"]) * rate
    return row


def refund_stage(row: dict) -> dict:
    if int(row["user_id"]) > PREMIUM_THRESHOLD:
        row["refund_usd"] = row["amount_usd"] * REFUND_RATE
    else:
        row["refund_usd"] = 0.0
    return row


def update_aggregate(agg: Aggregate, row: dict) -> None:
    agg.total_usd += row["amount_usd"]
    agg.refund_usd += row["refund_usd"]
    agg.per_category[row["category"]] += row["amount_usd"]
    agg.per_category_refund[row["category"]] += row["refund_usd"]
    agg.count += 1


# ---------- читання вхідного CSV у міні-батчах ----------
def read_batches(path: Path, batch_size: int = 5_000) -> Iterator[List[dict]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch: List[dict] = []
        for row in reader:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


# =========================================================
# Sequential baseline
# =========================================================
def run_sequential(path: Path) -> Aggregate:
    agg = Aggregate()
    for batch in read_batches(path):
        for row in batch:
            row = fx_convert(row)
            row = refund_stage(row)
            update_aggregate(agg, row)
    return agg


# =========================================================
# Pipeline pattern: 3 етапи, кожен у власному потоці,
# зв'язок через bounded queues
# =========================================================
def _stage(in_q: Queue, out_q: Queue, fn) -> None:
    while True:
        batch = in_q.get()
        if batch is SENTINEL:
            out_q.put(SENTINEL)
            return
        out_q.put([fn(r) for r in batch])


def run_pipeline(path: Path, queue_size: int = 16) -> Aggregate:
    q_fx = Queue(maxsize=queue_size)
    q_refund = Queue(maxsize=queue_size)
    q_aggr = Queue(maxsize=queue_size)

    agg = Aggregate()

    def _producer():
        for batch in read_batches(path):
            q_fx.put(batch)
        q_fx.put(SENTINEL)

    def _aggregator():
        while True:
            batch = q_aggr.get()
            if batch is SENTINEL:
                return
            for r in batch:
                update_aggregate(agg, r)

    threads = [
        threading.Thread(target=_producer, name="producer"),
        threading.Thread(target=_stage, args=(q_fx, q_refund, fx_convert), name="fx"),
        threading.Thread(target=_stage, args=(q_refund, q_aggr, refund_stage), name="refund"),
        threading.Thread(target=_aggregator, name="aggregator"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return agg


# =========================================================
# Producer-Consumer pattern: один продюсер, N споживачів.
# Кожен споживач виконує ВСІ етапи на своєму батчі та
# повертає локальний Aggregate; головний потік зливає.
# =========================================================
def _process_batch(batch: List[dict]) -> Aggregate:
    local = Aggregate()
    for row in batch:
        row = fx_convert(row)
        row = refund_stage(row)
        update_aggregate(local, row)
    return local


def _merge(into: Aggregate, other: Aggregate) -> None:
    into.total_usd += other.total_usd
    into.refund_usd += other.refund_usd
    into.count += other.count
    for k, v in other.per_category.items():
        into.per_category[k] += v
    for k, v in other.per_category_refund.items():
        into.per_category_refund[k] += v


def run_producer_consumer(path: Path, consumers: int, queue_size: int = 32) -> Aggregate:
    q: Queue = Queue(maxsize=queue_size)
    out_q: Queue = Queue()

    def _producer():
        for batch in read_batches(path):
            q.put(batch)
        for _ in range(consumers):
            q.put(SENTINEL)

    def _consumer():
        local = Aggregate()
        while True:
            batch = q.get()
            if batch is SENTINEL:
                out_q.put(local)
                return
            _merge(local, _process_batch(batch))

    prod = threading.Thread(target=_producer, name="producer")
    cons = [threading.Thread(target=_consumer, name=f"c{i}") for i in range(consumers)]
    prod.start()
    for c in cons:
        c.start()
    prod.join()
    for c in cons:
        c.join()

    total = Aggregate()
    while not out_q.empty():
        _merge(total, out_q.get())
    return total
