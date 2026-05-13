"""
Лабораторна робота №3 - Завдання 1.
Демонстрація проблем Race Condition та Deadlock на прикладі переказів
між банківськими рахунками (>100 рахунків, >1000 потоків).

Режими (CLI --mode):
  unsafe          - без синхронізації => Race Condition (сума не зберігається).
  safe            - мьютекс на кожен рахунок, впорядковане захоплення => коректно.
  deadlock        - мьютекси захоплюються (from, to) => Deadlock.
  deadlock-fixed  - мьютекси (min, max) id => Deadlock уникнено.
  bench           - послідовно прогоняє всі режими та зберігає JSON.

Приклади:
  python bank_accounts.py --mode bench
  python bank_accounts.py --mode unsafe --accounts 150 --threads 1200 --ops 100
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Account:
    id: int
    balance: int
    lock: threading.Lock


def make_accounts(n: int, smin: int = 100, smax: int = 10_000) -> list[Account]:
    return [
        Account(id=i, balance=random.randint(smin, smax), lock=threading.Lock())
        for i in range(n)
    ]


def total_money(accounts: list[Account]) -> int:
    return sum(a.balance for a in accounts)


# UNSAFE: явний read-modify-write з yield => Race Condition
def transfer_unsafe(src: Account, dst: Account, amount: int) -> None:
    if src.balance >= amount:
        src_old = src.balance
        dst_old = dst.balance
        time.sleep(0)  # явна точка перемикання GIL
        src.balance = src_old - amount
        dst.balance = dst_old + amount


# SAFE: лок на меншому id першим => без deadlock та без race
def transfer_safe(src: Account, dst: Account, amount: int) -> None:
    first, second = (src, dst) if src.id < dst.id else (dst, src)
    with first.lock, second.lock:
        if src.balance >= amount:
            src.balance -= amount
            dst.balance += amount


# DEADLOCK: захоплюємо src->dst, гарантовано блокується на зустрічних переказах
def transfer_deadlock(src: Account, dst: Account, amount: int) -> None:
    with src.lock:
        time.sleep(0.0005)
        with dst.lock:
            if src.balance >= amount:
                src.balance -= amount
                dst.balance += amount


# DEADLOCK-FIXED: впорядковане захоплення локів
def transfer_deadlock_fixed(src: Account, dst: Account, amount: int) -> None:
    first, second = (src, dst) if src.id < dst.id else (dst, src)
    with first.lock:
        time.sleep(0.0005)
        with second.lock:
            if src.balance >= amount:
                src.balance -= amount
                dst.balance += amount


TRANSFER_FNS = {
    "unsafe":         transfer_unsafe,
    "safe":           transfer_safe,
    "deadlock":       transfer_deadlock,
    "deadlock-fixed": transfer_deadlock_fixed,
}


def worker(accounts, transfer_fn, n_ops: int, seed: int):
    rng = random.Random(seed)
    n = len(accounts)
    for _ in range(n_ops):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        transfer_fn(accounts[i], accounts[j], rng.randint(1, 50))


def run_mode(mode: str,
             n_accounts: int,
             n_threads: int,
             ops_per_thread: int,
             seed: int = 42,
             join_timeout: float = 4.0) -> dict:
    random.seed(seed)
    accounts = make_accounts(n_accounts)
    expected = total_money(accounts)
    fn = TRANSFER_FNS[mode]
    threads = [
        threading.Thread(
            target=worker,
            args=(accounts, fn, ops_per_thread, seed + t),
            daemon=True,
        )
        for t in range(n_threads)
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()
    deadline = start + join_timeout
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.perf_counter()))
    elapsed = time.perf_counter() - start
    hung = sum(1 for t in threads if t.is_alive())
    actual = total_money(accounts)
    diff = actual - expected
    return {
        "mode":               mode,
        "accounts":           n_accounts,
        "threads":            n_threads,
        "ops_per_thread":     ops_per_thread,
        "planned_operations": n_threads * ops_per_thread,
        "elapsed_s":          round(elapsed, 3),
        "expected_total":     expected,
        "actual_total":       actual,
        "diff":               diff,
        "hung_threads":       hung,
        "consistent":         diff == 0 and hung == 0,
    }


def print_result(r: dict) -> None:
    print(f"\n=== Режим: {r['mode'].upper()} ===")
    print(f"  Рахунків:             {r['accounts']}")
    print(f"  Потоків:              {r['threads']}")
    print(f"  Операцій на потік:    {r['ops_per_thread']}")
    print(f"  Заплановано операцій: {r['planned_operations']}")
    print(f"  Час виконання:        {r['elapsed_s']} с")
    print(f"  Сума до:              {r['expected_total']}")
    print(f"  Сума після:           {r['actual_total']}")
    print(f"  Різниця:              {r['diff']}")
    print(f"  Заблоковано потоків:  {r['hung_threads']}")
    print(f"  Узгодженість:         {'OK' if r['consistent'] else 'ПОРУШЕНА'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bench",
                    choices=list(TRANSFER_FNS.keys()) + ["bench"])
    ap.add_argument("--accounts", type=int, default=150)
    ap.add_argument("--threads", type=int, default=1200)
    ap.add_argument("--ops", type=int, default=100)
    ap.add_argument("--output", default="results_task1.json")
    args = ap.parse_args()

    if args.mode != "bench":
        print_result(run_mode(args.mode, args.accounts, args.threads, args.ops))
        return

    results = []
    print("=" * 70)
    print(f"Bench: рахунків={args.accounts}, потоків={args.threads}, "
          f"операцій/потік={args.ops}")
    print("=" * 70)
    for mode in ["unsafe", "safe", "deadlock", "deadlock-fixed"]:
        if mode == "deadlock":
            ops = 20
        elif mode == "deadlock-fixed":
            ops = 30
        else:
            ops = args.ops
        r = run_mode(mode, args.accounts, args.threads, ops)
        print_result(r)
        results.append(r)

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nЗбережено результати у {args.output}")


if __name__ == "__main__":
    main()
