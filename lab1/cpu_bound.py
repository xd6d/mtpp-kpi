"""
CPU-bound задачі:
  1) Обчислення числа π методом Монте-Карло
  2) Факторизація великих чисел
  3) Пошук простих чисел у заданому діапазоні

Виконуються послідовно, через ThreadPoolExecutor (GIL — не дає реального прискорення)
та через ProcessPoolExecutor (справжній паралелізм).
"""
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


# ------- Монте-Карло -------
def monte_carlo_pi(iterations: int, seed: int = 0) -> float:
    rng = random.Random(seed)
    inside = 0
    for _ in range(iterations):
        x, y = rng.random(), rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / iterations


# ------- Факторизація -------
def factorize(n: int) -> list[int]:
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


# ------- Прості числа в діапазоні -------
def primes_in_range(rng: tuple[int, int]) -> int:
    lo, hi = rng
    count = 0
    for n in range(max(2, lo), hi):
        if n < 4:
            count += 1
            continue
        if n % 2 == 0:
            continue
        is_prime = True
        for d in range(3, int(math.isqrt(n)) + 1, 2):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count


# ------- Підготовка задач (однаковий загальний обсяг роботи для всіх режимів) -------
PI_TOTAL_ITERS = 4_000_000
PI_CHUNKS = 8                                     # 8 незалежних підзадач
PI_CHUNK = PI_TOTAL_ITERS // PI_CHUNKS

FACTORIZE_NUMBERS = [
    961_748_927, 982_451_653, 999_999_937,
    1_000_000_007, 1_000_000_033, 1_000_000_087,
    1_000_000_093, 1_000_000_097,
]                                                 # 8 великих простих/майже-простих

PRIMES_RANGES = [(i * 200_000, (i + 1) * 200_000) for i in range(8)]


def run_sequential():
    t0 = time.perf_counter()
    # π
    pi_estimates = [monte_carlo_pi(PI_CHUNK, seed=i) for i in range(PI_CHUNKS)]
    pi_val = sum(pi_estimates) / len(pi_estimates)
    # Факторизація
    _ = [factorize(n) for n in FACTORIZE_NUMBERS]
    # Прості числа
    prime_counts = [primes_in_range(r) for r in PRIMES_RANGES]
    elapsed = time.perf_counter() - t0
    return elapsed, pi_val, sum(prime_counts)


def run_threads(workers: int):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pi_estimates = list(ex.map(monte_carlo_pi, [PI_CHUNK] * PI_CHUNKS, range(PI_CHUNKS)))
        list(ex.map(factorize, FACTORIZE_NUMBERS))
        prime_counts = list(ex.map(primes_in_range, PRIMES_RANGES))
    elapsed = time.perf_counter() - t0
    pi_val = sum(pi_estimates) / len(pi_estimates)
    return elapsed, pi_val, sum(prime_counts)


def run_processes(workers: int):
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pi_estimates = list(ex.map(monte_carlo_pi, [PI_CHUNK] * PI_CHUNKS, range(PI_CHUNKS)))
        list(ex.map(factorize, FACTORIZE_NUMBERS))
        prime_counts = list(ex.map(primes_in_range, PRIMES_RANGES))
    elapsed = time.perf_counter() - t0
    pi_val = sum(pi_estimates) / len(pi_estimates)
    return elapsed, pi_val, sum(prime_counts)


if __name__ == "__main__":
    print(f"CPU cores: {os.cpu_count()}")
    seq_t, pi_val, primes = run_sequential()
    print(f"Sequential:   {seq_t:.3f}s   π≈{pi_val:.5f}   primes={primes}")
    for w in (2, 4, 8):
        t, _, _ = run_threads(w)
        print(f"Threads w={w}: {t:.3f}s   speedup={seq_t / t:.2f}x")
    for w in (2, 4, 8):
        t, _, _ = run_processes(w)
        print(f"Process w={w}: {t:.3f}s   speedup={seq_t / t:.2f}x")
