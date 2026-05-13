"""
Лабораторна робота №3 - Завдання 2.
Передача даних між процесами різного роду.

Основний процес  - Python (фреймворк multiprocessing/subprocess + socket + mmap).
Допоміжний процес - Node.js (фреймворк child_process + net + fs/mmap).

Три методи передачі даних, що порівнюємо:
  1) pipe   - stdin/stdout (Message Passing, локальний канал ОС)
  2) socket - TCP socket loopback (Message Passing з мережевим стеком)
  3) shmem  - mmap файл (Shared Memory між різними мовами)

Для кожного методу N разів виконуємо круг
"main -> random int -> worker logs -> back to main" та міряємо середній час.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER = SCRIPT_DIR / "worker.js"


def _node_cmd(mode: str, *extra: str) -> list[str]:
    return ["node", str(WORKER), mode, *extra]


# --------------------------- 1) PIPE (stdin/stdout) -------------------------
def run_pipe(rounds: int) -> dict:
    """Передача через stdin/stdout процесу Node.js (message passing)."""
    proc = subprocess.Popen(
        _node_cmd("pipe"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
    )
    assert proc.stdin and proc.stdout

    times = []
    for _ in range(rounds):
        value = random.randint(0, 1_000_000)
        t0 = time.perf_counter()
        proc.stdin.write(f"{value}\n".encode())
        proc.stdin.flush()
        line = proc.stdout.readline()
        t1 = time.perf_counter()
        echoed = int(line.strip())
        assert echoed == value, f"pipe mismatch {echoed} != {value}"
        times.append(t1 - t0)

    proc.stdin.close()
    proc.wait(timeout=5)
    return _summary("pipe", times)


# --------------------------- 2) TCP SOCKET ----------------------------------
def run_socket(rounds: int) -> dict:
    """Передача через TCP loopback (message passing з мережевим стеком)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    proc = subprocess.Popen(
        _node_cmd("socket", str(port)),
        stderr=None,
    )
    conn, _ = server.accept()
    conn_file = conn.makefile("rwb", buffering=0)

    times = []
    for _ in range(rounds):
        value = random.randint(0, 1_000_000)
        t0 = time.perf_counter()
        conn_file.write(f"{value}\n".encode())
        line = conn_file.readline()
        t1 = time.perf_counter()
        echoed = int(line.strip())
        assert echoed == value, f"socket mismatch {echoed} != {value}"
        times.append(t1 - t0)

    conn_file.close()
    conn.close()
    server.close()
    proc.wait(timeout=5)
    return _summary("socket", times)


# --------------------------- 3) SHARED MEMORY (mmap) ------------------------
# Layout у файлі (32 байти):
#   [0..3]   uint32  little-endian   request id (інкремент)
#   [4..7]   uint32                  request value
#   [8..11]  uint32                  response id (=request id після відповіді)
#   [12..15] uint32                  response value
#   [16..31] reserved
def run_shmem(rounds: int) -> dict:
    """Передача через mmap-файл (shared memory між мовами)."""
    fd, path = tempfile.mkstemp(prefix="ipc_shmem_", suffix=".bin")
    os.close(fd)
    Path(path).write_bytes(b"\x00" * 32)

    proc = subprocess.Popen(
        _node_cmd("shmem", path),
        stderr=None,
    )

    try:
        f = open(path, "r+b")
        mm = mmap.mmap(f.fileno(), 32)
        times = []
        for i in range(1, rounds + 1):
            value = random.randint(0, 1_000_000)
            t0 = time.perf_counter()
            # пишемо request: value спочатку, id (барʼєр) - в кінці
            mm[4:8] = struct.pack("<I", value)
            mm[0:4] = struct.pack("<I", i)
            # чекаємо response.id == i (busy-wait, бо очікувані затримки <1мс)
            while True:
                resp_id = struct.unpack("<I", mm[8:12])[0]
                if resp_id == i:
                    break
            t1 = time.perf_counter()
            echoed = struct.unpack("<I", mm[12:16])[0]
            assert echoed == value, f"shmem mismatch {echoed} != {value}"
            times.append(t1 - t0)
        # сигнал воркеру про завершення
        mm[0:4] = struct.pack("<I", 0xFFFFFFFF)
        mm.close()
        f.close()
    finally:
        proc.wait(timeout=5)
        try:
            os.unlink(path)
        except OSError:
            pass

    return _summary("shmem", times)


# --------------------------- helpers ----------------------------------------
def _summary(name: str, times: list[float]) -> dict:
    n = len(times)
    total = sum(times)
    return {
        "method":         name,
        "rounds":         n,
        "total_s":        round(total, 4),
        "avg_us":         round(total / n * 1e6, 2),
        "min_us":         round(min(times) * 1e6, 2),
        "max_us":         round(max(times) * 1e6, 2),
        "throughput_ops": round(n / total, 1) if total > 0 else 0,
    }


def print_summary(s: dict) -> None:
    print(f"\n=== Метод: {s['method'].upper()} ===")
    print(f"  Кругів:        {s['rounds']}")
    print(f"  Загалом:       {s['total_s']} с")
    print(f"  Середн. круг:  {s['avg_us']} мкс")
    print(f"  Мін / Макс:    {s['min_us']} / {s['max_us']} мкс")
    print(f"  Пропускна:     {s['throughput_ops']} оп/с")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all",
                    choices=["pipe", "socket", "shmem", "all"])
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--output", default="results_task2.json")
    args = ap.parse_args()

    if not WORKER.exists():
        sys.exit(f"Не знайдено worker.js: {WORKER}")

    random.seed(0)
    fns = {"pipe": run_pipe, "socket": run_socket, "shmem": run_shmem}
    results = []

    print("=" * 60)
    print(f"IPC demo: кругів={args.rounds}")
    print("=" * 60)

    methods = [args.method] if args.method != "all" else ["pipe", "socket", "shmem"]
    for m in methods:
        s = fns[m](args.rounds)
        print_summary(s)
        results.append(s)

    if len(results) > 1:
        results.sort(key=lambda r: r["avg_us"])
        print("\nРейтинг (найкоротший круг -> найдовший):")
        for r in results:
            print(f"  {r['method']:<8} {r['avg_us']:>10.2f} мкс/круг")

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nЗбережено результати у {args.output}")


if __name__ == "__main__":
    main()
