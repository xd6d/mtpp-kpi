"""Побудова графіків часу та прискорення для звіту."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT.parent / "plots"
PLOTS.mkdir(exist_ok=True)


def _by_pattern(rows, pattern):
    items = [r for r in rows if r["pattern"] == pattern]
    items.sort(key=lambda r: r["workers"])
    return [r["workers"] for r in items], [r["time"] for r in items]


def plot_task1(name: str, payload: dict, title: str) -> None:
    baseline = payload["baseline"]
    rows = payload["runs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for pat in ("Map-Reduce", "Fork-Join", "Worker Pool"):
        ws, ts = _by_pattern(rows, pat)
        ax1.plot(ws, ts, marker="o", label=pat)
        ax2.plot(ws, [baseline / t for t in ts], marker="o", label=pat)
    ax1.axhline(baseline, ls="--", c="gray", label=f"Sequential ({baseline:.3f}s)")
    ax1.set_xlabel("workers")
    ax1.set_ylabel("time, s")
    ax1.set_title(f"{title}: час")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.axhline(1.0, ls="--", c="gray", label="Sequential = 1×")
    ax2.set_xlabel("workers")
    ax2.set_ylabel("speedup ×")
    ax2.set_title(f"{title}: прискорення")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOTS / f"{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("[ok]", out)


def plot_task2(payload: dict) -> None:
    baseline = payload["baseline"]
    rows = payload["runs"]
    labels, times = [], []
    for r in rows:
        labels.append(f"{r['pattern']}\n(w={r['workers']})")
        times.append(r["time"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#888"] + ["#4C78A8"] + ["#F58518"] * (len(rows) - 2)
    ax1.bar(range(len(rows)), times, color=colors)
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("time, s")
    ax1.set_title("Задача 2: час")
    ax1.grid(alpha=0.3, axis="y")

    speedups = [baseline / t for t in times]
    ax2.bar(range(len(rows)), speedups, color=colors)
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("speedup ×")
    ax2.set_title("Задача 2: прискорення")
    ax2.axhline(1.0, ls="--", c="gray")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = PLOTS / "task2.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("[ok]", out)


def plot_summary(data: dict) -> None:
    """Стовпчики: найкраще прискорення для кожної задачі та патерну."""
    tasks = [("task1_html", "HTML tags"), ("task1_array", "Array stats"),
             ("task1_matrix", "Matrix mult"), ("task2", "Transactions")]
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.25
    patterns_t1 = ["Map-Reduce", "Fork-Join", "Worker Pool"]
    patterns_t2 = ["Pipeline", "Producer-Consumer"]
    xs = list(range(len(tasks)))
    for i, p in enumerate(patterns_t1):
        vals = []
        for tkey, _ in tasks:
            baseline = data[tkey]["baseline"]
            same = [r for r in data[tkey]["runs"] if r["pattern"] == p]
            if same:
                best = max(baseline / r["time"] for r in same)
            else:
                best = 0
            vals.append(best)
        ax.bar([x + (i - 1) * width for x in xs], vals, width, label=p)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in tasks])
    ax.set_ylabel("найкраще прискорення ×")
    ax.axhline(1.0, ls="--", c="gray")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Найкраще прискорення (відносно Sequential) по задачах та патернах")
    fig.tight_layout()
    out = PLOTS / "summary.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("[ok]", out)


def main():
    data = json.load(open(ROOT / "results.json", "r", encoding="utf-8"))
    plot_task1("task1_html", data["task1_html"], "Підрахунок HTML-тегів")
    plot_task1("task1_array", data["task1_array"], "Статистика масиву")
    plot_task1("task1_matrix", data["task1_matrix"], "Множення матриць")
    plot_task2(data["task2"])
    plot_summary(data)


if __name__ == "__main__":
    main()
