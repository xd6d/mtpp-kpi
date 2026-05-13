"""
Будує графіки часу виконання та прискорення для трьох типів задач.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data", "results.json"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)


def _xy(d):
    keys = sorted(int(k) for k in d.keys())
    return keys, [d[str(k)] if str(k) in d else d[k] for k in keys]


def plot_block(results, title, fname):
    seq = results["sequential"]
    tw, tv = _xy(results["threads"])
    pw, pv = _xy(results["processes"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(tw, tv, "o-", label="Threads", color="#1f77b4")
    ax1.plot(pw, pv, "s--", label="Processes", color="#d62728")
    ax1.axhline(seq, ls=":", color="gray", label=f"Sequential ({seq:.2f}s)")
    ax1.set_xlabel("Workers")
    ax1.set_ylabel("Час, с")
    ax1.set_title(f"{title} — час виконання")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(tw, [seq / v for v in tv], "o-", label="Threads", color="#1f77b4")
    ax2.plot(pw, [seq / v for v in pv], "s--", label="Processes", color="#d62728")
    ax2.axhline(1.0, ls=":", color="gray")
    ax2.set_xlabel("Workers")
    ax2.set_ylabel("Speedup, x")
    ax2.set_title(f"{title} — прискорення")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG, fname)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def plot_summary(results):
    """Зведена діаграма-стовпчики: найкращий speedup кожного типу задач."""
    labels = ["CPU-bound", "Memory-bound", "I/O-bound"]
    keys = ["cpu", "memory", "io"]
    seq_times, best_thr, best_proc = [], [], []
    for k in keys:
        r = results[k]
        seq = r["sequential"]
        seq_times.append(seq)
        best_thr.append(seq / min(float(v) for v in r["threads"].values()))
        best_proc.append(seq / min(float(v) for v in r["processes"].values()))
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - w / 2 for i in x], best_thr, w, label="Threads (best)", color="#1f77b4")
    ax.bar([i + w / 2 for i in x], best_proc, w, label="Processes (best)", color="#d62728")
    for i, (t, p) in enumerate(zip(best_thr, best_proc)):
        ax.text(i - w / 2, t + 0.1, f"{t:.2f}x", ha="center", fontsize=9)
        ax.text(i + w / 2, p + 0.1, f"{p:.2f}x", ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("Найкращий speedup")
    ax.set_title("Підсумкове прискорення за типом задач")
    ax.axhline(1.0, ls=":", color="gray")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    out = os.path.join(FIG, "summary.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    with open(DATA) as f:
        results = json.load(f)
    plot_block(results["cpu"], "CPU-bound", "cpu_bound.png")
    plot_block(results["memory"], "Memory-bound", "memory_bound.png")
    plot_block(results["io"], "I/O-bound", "io_bound.png")
    plot_summary(results)
