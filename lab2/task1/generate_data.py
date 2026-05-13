"""Генерація вхідних даних для Завдання 1:
- 1000 HTML-документів (різного розміру / наповнення)
- Масив з понад 1_000_000 чисел (numpy .npy)
- Дві матриці > 1000 елементів (1000x1000) для множення (numpy .npy)
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "data"
HTML_DIR = ROOT / "html_docs"
ARRAY_PATH = ROOT / "array.npy"
MAT_A_PATH = ROOT / "matrix_a.npy"
MAT_B_PATH = ROOT / "matrix_b.npy"

TAGS = ["div", "span", "p", "a", "img", "li", "ul", "ol", "h1", "h2",
        "h3", "table", "tr", "td", "th", "section", "article", "nav", "footer", "header"]


def _build_html(tag_budget: int) -> str:
    parts = ["<!doctype html><html><head><title>doc</title></head><body>"]
    for _ in range(tag_budget):
        t = random.choice(TAGS)
        parts.append(f"<{t}>x</{t}>")
    parts.append("</body></html>")
    return "".join(parts)


def generate_html(n_files: int = 1000, seed: int = 42) -> None:
    random.seed(seed)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        budget = random.randint(40, 400)  # різний розмір
        path = HTML_DIR / f"doc_{i:04d}.html"
        path.write_text(_build_html(budget), encoding="utf-8")
    print(f"[ok] {n_files} HTML-документів у {HTML_DIR}")


def generate_array(n: int = 2_000_000, seed: int = 7) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # суміш розподілів — підкреслимо, що не нормальний
    a = rng.exponential(scale=5.0, size=n) + rng.uniform(-10, 10, size=n)
    np.save(ARRAY_PATH, a.astype(np.float64))
    print(f"[ok] масив {n} чисел -> {ARRAY_PATH} ({ARRAY_PATH.stat().st_size/1e6:.1f} MB)")


def generate_matrices(n: int = 1000, seed: int = 11) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float64)
    B = rng.standard_normal((n, n)).astype(np.float64)
    np.save(MAT_A_PATH, A)
    np.save(MAT_B_PATH, B)
    print(f"[ok] матриці {n}x{n} -> {MAT_A_PATH.name}, {MAT_B_PATH.name}")


if __name__ == "__main__":
    generate_html()
    generate_array()
    generate_matrices()
