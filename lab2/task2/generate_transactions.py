"""Генерація вхідних транзакцій для Завдання 2.

Кожна транзакція — це рядок CSV:
    ID_користувача, сума, валюта, дата, тип_товару
"""
from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data"
TX_FILE = ROOT / "transactions.csv"

CURRENCIES = ["USD", "EUR", "UAH", "GBP", "PLN"]
CATEGORIES = ["electronics", "books", "groceries", "fashion", "services", "travel"]


def generate(n_rows: int = 200_000, seed: int = 2026) -> Path:
    random.seed(seed)
    ROOT.mkdir(parents=True, exist_ok=True)
    start = datetime(2025, 1, 1)
    with TX_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "amount", "currency", "date", "category"])
        for _ in range(n_rows):
            uid = random.randint(1, 50_000)  # >ID 25000 — повернемо 20% коштів
            amount = round(random.uniform(1, 5000), 2)
            cur = random.choice(CURRENCIES)
            date = (start + timedelta(seconds=random.randint(0, 365 * 24 * 3600))).strftime("%Y-%m-%d")
            cat = random.choice(CATEGORIES)
            w.writerow([uid, amount, cur, date, cat])
    print(f"[ok] {n_rows} транзакцій -> {TX_FILE} ({TX_FILE.stat().st_size/1e6:.1f} MB)")
    return TX_FILE


if __name__ == "__main__":
    generate()
