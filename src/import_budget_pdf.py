"""Import the October 2027 budget ('2027 Sales Objectives by Café', the one-page PDF prepared October 2026 from
PK_2027_Sales_Budget_Workbook.xlsx) into data/budget_2027_oct.csv.

    .venv/bin/python -I src/import_budget_pdf.py path/to/2027_Parakeet_Sales_Budget.pdf

Reads the PDF's text (pdftotext -layout) and writes one row per café: 2026 estimated finish, 2027 objective, their
growth, the $ increase and the key assumption. Every row is checked against the PDF's own totals and percentages.
The PDF shows La Jolla Café and La Jolla Juicery as one line.

Standalone on purpose (no project imports) so it can run in isolated mode on an untrusted file.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "budget_2027_oct.csv"
# budget name -> Toast store name(s); new cafés do not trade yet
STORE = {"Beverly Hills": "Beverly Hills", "Brentwood": "Brentwood", "Carlsbad": "Carlsbad", "Coronado": "Coronado",
         "Del Mar Café": "Carmel Valley", "La Jolla Café & Juicery": "La Jolla|La Jolla Juicery",
         "Little Italy Café": "Little Italy", "Newport Beach": "Newport", "La Jolla Cove": "La Jolla Cove"}
MONEY = r"\$([\d,]+)"
PCT = r"\(?([\d.]+)%\)?"


def money(s: str) -> float:
    return float(s.replace(",", ""))


def main(path: str) -> None:
    text = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in text.splitlines():
        line = line.strip()
        # existing cafés: name, 2026 finish, 2026 growth, 2027 objective, 2027 growth, $ increase, key assumption
        m = re.match(rf"^(.+?)\s+{MONEY}\s+{PCT}\s+{MONEY}\s+{PCT}\s+{MONEY}\s*(.*)$", line)
        if not m:
            continue
        name = re.sub(r"\s*\(.*\)$", "", m.group(1)).strip()
        rows.append({"budget_name": name, "store": STORE.get(name), "est_2026": money(m.group(2)),
                     "objective_2027": money(m.group(4)), "growth_2027": float(m.group(5)) / 100,
                     "increase_2027": money(m.group(6)), "key_assumption": m.group(7).strip()})
    df = pd.DataFrame(rows)
    totals = df[df.budget_name.isin(["Existing Cafés Total", "New & Remodeled Total", "TOTAL COMPANY"])]
    cafes = df[df.store.notna()].copy()
    nine = cafes[cafes.store != "La Jolla Cove"]  # the nine existing cafés, on eight lines
    exist = totals.set_index("budget_name").loc["Existing Cafés Total"]
    assert len(nine) == 8 and len(cafes) == 9, cafes.budget_name.tolist()
    assert abs(nine.est_2026.sum() - exist.est_2026) < 1 and abs(nine.objective_2027.sum() - exist.objective_2027) < 1
    for r in cafes.itertuples():
        assert abs(r.objective_2027 - r.est_2026 - r.increase_2027) <= 1, r.budget_name
        assert abs(round((r.objective_2027 / r.est_2026 - 1) * 100, 1) - r.growth_2027 * 100) < 0.051, r.budget_name
    OUT.parent.mkdir(exist_ok=True)
    cafes.to_csv(OUT, index=False)
    print(cafes.to_string(index=False))
    print(f"nine existing cafés: {nine.est_2026.sum():,.0f} -> {nine.objective_2027.sum():,.0f} "
          f"({nine.objective_2027.sum() / nine.est_2026.sum() - 1:+.2%}); wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
