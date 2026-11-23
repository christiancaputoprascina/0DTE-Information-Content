"""
check_thesis_numbers.py: summarizes the comparisons between the outputs and the numbers
reported in the thesis (files NN_thesis_check.csv written by the scripts).

  python3 tests/check_thesis_numbers.py

Expected: 610 comparisons; 601 match, 6 are informational and 3 concern the wording of
three sentences rather than numbers. Exit code 0 if the result is as expected.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "Code"))

import pandas as pd  # noqa: E402

from common import DATA_DIR  # noqa: E402

EXPECTED = {"OK": 601, "INFO": 6, "DIVERSO": 3}
KNOWN_DIFFERENCES = {
    ("09_thesis_check.csv", "agreement closer near ATM"),
    ("10_thesis_check.csv", "factor of about six"),
    ("11_thesis_check.csv", "give the same strips"),
}


def main():
    files = sorted(DATA_DIR.glob("*/*_thesis_check.csv"))
    if not files:
        sys.exit(f"No *_thesis_check.csv files in {DATA_DIR}: run the pipeline first.")

    totals, unexpected = {}, []
    print(f"{'file':42s} {'OK':>5s} {'INFO':>5s} {'DIVERSO':>8s}")
    for f in files:
        d = pd.read_csv(f)
        counts = d["esito"].astype(str).value_counts().to_dict()
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        print(f"{f.parent.name + '/' + f.name:42s} {counts.get('OK', 0):5d} {counts.get('INFO', 0):5d} "
              f"{counts.get('DIVERSO', 0):8d}")
        for _, row in d[d["esito"].astype(str) == "DIVERSO"].iterrows():
            voce = str(row["voce"])
            if not any(f.name == name and key in voce for name, key in KNOWN_DIFFERENCES):
                unexpected.append(f"{f.name}, p. {row['pagina']}: {voce}")

    n = sum(totals.values())
    print(f"\nTotal: {n} comparisons, " + ", ".join(f"{k} {v}" for k, v in sorted(totals.items())))
    if totals == EXPECTED and not unexpected:
        print("Result as expected: every number checked is reproduced.")
        sys.exit(0)
    for item in unexpected:
        print(f"Unexpected difference: {item}")
    print(f"Result differs from the expected {EXPECTED}.")
    sys.exit(1)


if __name__ == "__main__":
    main()
