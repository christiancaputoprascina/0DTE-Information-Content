"""
run_replication.py: runs scripts 01-18 in order and updates results/.

  python3 Code/run_replication.py             check, run 01 ... 18, update results/
  python3 Code/run_replication.py --check     only check Python, packages and input data
  python3 Code/run_replication.py --from 07   start from step 07 (outputs of 01-06 must exist)
  python3 Code/run_replication.py --to 06     stop after step 06
  python3 Code/run_replication.py --results   only update results/ from the existing outputs

Each script runs in its own process; its output goes to CBOE/Data/_logs/.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import pandas as pd

from common import DATA_DIR, INPUT_DIR, ROOT, YEAR

CODE_DIR = Path(__file__).absolute().parent
STEPS = sorted(CODE_DIR.glob("[0-9][0-9]_*.py"))
LOG_DIR = DATA_DIR / "_logs"
RESULTS_DIR = ROOT / "results"

# Tables and figures copied to results/ (columns with file paths or internal checks are dropped).
RESULT_TABLES = [
    "10_surface_descriptives/10_table_2_1_2025.csv",
    "10_surface_descriptives/10_table_2_2_2025.csv",
    "11_mfiv_diagnostics/11_table_3_1_2025.csv",
    "12_rv_signature/12_rv_signature_spx_spy_2025.csv",
    "13_daily_regressions/13_daily_regression_results_2025.csv",
    "13_daily_regressions/13_daily_hvol_only_2025.csv",
    "14_daily_mz_robustness/14_mincer_zarnowitz_by_subsample_2025.csv",
    "14_daily_mz_robustness/14_mz_ex_ante_2025.csv",
    "15_intraday_regressions/15_registered_spec_check_results.csv",
    "15_intraday_regressions/15_registered_spec_check_terciles.csv",
    "16_intraday_robustness/16_gamma_effect_size_ci.csv",
    "16_intraday_robustness/16_hvol_breadth_controls_intraday.csv",
]
DROPPED_COLUMNS = {"merged_panel_file", "valore_tesi", "fonte", "convenzione"}
RESULT_FIGURES = [
    "17_fig_A1_smiles_representative_days.pdf",
    "17_fig_S2D_iv_surface_3d_2025-09-25_full_day.pdf",
    "17_fig_S2D_iv_surface_3d_2025-04-15_full_day.pdf",
    "17_fig_S2D_iv_surface_3d_2025-04-07_full_day.pdf",
    "17_fig_B1_gamma_time_profile.pdf",
    "17_fig_B2_gamma_oi_share_by_moneyness.pdf",
    "17_fig_B3_gamma_vol_share_by_moneyness.pdf",
    "17_fig_C12_liquidity_by_moneyness.pdf",
    "17_fig_R1_realized_vs_implied_by_regime.pdf",
    "17_fig_R3_implied_vs_realized_vol_2025.pdf",
    "17_fig_R4_log_gap_distribution_2025.pdf",
    "17_fig_R2_convexity_effects_by_horizon.pdf",
    "17_fig_A2_iv_atm_by_time.pdf",
    "17_fig_A3_skew_curvature_by_time.pdf",
]


def step_number(script: Path) -> int:
    return int(script.name[:2])


def check_environment(first: int, last: int) -> bool:
    """Prints what is available and returns False if a requested step cannot run."""
    ok = True
    print("Checks")

    version = ".".join(map(str, sys.version_info[:3]))
    note = "" if sys.version_info[:2] == (3, 12) else "  (tested with 3.12)"
    print(f"  Python {version}{note}")

    requirements = ROOT / "requirements.txt"
    for line in requirements.read_text().splitlines():
        m = re.match(r"\s*([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if not m:
            continue
        name, wanted = m.groups()
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            print(f"  MISSING  {name} {wanted}: run pip install -r requirements.txt")
            ok = False
            continue
        if found != wanted:
            print(f"  note     {name} {found} installed, results were produced with {wanted}")

    def need(step, condition, message):
        nonlocal ok
        if first <= step <= last and not condition:
            print(f"  MISSING  for step {step:02d}: {message}")
            ok = False

    raw = {f.name.split(".")[0] for f in (INPUT_DIR / "CBOE_raw").glob(
        f"UnderlyingOptionsIntervals_60sec_calcs_oi_{YEAR}-*") if f.suffix in (".csv", ".zip")}
    spx = [INPUT_DIR / "SPX_excel" / f"{m}.xlsx" for m in
           ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")]
    spy = [INPUT_DIR / "SPY_excel" / f"25_{i:02d}.xlsx" for i in range(1, 13)]
    spy_csv = DATA_DIR / "06_spy_minute_bars" / "06_SPY_2025_1min.csv"
    sofr = INPUT_DIR / "sofr_daily_2025.csv"

    print(f"  Cboe raw files (zip or csv): {len(raw)} days (250 expected)")
    print(f"  SPX Excel files: {sum(f.exists() for f in spx)} of 12")
    print(f"  SPY Excel files: {sum(f.exists() for f in spy)} of 12"
          + ("; unified SPY file present" if spy_csv.exists() else ""))
    print(f"  SOFR file: {'present' if sofr.exists() else 'missing'}")

    need(1, len(raw) > 0, f"Cboe raw files in {INPUT_DIR / 'CBOE_raw'}")
    need(2, sofr.exists(), f"{sofr}")
    need(5, all(f.exists() for f in spx), f"the 12 SPX Excel files in {INPUT_DIR / 'SPX_excel'}")
    need(6, all(f.exists() for f in spy) or spy_csv.exists(),
         f"the 12 SPY Excel files in {INPUT_DIR / 'SPY_excel'} or {spy_csv.name}")

    # starting later in the pipeline: the outputs of the earlier steps must already exist
    for script in STEPS:
        n = step_number(script)
        if n < first:
            folders = list(DATA_DIR.glob(f"{n:02d}_*"))
            if not folders or not any(f.is_file() for f in folders[0].rglob("*")):
                print(f"  MISSING  outputs of step {n:02d} (needed when starting from step {first:02d})")
                ok = False

    print("  ready" if ok else "  not ready: see the lines marked MISSING")
    return ok


def run_steps(first: int, last: int) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    times = []
    for script in STEPS:
        n = step_number(script)
        if not first <= n <= last:
            continue
        log = LOG_DIR / f"{script.stem}.log"
        print(f"{time.strftime('%H:%M:%S')}  {script.name} ...", end="", flush=True)
        start = time.time()
        with open(log, "w") as fh:
            result = subprocess.run([sys.executable, str(script)], stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
        minutes = (time.time() - start) / 60
        times.append((script.name, minutes))
        if result.returncode != 0:
            print(f" ERROR after {minutes:.1f} min (log: {log})")
            print("".join(log.read_text(errors="replace").splitlines(keepends=True)[-20:]))
            return False
        print(f" done in {minutes:.1f} min")
    print("All steps completed.")
    collect_results()
    print("Next: python3 tests/check_thesis_numbers.py")
    return True


def collect_results():
    """Copies the thesis tables (CSV) and figures (PDF) to results/."""
    tables, figures = RESULTS_DIR / "tables", RESULTS_DIR / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rel in RESULT_TABLES:
        src = DATA_DIR / rel
        if src.exists():
            df = pd.read_csv(src, dtype=str, keep_default_na=False)
            df.drop(columns=[c for c in df.columns if c in DROPPED_COLUMNS]).to_csv(tables / src.name, index=False)
            copied += 1
    for name in RESULT_FIGURES:
        src = DATA_DIR / "17_figures" / name
        if src.exists():
            shutil.copyfile(src, figures / name)
            copied += 1
    print(f"results/ updated: {copied} of {len(RESULT_TABLES) + len(RESULT_FIGURES)} files")


def main():
    parser = argparse.ArgumentParser(description="Runs the replication pipeline in order.")
    parser.add_argument("--check", action="store_true", help="only check the environment and the data")
    parser.add_argument("--results", action="store_true", help="only update results/ from existing outputs")
    parser.add_argument("--from", dest="first", type=int, default=1, help="first step (default 1)")
    parser.add_argument("--to", dest="last", type=int, default=18, help="last step (default 18)")
    args = parser.parse_args()

    if args.results:
        collect_results()
        return

    ready = check_environment(args.first, args.last)
    if args.check:
        sys.exit(0 if ready else 1)
    if not ready:
        sys.exit("Fix the missing items above, or choose other steps with --from / --to.")
    print()
    sys.exit(0 if run_steps(args.first, args.last) else 1)


if __name__ == "__main__":
    main()
