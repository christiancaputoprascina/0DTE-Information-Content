# The Information Content of 0DTE Options

Replication code for the MSc thesis *The Information Content of 0DTE Options: Implied Variance
and Expiring Gamma Measures for S&P 500 Variance* (Christian Caputo Prascina, MSc in Accounting,
Financial Management and Control, Bocconi University, 2026).

The code uses one-minute quotes of S&P 500 options expiring the same day (0DTE SPXW) for the 250
trading sessions of 2025 and produces the tables, figures and numbers reported in the thesis.

## Contents

```
Code/       scripts 01–18 (run in order), common.py, run_replication.py
results/    tables (CSV) and figures (PDF) of the thesis, as produced by the code
tests/      check_thesis_numbers.py
```

## Data

The data are not included: the Cboe DataShop and Bloomberg licenses do not allow their
redistribution. They go in `CBOE/Data/00_inputs/`:

| File | Source |
|---|---|
| `CBOE_raw/UnderlyingOptionsIntervals_60sec_calcs_oi_2025-MM-DD.zip`, one per trading day | Cboe DataShop, SPX Option Quote Intervals (1-minute, with calculations and open interest) |
| `SPX_excel/Jan.xlsx` … `Dec.xlsx`, one-minute S&P 500 bars | Bloomberg |
| `SPY_excel/25_01.xlsx` … `25_12.xlsx`, one-minute SPY bars (Appendix B) | Bloomberg |
| `sofr_daily_2025.csv`, daily SOFR | Federal Reserve Bank of New York |

## Usage

```bash
pip install -r requirements.txt
python3 Code/run_replication.py         # runs scripts 01–18 and updates results/
python3 tests/check_thesis_numbers.py   # compares the outputs with the numbers in the thesis
```

`python3 Code/run_replication.py --check` checks the environment and the data without running
the scripts. Each script can also be run on its own; outputs are written to `CBOE/Data/`.
The code was run with Python 3.12 and the package versions in `requirements.txt`.

## Scripts

| Script | Content |
|---|---|
| `01_Filter_0DTE.py` | 0DTE quotes of each session, cut at the close |
| `02_Black76_IV_greeks.py` | forward from put-call parity, Black '76 implied volatility and greeks |
| `03_MFIV.py` | model-free implied variance |
| `04_Gamma_Herfindahl.py` | concentration of expiring gamma (H^OI, H^VOL) |
| `05_SPX_minute_bars.py` | S&P 500 minute bars and realized variance |
| `06_SPY_minute_bars.py` | SPY minute bars and realized variance |
| `07_Daily_panel.py` | daily panel |
| `08_Intraday_panel.py` | intraday panel |
| `09_Appendix_A_checks.py` | data checks and comparison with vendor values |
| `10_Surface_descriptives.py` | implied-volatility surface and gamma descriptives |
| `11_MFIV_diagnostics.py` | MFIV strike-strip diagnostics |
| `12_RV_signature.py` | volatility signature |
| `13_Daily_regressions.py` | daily designs A1–A5 and B1 |
| `14_Daily_MZ_robustness.py` | Mincer–Zarnowitz regressions and daily robustness checks |
| `15_Intraday_regressions.py` | intraday designs I1–I3 |
| `16_Intraday_robustness.py` | intraday robustness checks and effect sizes |
| `17_Figures.py` | figures |
| `18_Text_numbers.py` | numbers quoted in the text |

## Tables and figures

| Thesis | Script | File in `results/` |
|---|---|---|
| Table 2.1 | 10 | `tables/10_table_2_1_2025.csv` |
| Table 2.2 | 10 | `tables/10_table_2_2_2025.csv` |
| Table 3.1 | 11 | `tables/11_table_3_1_2025.csv` |
| Tables 5.1, 5.2 | 13, 16 | `tables/16_gamma_effect_size_ci.csv`, `tables/13_daily_hvol_only_2025.csv` |
| Table 5.3 | 14 | `tables/14_mincer_zarnowitz_by_subsample_2025.csv` |
| Table 5.4 | 14 | `tables/14_mz_ex_ante_2025.csv` |
| Table 5.5 | 15 | `tables/15_registered_spec_check_terciles.csv` |
| Table B.1 | 12 | `tables/12_rv_signature_spx_spy_2025.csv` |
| Table D.2 | 16 | `tables/16_hvol_breadth_controls_intraday.csv` |
| Tables D.3, D.4 | 13 | `tables/13_daily_regression_results_2025.csv`, `tables/13_daily_hvol_only_2025.csv` |
| Tables D.5–D.7 | 15 | `tables/15_registered_spec_check_results.csv` |
| Figures | 17 | `figures/` |

## Citation

```bibtex
@mastersthesis{CaputoPrascina2026,
  author = {Caputo Prascina, Christian},
  title  = {The Information Content of 0DTE Options: Implied Variance and Expiring Gamma Measures for {S\&P} 500 Variance},
  school = {Bocconi University},
  type   = {{MSc} thesis},
  year   = {2026}
  note   = {Thesis supervisor: Prof. Claudio Tebaldi}
}
```

## License

Code: MIT License (see `LICENSE`). The license does not cover the data, which are not distributed.
