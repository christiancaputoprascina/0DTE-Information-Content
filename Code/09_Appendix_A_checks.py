"""
Controlli dell'Appendice A sui file per contratto con IV e greche Black '76.
  Parte 1: open_interest costante nella giornata per ogni contratto 0DTE, e controlli di integrita'.
  Parte 2: IV, delta e gamma Black '76 contro i campi del vendor vicino al denaro.
  Parte 3: passaggio completo sui 250 file (vendor, tau, r * tau, gamma di call e put, ask nulli,
           distanza fra strike) e confronto con i numeri stampati nella tesi.
Input:  CBOE/Data/02_black76_iv_greeks/ (file giornalieri e report annuale)
Output: CBOE/Data/09_appendix_a_checks/
"""

import glob
import os
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from common import DATA_DIR, EARLY_CLOSE_DATES, YEAR, output_dir, path, require

OUTPUT_DIR = output_dir("09_appendix_a_checks")

# File giornalieri per contratto di 02_Black76_IV_greeks.py: 250 attesi.
B76_DIR = DATA_DIR / "02_black76_iv_greeks"
B76_FILE_RE = re.compile(r"^02_0DTE_(\d{4})_(\d{2})_(\d{2})_black76_365\.csv$")
EXPECTED_DAYS = 250


# =====================================================================
# PARTE 1: OPEN INTEREST COSTANTE NELLA GIORNATA
# =====================================================================
# Per la specifica Cboe DataShop l'OI e' quello di inizio giornata (OCC). Un contratto
# (expiration, strike, option_type) varia se ha piu' di un valore distinto di open_interest,
# contando il mancante come valore. I controlli di integrita' non entrano nel verdetto.

OUT_OI_BY_DAY = "09_oi_constancy_by_day_2025.csv"      # una riga per giorno
OUT_OI_SUMMARY = "09_oi_constancy_summary_2025.csv"    # una riga, con il verdetto

COLS = ["quote_date", "quote_time", "root", "expiration", "strike",
        "option_type", "open_interest"]
KEY = ["expiration", "strike", "option_type"]


def check_day(path, day):
    df = pd.read_csv(path, usecols=COLS,
                     dtype={"quote_date": str, "quote_time": str, "root": str,
                            "expiration": str, "option_type": str,
                            "open_interest": str})
    raw_oi = df["open_interest"]
    oi = pd.to_numeric(raw_oi, errors="coerce")
    n_missing = int(raw_oi.isna().sum())
    n_nonnumeric = int((oi.isna() & raw_oi.notna()).sum())
    df["open_interest"] = oi
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.strftime("%Y-%m-%d")
    qd = pd.to_datetime(df["quote_date"]).dt.strftime("%Y-%m-%d")

    nun = df.groupby(KEY, dropna=False)["open_interest"].nunique(dropna=False)
    varying = nun[nun > 1]
    example = ""
    if len(varying) > 0:
        e, k, t = varying.index[0]
        sel = (df["expiration"] == e) & (df["strike"] == k) & (df["option_type"] == t)
        vals = df.loc[sel, "open_interest"].drop_duplicates().tolist()[:6]
        example = f"{e} {k} {t}: {vals}"

    return dict(
        quote_date=day,
        n_rows=len(df),
        n_minutes=int(df["quote_time"].nunique()),
        n_contracts=int(len(nun)),
        n_contracts_oi_varying=int(len(varying)),
        max_distinct_oi_per_contract=int(nun.max()) if len(nun) else 0,
        n_rows_oi_missing=n_missing,
        n_rows_oi_nonnumeric=n_nonnumeric,
        n_duplicate_rows=int(df.duplicated(subset=["quote_time"] + KEY).sum()),
        quote_date_matches_file=bool((qd == day).all()),
        expiration_equals_quote_date=bool((df["expiration"] == day).all()),
        root_values="|".join(sorted(df["root"].dropna().unique())),
        example_varying_contract=example,
    )


def oi_constancy_check():
    t0 = time.time()
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(B76_DIR, "02_0DTE_*.csv")))
    matched = [(n, B76_FILE_RE.match(n)) for n in names]
    files = [(n, m) for n, m in matched if m]
    skipped = [n for n, m in matched if not m]
    print(f"Cartella: {B76_DIR}")
    print(f"File trovati: {len(files)} (attesi {EXPECTED_DAYS}); "
          f"file 02_0DTE_* con nome non riconosciuto e ignorati: {len(skipped)}")
    for n in skipped:
        print("  ignorato:", n)

    rows = []
    for i, (name, m) in enumerate(files, 1):
        day = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        r = check_day(os.path.join(B76_DIR, name), day)
        rows.append(r)
        flag = "" if r["n_contracts_oi_varying"] == 0 else "   <-- OI VARIA"
        print(f"[{i:3d}/{len(files)}] {day}  contratti {r['n_contracts']:5d}  "
              f"OI variabile {r['n_contracts_oi_varying']:4d}{flag}")

    by_day = pd.DataFrame(rows)
    by_day.to_csv(OUTPUT_DIR / OUT_OI_BY_DAY, index=False)

    n_days = len(by_day)
    tot_var = int(by_day["n_contracts_oi_varying"].sum()) if n_days else 0
    days_var = int((by_day["n_contracts_oi_varying"] > 0).sum()) if n_days else 0
    roots = sorted(set("|".join(by_day["root_values"]).split("|"))) if n_days else []
    integrity = {
        "all_quote_dates_match_file": bool(by_day["quote_date_matches_file"].all()) if n_days else False,
        "all_expirations_equal_quote_date": bool(by_day["expiration_equals_quote_date"].all()) if n_days else False,
        "roots_found": "|".join(r for r in roots if r),
        "total_rows_oi_missing": int(by_day["n_rows_oi_missing"].sum()) if n_days else 0,
        "total_rows_oi_nonnumeric": int(by_day["n_rows_oi_nonnumeric"].sum()) if n_days else 0,
        "total_duplicate_rows": int(by_day["n_duplicate_rows"].sum()) if n_days else 0,
    }
    verified = (n_days == EXPECTED_DAYS) and (tot_var == 0)
    summary = dict(
        n_days_expected=EXPECTED_DAYS,
        n_days_checked=n_days,
        n_contract_days_checked=int(by_day["n_contracts"].sum()) if n_days else 0,
        n_rows_checked=int(by_day["n_rows"].sum()) if n_days else 0,
        n_days_with_varying_oi=days_var,
        n_contract_days_with_varying_oi=tot_var,
        **integrity,
        oi_constant_within_day_verified=verified,
        elapsed_minutes=round((time.time() - t0) / 60, 2),
    )
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / OUT_OI_SUMMARY, index=False)

    print("\n=== RIEPILOGO ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nVERDETTO:", "OI COSTANTE NELLA GIORNATA SU TUTTI I CONTRATTI E TUTTI I GIORNI"
          if verified else f"NON VERIFICATO: leggere {OUT_OI_BY_DAY}")
    if not (integrity["all_quote_dates_match_file"] and integrity["all_expirations_equal_quote_date"]
            and integrity["roots_found"] == "SPXW" and integrity["total_rows_oi_missing"] == 0
            and integrity["total_rows_oi_nonnumeric"] == 0 and integrity["total_duplicate_rows"] == 0):
        print("ATTENZIONE: almeno un controllo di integrita' non e' pulito; vedi il riepilogo.")
    print(f"Output in: {OUTPUT_DIR}")


# =====================================================================
# PARTE 2: IV, DELTA E GAMMA BLACK '76 CONTRO IL VENDOR, VICINO AL DENARO
# =====================================================================
# Errori assoluti, relativi e con segno, per fascia di |log-moneyness|. Gli errori relativi
# servono perche' il livello della IV varia di un fattore tre nel campione. Solo righe con
# black76_validation_usable_365 == 1 (OTM, IV convergente e non al limite).

OUT_VALIDATION_SUMMARY = "09_validation_ATM_summary.csv"
OUT_VALIDATION_POOLED = "09_validation_ATM_pooled.csv"

# Nei nomi dei file giornalieri la data (YYYY_MM_DD) segue subito il prefisso.
INPUT_FILE_PREFIX = "02_0DTE_"

# Fasce di |log-moneyness|: quella centrale e' la definizione principale di ATM, le altre
# due mostrano come varia l'accordo allontanandosi dal denaro.
MONEYNESS_BANDS = {
    "tight_0.01": 0.01,
    "atm_0.02": 0.02,
    "wide_0.05": 0.05,
}

HEADLINE_BAND = "atm_0.02"

# Il forward deve essere stimato da almeno questo numero di coppie put-call
MIN_F_PAIR_COUNT = 3

# Il gamma del vendor e' arrotondato a quattro decimali: con errore uniforme in +-5e-05 la
# discrepanza assoluta media dovuta al solo arrotondamento e' 2.5e-05, la soglia con cui
# confrontare il MAE del gamma.
VENDOR_GAMMA_ROUNDING = 1e-4
GAMMA_ROUNDING_FLOOR = VENDOR_GAMMA_ROUNDING / 4.0


def validation_atm():
    output_file = OUTPUT_DIR / OUT_VALIDATION_SUMMARY
    output_file_pooled = OUTPUT_DIR / OUT_VALIDATION_POOLED

    files = sorted(B76_DIR.glob(f"{INPUT_FILE_PREFIX}*_black76_365.csv"))

    if not files:
        raise FileNotFoundError(f"No files found in {B76_DIR}")

    print("=" * 78)
    print("ATM VALIDATION vs VENDOR")
    print("=" * 78)
    print(f"Input folder:  {B76_DIR}")
    print(f"Output:        {output_file.name}, {output_file_pooled.name}")
    print(f"Files:         {len(files)}")
    print(f"Bands:         {MONEYNESS_BANDS}")
    print(f"Gamma floor:   {GAMMA_ROUNDING_FLOOR:.2e} (vendor rounds to {VENDOR_GAMMA_ROUNDING})")
    print("=" * 78)

    required_columns = [
        "black76_validation_usable_365",
        "black76_log_moneyness_365",
        "black76_F_pair_count_365",
        "black76_implied_volatility_365",
        "implied_volatility",
        "black76_delta_365",
        "delta",
        "black76_gamma_365",
        "gamma",
    ]

    results = []
    pooled_chunks = []

    for index, f in enumerate(files, start=1):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)

        missing_columns = [c for c in required_columns if c not in df.columns]

        if missing_columns:
            print(f"  WARNING - {f.name}: missing {missing_columns}, skipped")
            results.append({
                "file": f.name,
                "status": "missing_columns",
                "missing_columns": ";".join(missing_columns),
            })
            continue

        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        base = df[
            (df["black76_validation_usable_365"] == 1)
            & (df["black76_F_pair_count_365"] >= MIN_F_PAIR_COUNT)
            & df["black76_log_moneyness_365"].notna()
        ].copy()

        row = {
            "file": f.name,
            "quote_date": f.name[len(INPUT_FILE_PREFIX):len(INPUT_FILE_PREFIX) + 10].replace("_", "-"),
            "status": "ok",
            "missing_columns": "",
        }

        for band_name, threshold in MONEYNESS_BANDS.items():
            sub = base[base["black76_log_moneyness_365"].abs() < threshold]

            n = len(sub)
            row[f"n_{band_name}"] = n

            if n == 0:
                for stat in ["mae_iv", "mre_iv", "bias_iv", "iv_level",
                             "mae_delta", "bias_delta",
                             "mae_gamma", "mre_gamma", "gamma_level"]:
                    row[f"{stat}_{band_name}"] = np.nan
                continue

            d_iv = sub["black76_implied_volatility_365"] - sub["implied_volatility"]
            d_de = sub["black76_delta_365"] - sub["delta"]
            d_ga = sub["black76_gamma_365"] - sub["gamma"]

            row[f"mae_iv_{band_name}"] = float(d_iv.abs().mean())
            row[f"mre_iv_{band_name}"] = float(
                (d_iv.abs() / sub["implied_volatility"]).median()
            )
            row[f"bias_iv_{band_name}"] = float(d_iv.mean())
            row[f"iv_level_{band_name}"] = float(sub["implied_volatility"].median())

            row[f"mae_delta_{band_name}"] = float(d_de.abs().mean())
            row[f"bias_delta_{band_name}"] = float(d_de.mean())

            row[f"mae_gamma_{band_name}"] = float(d_ga.abs().mean())
            row[f"mre_gamma_{band_name}"] = float(
                (d_ga.abs() / sub["gamma"]).median()
            )
            row[f"gamma_level_{band_name}"] = float(sub["gamma"].median())

            if band_name == HEADLINE_BAND:
                pooled_chunks.append(pd.DataFrame({
                    "quote_date": row["quote_date"],
                    "log_moneyness": sub["black76_log_moneyness_365"].to_numpy(),
                    "iv_own": sub["black76_implied_volatility_365"].to_numpy(),
                    "iv_vendor": sub["implied_volatility"].to_numpy(),
                    "delta_own": sub["black76_delta_365"].to_numpy(),
                    "delta_vendor": sub["delta"].to_numpy(),
                    "gamma_own": sub["black76_gamma_365"].to_numpy(),
                    "gamma_vendor": sub["gamma"].to_numpy(),
                }))

        results.append(row)

        h = HEADLINE_BAND
        print(
            f"[{index:3d}/{len(files)}] {row['quote_date']} | "
            f"n={row.get(f'n_{h}', 0):5,} | "
            f"MAE_IV={row.get(f'mae_iv_{h}', np.nan):.5f} | "
            f"rel={100 * row.get(f'mre_iv_{h}', np.nan):5.2f}% | "
            f"bias={row.get(f'bias_iv_{h}', np.nan):+.5f} | "
            f"MAE_g={row.get(f'mae_gamma_{h}', np.nan):.2e}",
            flush=True
        )

    out = pd.DataFrame(results)
    out.to_csv(output_file, index=False)

    pooled = pd.concat(pooled_chunks, ignore_index=True) if pooled_chunks else pd.DataFrame()

    if not pooled.empty:
        pooled.to_csv(output_file_pooled, index=False)

    valid = out[out["status"] == "ok"].copy()

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Files processed:  {len(out)}")
    print(f"Days with data:   {int((valid.get(f'n_{HEADLINE_BAND}', pd.Series(dtype=float)) > 0).sum())}")
    print(f"Skipped:          {int((out['status'] != 'ok').sum())}")

    print()
    print("PER BAND, unweighted mean of daily figures")
    print("-" * 78)
    print(f"{'band':<12} {'n tot':>10} {'MAE IV':>10} {'rel IV':>8} {'bias IV':>10} "
          f"{'IV lvl':>8} {'MAE gam':>10} {'rel gam':>8}")

    for band_name in MONEYNESS_BANDS:
        if f"n_{band_name}" not in valid.columns:
            continue
        ntot = int(valid[f"n_{band_name}"].sum())
        print(
            f"{band_name:<12} {ntot:>10,} "
            f"{valid[f'mae_iv_{band_name}'].mean():>10.5f} "
            f"{100 * valid[f'mre_iv_{band_name}'].mean():>7.2f}% "
            f"{valid[f'bias_iv_{band_name}'].mean():>+10.5f} "
            f"{valid[f'iv_level_{band_name}'].median():>8.4f} "
            f"{valid[f'mae_gamma_{band_name}'].mean():>10.2e} "
            f"{100 * valid[f'mre_gamma_{band_name}'].mean():>7.2f}%"
        )

    if not pooled.empty:
        d_iv = pooled["iv_own"] - pooled["iv_vendor"]
        d_de = pooled["delta_own"] - pooled["delta_vendor"]
        d_ga = pooled["gamma_own"] - pooled["gamma_vendor"]

        print()
        print(f"POOLED over all rows in band {HEADLINE_BAND} (n = {len(pooled):,})")
        print("-" * 78)
        print(f"IV     MAE {d_iv.abs().mean():.6f} | "
              f"median rel {100 * (d_iv.abs() / pooled['iv_vendor']).median():.2f}% | "
              f"p95 rel {100 * (d_iv.abs() / pooled['iv_vendor']).quantile(.95):.2f}% | "
              f"bias {d_iv.mean():+.6f} | corr {pooled['iv_own'].corr(pooled['iv_vendor']):.6f}")
        print(f"Delta  MAE {d_de.abs().mean():.6f} | "
              f"bias {d_de.mean():+.6f} | corr {pooled['delta_own'].corr(pooled['delta_vendor']):.6f}")
        print(f"Gamma  MAE {d_ga.abs().mean():.3e} | "
              f"median rel {100 * (d_ga.abs() / pooled['gamma_vendor']).median():.2f}% | "
              f"bias {d_ga.mean():+.3e} | corr {pooled['gamma_own'].corr(pooled['gamma_vendor']):.6f}")
        print()
        print(f"Gamma rounding floor: {GAMMA_ROUNDING_FLOOR:.3e}. "
              f"Measured MAE is {d_ga.abs().mean() / GAMMA_ROUNDING_FLOOR:.2f}x the floor.")
        print("If that ratio is near 1, the gamma discrepancy is quantisation and")
        print("nothing else. If it is well above 1, part of it is real disagreement.")

        print()
        print("Expected: agreement within 1 to 2 per cent on liquid ATM strikes.")
        rel = 100 * (d_iv.abs() / pooled["iv_vendor"])
        print(f"  share of ATM rows within 1%: {100 * (rel < 1).mean():.1f}%")
        print(f"  share within 2%:             {100 * (rel < 2).mean():.1f}%")
        print(f"  share within 5%:             {100 * (rel < 5).mean():.1f}%")

    print()
    print(f"Saved: {output_file}")
    if not pooled.empty:
        print(f"Saved: {output_file_pooled}")
    print("=" * 78)


# =====================================================================
# PARTE 3: PASSAGGIO COMPLETO, CONFRONTO CON IL VENDOR E NUMERI DELLA TESI
# =====================================================================
# Legge i 250 file una giornata alla volta (duckdb, solo le colonne necessarie) e il report
# annuale di 02_Black76_IV_greeks.py, e confronta i risultati con i numeri della tesi.
#
# Validation set: black76_validation_usable_365 == 1 (solver riuscito, OTM, IV convergente e
# non al limite, almeno 5 minuti alla scadenza, IV, delta e gamma del vendor validi), quote con
# bid = 0 comprese. Il flag viene ricostruito dalle componenti e deve coincidere con quello salvato.
#
# Le MAE della tesi sono la media non pesata delle 250 MAE giornaliere; si riporta anche la
# media pooled. Le fasce di |z| usano la volatilita' Black '76 dello strike stesso, perche'
# l'IV ATM interpolata non e' disponibile riga per riga.
#
# Le quote con ask nullo sono usate da 11_MFIV_diagnostics.py, le distanze fra strike vicino
# al forward da 10_Surface_descriptives.py.

REPORT_B76 = ("02_black76_iv_greeks", "02_black76_365_report_2025_ALL.csv")
SUMMARY_OI = ("09_appendix_a_checks", OUT_OI_SUMMARY)       # della parte 1

OUT_BY_DAY = "09_vendor_validation_by_day_2025.csv"
OUT_SUMMARY = "09_vendor_validation_summary_2025.csv"
OUT_BUCKET = "09_vendor_validation_by_bucket_2025.csv"
OUT_ZERO_ASK = "09_zero_ask_quotes_2025.csv"
OUT_SPACING = "09_strike_spacing_near_forward_2025.csv"
OUT_CHECK = "09_thesis_check.csv"

EARLY_CLOSES = set(EARLY_CLOSE_DATES)
GAMMA_CHECK_DAY = "2025-12-22"

MINUTES_PER_YEAR = 365 * 24 * 60
ROUNDING_TOL = 1e-6                   # tolleranza su gamma * 1e4 - round(.)
GAMMA_CP_TOL = 1e-12                   # soglia per "identici" (precisione macchina)
NEARLY_EVERY_SHARE = 0.95              # "nearly every session" = almeno il 95%

GRID_TIMES = [f"{h:02d}:{m:02d}:00" for h in range(9, 16) for m in range(0, 60, 10)
              if "09:40" <= f"{h:02d}:{m:02d}" <= "15:50"]
N_NEAR_STRIKES_PER_SIDE = 4

ABS_K_BINS = [0.0, 0.005, 0.01, 0.02, 0.05, np.inf]
ABS_K_LABELS = ["[0,0.005)", "[0.005,0.01)", "[0.01,0.02)", "[0.02,0.05)", ">=0.05"]
ABS_Z_BINS = [0.0, 0.5, 1.0, 2.0, 3.0, np.inf]
ABS_Z_LABELS = ["[0,0.5)", "[0.5,1)", "[1,2)", "[2,3)", ">=3"]

COLUMNS = {
    "quote_date": "VARCHAR",
    "quote_time": "VARCHAR",
    "expiration_time": "VARCHAR",
    "strike": "DOUBLE",
    "option_type": "VARCHAR",
    "bid": "DOUBLE",
    "ask": "DOUBLE",
    "implied_volatility": "DOUBLE",
    "delta": "DOUBLE",
    "gamma": "DOUBLE",
    "black76_r": "DOUBLE",
    "black76_T_365": "DOUBLE",
    "black76_F_365": "DOUBLE",
    "black76_log_moneyness_365": "DOUBLE",
    "black76_implied_volatility_365": "DOUBLE",
    "black76_gamma_365": "DOUBLE",
    "black76_is_otm_365": "DOUBLE",
    "black76_iv_converged_365": "DOUBLE",
    "black76_iv_at_bound_365": "DOUBLE",
    "black76_near_expiry_flag_365": "DOUBLE",
    "black76_solver_success_365": "DOUBLE",
    "black76_validation_usable_365": "DOUBLE",
}

# Valori come stampati nella tesi.
THESIS = {
    "tau_0931_regular_min": 389,
    "tau_0931_regular_years": 7.4e-4,
    "tau_0931_early_min": 209,
    "r_tau_bound": 3.4e-5,
    "n_rows": 48593820,
    "n_contract_days": 125458,
    "no_solution_share_pct": 9.7,
    "mae_iv": 0.0034,
    "mae_gamma": 7.1e-5,
    "corr_iv": 0.9999,
}


class CheckFailed(Exception):
    pass


def moments(x: np.ndarray, y: np.ndarray) -> dict:
    """Momenti centrati di una giornata, da combinare in modo stabile (Chan)."""
    n = len(x)
    if n == 0:
        return {"n": 0, "mx": 0.0, "my": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0}
    mx, my = float(x.mean()), float(y.mean())
    dx, dy = x - mx, y - my
    return {"n": n, "mx": mx, "my": my, "sxx": float(dx @ dx),
            "syy": float(dy @ dy), "sxy": float(dx @ dy)}


def combine(a: dict, b: dict) -> dict:
    if a["n"] == 0:
        return dict(b)
    if b["n"] == 0:
        return dict(a)
    n = a["n"] + b["n"]
    dx, dy = b["mx"] - a["mx"], b["my"] - a["my"]
    f = a["n"] * b["n"] / n
    return {"n": n,
            "mx": a["mx"] + dx * b["n"] / n,
            "my": a["my"] + dy * b["n"] / n,
            "sxx": a["sxx"] + b["sxx"] + dx * dx * f,
            "syy": a["syy"] + b["syy"] + dy * dy * f,
            "sxy": a["sxy"] + b["sxy"] + dx * dy * f}


def corr_of(m: dict) -> float:
    if m["n"] < 2 or m["sxx"] <= 0 or m["syy"] <= 0:
        return np.nan
    return m["sxy"] / np.sqrt(m["sxx"] * m["syy"])


def minutes_to_hhmm(t: str) -> str:
    return t[:5] if isinstance(t, str) else ""


def round_sig(x: float, sig: int) -> float:
    if not np.isfinite(x) or x == 0:
        return x
    return float(f"{x:.{sig}g}")


def load_day(con, f: Path) -> pd.DataFrame:
    types = ", ".join(f"'{c}': '{t}'" for c, t in COLUMNS.items())
    cols = ", ".join(COLUMNS)
    q = (f"SELECT {cols} FROM read_csv('{f.as_posix()}', header=true, "
         f"auto_detect=true, types={{{types}}})")
    return con.execute(q).df()


def vendor_validation_full_pass():
    print("=" * 140)
    print("PARTE 3: VENDOR VALIDATION FULL PASS: confronto Black '76 / vendor, tau, r*tau, "
          "gamma call/put, ask nulli")
    print("=" * 140)

    for p in (path(REPORT_B76), path(SUMMARY_OI)):
        require(p)

    files = []
    for f in sorted(B76_DIR.iterdir()):
        m = B76_FILE_RE.match(f.name)
        if m and m.group(1) == str(YEAR):
            files.append((f"{m.group(1)}-{m.group(2)}-{m.group(3)}", f))
    if len(files) != EXPECTED_DAYS:
        raise CheckFailed(f"{len(files)} file giornalieri in {B76_DIR}, attesi {EXPECTED_DAYS}")

    rep = pd.read_csv(path(REPORT_B76))
    rep["quote_date"] = rep["input_file"].str.extract(r"(\d{4})_(\d{2})_(\d{2})").agg("-".join, axis=1)
    rep = rep.set_index("quote_date")
    if len(rep) != EXPECTED_DAYS or set(rep.index) != {d for d, _ in files}:
        raise CheckFailed("il report dello stadio 02 non copre le stesse 250 giornate dei file")

    s19 = pd.read_csv(path(SUMMARY_OI)).iloc[0]       # riepilogo della parte 1

    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1GB'")

    # --- passaggio giorno per giorno

    day_rows, zero_ask_parts, spacing_rows = [], [], []
    bucket_acc = {}
    pool_all = {"n": 0, "mx": 0.0, "my": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0}
    pool_bid = dict(pool_all)
    sum_abs_iv = sum_abs_g = 0.0
    sum_abs_iv_b = sum_abs_g_b = 0.0
    gamma_cp_check_day = None
    t0 = time.time()

    for i, (qd, f) in enumerate(files, start=1):
        d = load_day(con, f)
        n_file = len(d)

        # --- integrita': righe e contatori come nel report dello stadio 02
        r = rep.loc[qd]
        if n_file != int(r["rows"]):
            raise CheckFailed(f"{qd}: {n_file} righe nel file, {int(r['rows'])} nel report dello stadio 02")
        n_bound = int((d["black76_iv_at_bound_365"] == 1).sum())
        n_noconv = int((d["black76_iv_converged_365"] == 0).sum())
        if n_bound != int(r["iv_at_bound_rows_365"]) or n_noconv != int(r["iv_not_converged_rows_365"]):
            raise CheckFailed(f"{qd}: righe al limite / non convergenti diverse dal report dello stadio 02")
        if not (d["quote_date"] == qd).all():
            raise CheckFailed(f"{qd}: quote_date non coincide con il nome del file")

        # --- validation set: ricostruito dalle componenti e confrontato con il flag dello stadio 02
        iv_v, de_v, g_v = d["implied_volatility"], d["delta"], d["gamma"]
        rebuilt = ((d["black76_solver_success_365"] == 1) & (d["black76_is_otm_365"] == 1)
                   & (d["black76_iv_converged_365"] == 1) & (d["black76_iv_at_bound_365"] == 0)
                   & (d["black76_near_expiry_flag_365"] == 0)
                   & np.isfinite(iv_v) & (iv_v > 0) & np.isfinite(de_v)
                   & np.isfinite(g_v) & (g_v > 0))
        flag = d["black76_validation_usable_365"] == 1
        if int((rebuilt != flag).sum()) != 0:
            raise CheckFailed(f"{qd}: il validation set ricostruito differisce dal flag dello stadio 02 "
                              f"su {int((rebuilt != flag).sum())} righe")
        v = d[flag]
        x = v["black76_implied_volatility_365"].to_numpy(float)
        y = v["implied_volatility"].to_numpy(float)
        gx = v["black76_gamma_365"].to_numpy(float)
        gy = v["gamma"].to_numpy(float)
        bidpos = (v["bid"].to_numpy(float) > 0)
        div, dg = x - y, gx - gy

        m_all = moments(x, y)
        m_bid = moments(x[bidpos], y[bidpos])
        pool_all = combine(pool_all, m_all)
        pool_bid = combine(pool_bid, m_bid)
        sum_abs_iv += float(np.abs(div).sum())
        sum_abs_g += float(np.abs(dg).sum())
        sum_abs_iv_b += float(np.abs(div[bidpos]).sum())
        sum_abs_g_b += float(np.abs(dg[bidpos]).sum())

        mae_iv = float(np.abs(div).mean()) if len(div) else np.nan
        mae_g = float(np.abs(dg).mean()) if len(dg) else np.nan
        if abs(mae_iv - float(r["mean_abs_iv_error_vs_cboe_365"])) > 1e-12 or \
           abs(mae_g - float(r["mean_abs_gamma_error_vs_cboe_365"])) > 1e-15:
            raise CheckFailed(f"{qd}: MAE giornaliero diverso da quello del report dello stadio 02")

        # --- fasce di |k| e di |z_own|
        absk = np.abs(v["black76_log_moneyness_365"].to_numpy(float))
        absz = absk / (x * np.sqrt(v["black76_T_365"].to_numpy(float)))
        for btype, vals, bins, labels in (("abs_k", absk, ABS_K_BINS, ABS_K_LABELS),
                                          ("abs_z_own", absz, ABS_Z_BINS, ABS_Z_LABELS)):
            cat = pd.cut(vals, bins=bins, labels=labels, right=False)
            tmp = pd.DataFrame({"b": cat, "aiv": np.abs(div), "ag": np.abs(dg),
                                "riv": np.abs(div) / y, "rg": np.abs(dg) / gy, "siv": div})
            agg = tmp.groupby("b", observed=False).agg(n=("aiv", "size"), aiv=("aiv", "sum"),
                                                       ag=("ag", "sum"), riv=("riv", "sum"),
                                                       rg=("rg", "sum"), siv=("siv", "sum"))
            for lab, row in agg.iterrows():
                key = (btype, str(lab))
                acc = bucket_acc.setdefault(key, np.zeros(6))
                acc += row[["n", "aiv", "ag", "riv", "rg", "siv"]].to_numpy(float)

        # --- gamma Black '76 di call e put allo stesso strike-minuto
        gc = d.loc[d["option_type"] == "C", ["quote_time", "strike", "black76_gamma_365"]]
        gp = d.loc[d["option_type"] == "P", ["quote_time", "strike", "black76_gamma_365"]]
        cp = gc.merge(gp, on=["quote_time", "strike"], suffixes=("_c", "_p"))
        both = cp.dropna(subset=["black76_gamma_365_c", "black76_gamma_365_p"])
        cp_diff = (both["black76_gamma_365_c"] - both["black76_gamma_365_p"]).abs()
        cp_max = float(cp_diff.max()) if len(cp_diff) else np.nan
        cp_n_above = int((cp_diff > GAMMA_CP_TOL).sum())
        if qd == GAMMA_CHECK_DAY:
            gamma_cp_check_day = {"pairs": len(cp), "pairs_both_finite": len(both),
                                  "max_absdiff": cp_max, "n_above_tol": cp_n_above}

        # --- arrotondamento del gamma del vendor
        gfin = d["gamma"].to_numpy(float)
        gfin = gfin[np.isfinite(gfin)]
        n_mult = int((np.abs(gfin / VENDOR_GAMMA_ROUNDING - np.round(gfin / VENDOR_GAMMA_ROUNDING)) < ROUNDING_TOL).sum())

        # --- orari, tau, r * tau
        qt = d["quote_time"].astype(str)
        first_q, last_q = qt.min(), qt.max()
        exp_times = sorted(d["expiration_time"].astype(str).unique())
        if len(exp_times) != 1:
            raise CheckFailed(f"{qd}: piu' di un expiration_time: {exp_times}")
        r_vals = d["black76_r"].dropna().unique()
        if len(r_vals) != 1:
            raise CheckFailed(f"{qd}: tasso non costante nella seduta: {r_vals[:5]}")
        tau_first = float(d.loc[qt == first_q, "black76_T_365"].max())
        max_r_tau = float((d["black76_r"] * d["black76_T_365"]).max())

        # --- ask nulli
        za = d[(d["ask"].isna()) | (d["ask"] <= 0)]
        if len(za):
            zero_ask_parts.append(za[["quote_date", "quote_time", "strike", "option_type", "bid",
                                      "ask", "black76_is_otm_365", "black76_F_365"]].copy())
        if len(za) != int(r["missing_price_365"]):
            raise CheckFailed(f"{qd}: {len(za)} quotazioni con ask nullo, "
                              f"{int(r['missing_price_365'])} prezzi mancanti nel report dello stadio 02")

        # --- distanza fra strike vicino al forward, ai punti di griglia
        g = d[qt.isin(GRID_TIMES)]
        for gt, gg in g.groupby("quote_time"):
            F = gg["black76_F_365"].dropna()
            if F.empty:
                continue
            F = float(F.median())
            ks = np.sort(gg["strike"].unique())
            below = ks[ks <= F][-N_NEAR_STRIKES_PER_SIDE:]
            above = ks[ks > F][:N_NEAR_STRIKES_PER_SIDE]
            near = np.concatenate([below, above])
            gaps = np.diff(near)
            spacing_rows.append({"quote_date": qd, "grid_time": gt, "forward": F,
                                 "n_near_strikes": len(near),
                                 "gap_min": float(gaps.min()) if len(gaps) else np.nan,
                                 "gap_max": float(gaps.max()) if len(gaps) else np.nan})

        day_rows.append({
            "quote_date": qd,
            "is_early_close": int(exp_times[0] == "13:00:00"),
            "n_rows_file": n_file,
            "n_contracts": int(d[["strike", "option_type"]].drop_duplicates().shape[0]),
            "n_iv_at_bound": n_bound,
            "n_iv_not_converged": n_noconv,
            "n_valid": len(v),
            "share_valid_bid_zero": float((~bidpos).mean()) if len(v) else np.nan,
            "mae_iv": mae_iv,
            "mae_gamma": mae_g,
            "mean_signed_iv_diff": float(div.mean()) if len(div) else np.nan,
            "median_signed_iv_diff": float(np.median(div)) if len(div) else np.nan,
            "corr_iv": corr_of(m_all),
            "n_valid_bid_pos": int(bidpos.sum()),
            "mae_iv_bid_pos": float(np.abs(div[bidpos]).mean()) if bidpos.any() else np.nan,
            "mae_gamma_bid_pos": float(np.abs(dg[bidpos]).mean()) if bidpos.any() else np.nan,
            "corr_iv_bid_pos": corr_of(m_bid),
            "gamma_cp_pairs": len(cp),
            "gamma_cp_pairs_both_finite": len(both),
            "gamma_cp_max_absdiff": cp_max,
            "gamma_cp_n_above_tol": cp_n_above,
            "n_vendor_gamma_finite": len(gfin),
            "n_vendor_gamma_multiple_1e4": n_mult,
            "first_quote": first_q,
            "last_quote": last_q,
            "expiration_time": exp_times[0],
            "tau_first_quote_years": tau_first,
            "tau_first_quote_minutes": tau_first * MINUTES_PER_YEAR,
            "r_d": float(r_vals[0]),
            "max_r_tau": max_r_tau,
            "n_zero_ask": len(za),
        })

        if i % 25 == 0 or i == len(files):
            print(f"  [{i:3d}/{len(files)}] {qd}  {time.time() - t0:6.0f} s", flush=True)
        del d, v, g

    con.close()
    by_day = pd.DataFrame(day_rows)

    # --- aggregati
    reg = by_day[by_day["is_early_close"] == 0]
    early = by_day[by_day["is_early_close"] == 1]
    early_dates = set(early["quote_date"])

    n_rows_total = int(by_day["n_rows_file"].sum())
    n_contract_days = int(by_day["n_contracts"].sum())
    share_bound = float(rep["iv_at_bound_rows_365"].sum() / rep["rows"].sum())
    share_noconv = float(rep["iv_not_converged_rows_365"].sum() / rep["rows"].sum())

    mae_iv_mdm = float(by_day["mae_iv"].mean())
    mae_g_mdm = float(by_day["mae_gamma"].mean())
    mae_iv_pool = sum_abs_iv / pool_all["n"]
    mae_g_pool = sum_abs_g / pool_all["n"]
    corr_pool = corr_of(pool_all)
    mae_iv_b_mdm = float(by_day["mae_iv_bid_pos"].mean())
    mae_g_b_mdm = float(by_day["mae_gamma_bid_pos"].mean())
    mae_iv_b_pool = sum_abs_iv_b / pool_bid["n"]
    mae_g_b_pool = sum_abs_g_b / pool_bid["n"]
    corr_b_pool = corr_of(pool_bid)
    share_bid_zero = 1.0 - pool_bid["n"] / pool_all["n"]
    n_neg_mean = int((by_day["mean_signed_iv_diff"] < 0).sum())
    n_neg_median = int((by_day["median_signed_iv_diff"] < 0).sum())
    share_mult = float(by_day["n_vendor_gamma_multiple_1e4"].sum() / by_day["n_vendor_gamma_finite"].sum())
    max_r_tau = float(by_day["max_r_tau"].max())
    day_max_r_tau = by_day.loc[by_day["max_r_tau"].idxmax(), "quote_date"]
    cp_max_all = float(by_day["gamma_cp_max_absdiff"].max())
    cp_n_above_all = int(by_day["gamma_cp_n_above_tol"].sum())

    bucket = []
    for (btype, lab), a in bucket_acc.items():
        n = a[0]
        bucket.append({"bucket_type": btype, "bucket": lab, "n_rows": int(n),
                       "share_rows": n / pool_all["n"],
                       "mae_iv": a[1] / n if n else np.nan,
                       "mae_gamma": a[2] / n if n else np.nan,
                       "mre_iv": a[3] / n if n else np.nan,
                       "mre_gamma": a[4] / n if n else np.nan,
                       "mean_signed_iv_diff": a[5] / n if n else np.nan})
    bucket = pd.DataFrame(bucket)
    order = {lab: j for j, lab in enumerate(ABS_K_LABELS + ABS_Z_LABELS)}
    bucket["_o"] = bucket["bucket"].map(order)
    bucket = bucket.sort_values(["bucket_type", "_o"]).drop(columns="_o")
    tot_rel_iv = float(bucket.loc[bucket["bucket_type"] == "abs_z_own", "mre_iv"].mul(
        bucket.loc[bucket["bucket_type"] == "abs_z_own", "n_rows"]).sum() / pool_all["n"])
    tot_rel_g = float(bucket.loc[bucket["bucket_type"] == "abs_z_own", "mre_gamma"].mul(
        bucket.loc[bucket["bucket_type"] == "abs_z_own", "n_rows"]).sum() / pool_all["n"])
    near = bucket[(bucket["bucket_type"] == "abs_z_own") & (bucket["bucket"] == "[0,0.5)")].iloc[0]

    zero_ask = (pd.concat(zero_ask_parts, ignore_index=True) if zero_ask_parts
                else pd.DataFrame(columns=["quote_date", "quote_time", "strike", "option_type",
                                           "bid", "ask", "black76_is_otm_365", "black76_F_365"]))
    zero_ask = zero_ask.rename(columns={"black76_is_otm_365": "is_otm", "black76_F_365": "forward"})
    zero_ask = zero_ask.sort_values(["quote_date", "quote_time", "strike", "option_type"]).reset_index(drop=True)
    za_minutes = zero_ask[["quote_date", "quote_time"]].drop_duplicates()
    za_otm = zero_ask[zero_ask["is_otm"] == 1]
    za_minutes_otm = za_otm[["quote_date", "quote_time"]].drop_duplicates()
    za_otm_days = "; ".join(f"{dd} {hhmm_min}-{hhmm_max} ({n} min)" for dd, hhmm_min, hhmm_max, n in
                            za_otm.groupby("quote_date")["quote_time"].agg(["min", "max", "nunique"])
                            .reset_index().itertuples(index=False))

    spacing = pd.DataFrame(spacing_rows)
    sp_reg = spacing[~spacing["quote_date"].isin(EARLY_CLOSES)]

    summary = [
        ("n_days", len(by_day), "giornate lette"),
        ("n_regular_days", len(reg), "sedute regolari (expiration_time 16:00)"),
        ("n_early_close_days", len(early), "chiusure anticipate (expiration_time 13:00)"),
        ("early_close_dates", ";".join(sorted(early_dates)), "date delle chiusure anticipate"),
        ("n_rows_total", n_rows_total, "righe dei 250 file (= rows del report di 02_black76_iv_greeks)"),
        ("n_rows_script19", int(s19["n_rows_checked"]), "righe secondo la parte 1"),
        ("n_contract_days", n_contract_days, "coppie (strike, tipo) distinte per giornata, sommate"),
        ("n_contract_days_script19", int(s19["n_contract_days_checked"]), "contratti-giorno secondo la parte 1"),
        ("share_iv_at_bound", share_bound, "somma iv_at_bound_rows_365 / somma rows (report di 02_black76_iv_greeks): 'no admissible solution'"),
        ("share_iv_not_converged", share_noconv, "somma iv_not_converged_rows_365 / somma rows (report di 02_black76_iv_greeks)"),
        ("n_valid", pool_all["n"], "righe del validation set (definizione nell'intestazione)"),
        ("share_valid_bid_zero", share_bid_zero, "quota del validation set con bid = 0"),
        ("mae_iv_mean_of_daily_means", mae_iv_mdm, "media non pesata delle 250 MAE IV giornaliere (cifra della tesi)"),
        ("mae_gamma_mean_of_daily_means", mae_g_mdm, "media non pesata delle 250 MAE gamma giornaliere (cifra della tesi)"),
        ("mae_iv_pooled", mae_iv_pool, "MAE IV su tutte le righe del validation set"),
        ("mae_gamma_pooled", mae_g_pool, "MAE gamma su tutte le righe del validation set"),
        ("corr_iv_pooled", corr_pool, "correlazione di Pearson IV Black '76 / vendor, tutte le righe"),
        ("corr_iv_daily_min", float(by_day["corr_iv"].min()), "correlazione giornaliera minima"),
        ("corr_iv_daily_median", float(by_day["corr_iv"].median()), "correlazione giornaliera mediana"),
        ("n_valid_bid_pos", pool_bid["n"], "righe del validation set con bid > 0"),
        ("mae_iv_bid_pos_mean_of_daily_means", mae_iv_b_mdm, "sottoinsieme bid > 0"),
        ("mae_gamma_bid_pos_mean_of_daily_means", mae_g_b_mdm, "sottoinsieme bid > 0"),
        ("mae_iv_bid_pos_pooled", mae_iv_b_pool, "sottoinsieme bid > 0"),
        ("mae_gamma_bid_pos_pooled", mae_g_b_pool, "sottoinsieme bid > 0"),
        ("corr_iv_bid_pos_pooled", corr_b_pool, "sottoinsieme bid > 0"),
        ("n_days_mean_signed_iv_diff_negative", n_neg_mean, "sedute con media di (IV propria - IV vendor) < 0"),
        ("n_days_median_signed_iv_diff_negative", n_neg_median, "sedute con mediana di (IV propria - IV vendor) < 0"),
        ("mre_iv_pooled", tot_rel_iv, "media di |dIV| / IV vendor, tutte le righe"),
        ("mre_gamma_pooled", tot_rel_g, "media di |dGamma| / gamma vendor, tutte le righe"),
        ("gamma_rounding_floor", GAMMA_ROUNDING_FLOOR, "errore atteso dal solo arrotondamento a 1e-4 (parte 2)"),
        ("share_vendor_gamma_multiple_1e4", share_mult, "quota dei gamma del vendor (finiti, tutte le righe) multipli di 1e-4"),
        ("gamma_cp_pairs_2025_12_22", gamma_cp_check_day["pairs"], "coppie call/put allo stesso strike-minuto, 22 dicembre"),
        ("gamma_cp_pairs_both_finite_2025_12_22", gamma_cp_check_day["pairs_both_finite"], "di cui con entrambi i gamma definiti"),
        ("gamma_cp_max_absdiff_2025_12_22", gamma_cp_check_day["max_absdiff"], "max |Gamma_C - Gamma_P|, 22 dicembre"),
        ("gamma_cp_n_above_tol_2025_12_22", gamma_cp_check_day["n_above_tol"], f"coppie con differenza > {GAMMA_CP_TOL}"),
        ("gamma_cp_max_absdiff_all_days", cp_max_all, "max |Gamma_C - Gamma_P| su tutte le giornate"),
        ("gamma_cp_n_above_tol_all_days", cp_n_above_all, f"coppie con differenza > {GAMMA_CP_TOL}, tutte le giornate"),
        ("first_quote_all_days", ";".join(sorted(by_day["first_quote"].unique())), "prima quotazione"),
        ("last_quote_regular", ";".join(sorted(reg["last_quote"].unique())), "ultima quotazione, sedute regolari"),
        ("last_quote_early", ";".join(sorted(early["last_quote"].unique())), "ultima quotazione, chiusure anticipate"),
        ("tau_0931_regular_minutes_min", float(reg["tau_first_quote_minutes"].min()), "tau alla prima quotazione, regolari"),
        ("tau_0931_regular_minutes_max", float(reg["tau_first_quote_minutes"].max()), "tau alla prima quotazione, regolari"),
        ("tau_0931_regular_years", float(reg["tau_first_quote_years"].max()), "tau alla prima quotazione, regolari, anni"),
        ("tau_0931_early_minutes_min", float(early["tau_first_quote_minutes"].min()), "tau alla prima quotazione, chiusure anticipate"),
        ("tau_0931_early_minutes_max", float(early["tau_first_quote_minutes"].max()), "tau alla prima quotazione, chiusure anticipate"),
        ("max_r_tau", max_r_tau, f"max di black76_r * black76_T_365 su tutte le righe ({day_max_r_tau})"),
        ("n_zero_ask_quotes", len(zero_ask), "quotazioni con ask = 0 o mancante"),
        ("n_zero_ask_quotes_otm", int((zero_ask["is_otm"] == 1).sum()), "di cui OTM"),
        ("n_zero_ask_minutes", len(za_minutes), "minuti con almeno un ask nullo"),
        ("n_zero_ask_minutes_otm", len(za_minutes_otm), "minuti con almeno un ask nullo su un contratto OTM"),
        ("zero_ask_otm_minutes_by_day", za_otm_days, "giornate e intervalli dei minuti con ask nullo OTM"),
        ("n_missing_price_report04", int(rep["missing_price_365"].sum()), "prezzi mancanti nel report di 02_black76_iv_greeks"),
        ("spacing_grid_points_regular", len(sp_reg), "punti di griglia (giornata x minuto), sedute regolari"),
        ("spacing_share_all_gaps_5", float(((sp_reg["gap_min"] == 5) & (sp_reg["gap_max"] == 5)).mean()),
         "quota con tutte le 7 distanze fra gli 8 strike piu' vicini al forward pari a 5 punti"),
    ]
    summary = pd.DataFrame(summary, columns=["voce", "valore", "definizione"])

    # --- controllo contro la tesi
    check = []

    def add(pagina, voce, tesi, calc, precisione, ok, nota=""):
        check.append({"pagina": pagina, "voce": voce, "valore_tesi": tesi,
                      "valore_calcolato": calc, "precisione": precisione,
                      "esito": "OK" if ok else "DIVERSO", "nota": nota})

    tau_reg_min = sorted(set(np.round(reg["tau_first_quote_minutes"], 6)))
    tau_early_min = sorted(set(np.round(early["tau_first_quote_minutes"], 6)))
    add(8, "sessioni del 2025", 250, len(by_day), "intero", len(by_day) == 250)
    add(8, "sedute regolari", 247, len(reg), "intero", len(reg) == 247)
    add(8, "chiusure anticipate (3 lug, 28 nov, 24 dic)", "2025-07-03;2025-11-28;2025-12-24",
        ";".join(sorted(early_dates)), "date", early_dates == EARLY_CLOSES)
    add(8, "prima quotazione", "09:31", ";".join(sorted({minutes_to_hhmm(t) for t in by_day["first_quote"]})),
        "minuto", set(by_day["first_quote"]) == {"09:31:00"})
    add(8, "ultimo timestamp: 16:00 regolari, 13:00 chiusure anticipate", "16:00/13:00",
        f"{'/'.join(sorted({minutes_to_hhmm(t) for t in reg['last_quote']}))}/"
        f"{'/'.join(sorted({minutes_to_hhmm(t) for t in early['last_quote']}))}",
        "minuto", set(reg["last_quote"]) == {"16:00:00"} and set(early["last_quote"]) == {"13:00:00"})
    add(8, "tau alle 09:31, sedute regolari (minuti)", THESIS["tau_0931_regular_min"],
        ";".join(f"{t:g}" for t in tau_reg_min), "intero", tau_reg_min == [389.0])
    add(8, "tau alle 09:31, sedute regolari (anni)", THESIS["tau_0931_regular_years"],
        float(reg["tau_first_quote_years"].max()), "2 cifre significative",
        round_sig(float(reg["tau_first_quote_years"].max()), 2) == THESIS["tau_0931_regular_years"])
    add(8, "tau alle 09:31, chiusure anticipate (minuti)", THESIS["tau_0931_early_min"],
        ";".join(f"{t:g}" for t in tau_early_min), "intero", tau_early_min == [209.0])
    add(9, "r_d * tau non supera mai", THESIS["r_tau_bound"], max_r_tau, "limite superiore",
        max_r_tau <= THESIS["r_tau_bound"],
        f"massimo {max_r_tau:.3e} il {day_max_r_tau}: il limite vale; il valore stretto e' 3.3e-5")
    add(66, "righe dei 250 file di opzioni", THESIS["n_rows"], n_rows_total, "intero",
        n_rows_total == THESIS["n_rows"] == int(s19["n_rows_checked"]),
        f"parte 1: {int(s19['n_rows_checked']):,}")
    add(66, "contratti-giorno", THESIS["n_contract_days"], n_contract_days, "intero",
        n_contract_days == THESIS["n_contract_days"] == int(s19["n_contract_days_checked"]),
        f"parte 1: {int(s19['n_contract_days_checked']):,}")
    add(66, "quota di contratti-minuto senza soluzione ammissibile (%)", THESIS["no_solution_share_pct"],
        100 * share_bound, "1 decimale", round(100 * share_bound, 1) == THESIS["no_solution_share_pct"],
        f"righe con IV al limite del bracket; le non convergenti sono {100 * share_noconv:.2f}%")
    add(66, "22 dicembre: gamma Black '76 di call e put identici a ogni strike", "identici",
        gamma_cp_check_day["max_absdiff"], f"max |diff| <= {GAMMA_CP_TOL}",
        gamma_cp_check_day["n_above_tol"] == 0 and gamma_cp_check_day["max_absdiff"] <= GAMMA_CP_TOL,
        f"{gamma_cp_check_day['pairs_both_finite']:,} coppie strike-minuto; su tutte le giornate max {cp_max_all:.1e}")
    add(66, "differenza assoluta media dal vendor, IV", THESIS["mae_iv"], mae_iv_mdm, "4 decimali",
        round(mae_iv_mdm, 4) == THESIS["mae_iv"],
        f"media delle medie giornaliere; pooled {mae_iv_pool:.5f} (arrotonda a {round(mae_iv_pool, 4)})")
    add(66, "differenza assoluta media dal vendor, gamma", THESIS["mae_gamma"], mae_g_mdm, "2 cifre significative",
        round_sig(mae_g_mdm, 2) == THESIS["mae_gamma"],
        f"media delle medie giornaliere; pooled {mae_g_pool:.3e} (arrotonda a {round_sig(mae_g_pool, 2):.1e})")
    add(66, "correlazione IV Black '76 / vendor", THESIS["corr_iv"], corr_pool, "4 decimali",
        round(corr_pool, 4) == THESIS["corr_iv"], f"pooled su {pool_all['n']:,} righe; bid>0: {corr_b_pool:.5f}")
    add(66, "agreement closer near ATM: IV (MAE, |z_own|<0.5 contro tutte le righe)", "piu' vicino",
        f"{near['mae_iv']:.4f} contro {mae_iv_pool:.4f}", "confronto",
        near["mae_iv"] < mae_iv_pool,
        f"per IV l'accordo NON e' migliore vicino al forward (vedi {OUT_BUCKET})")
    add(66, "agreement closer near ATM: gamma (errore relativo, |z_own|<0.5 contro tutte le righe)", "piu' vicino",
        f"{near['mre_gamma']:.4f} contro {tot_rel_g:.4f}", "confronto", near["mre_gamma"] < tot_rel_g)
    add(66, "gamma del vendor arrotondato a quattro decimali", "4 decimali",
        share_mult, "quota multipli di 1e-4", share_mult == 1.0,
        f"{int(by_day['n_vendor_gamma_finite'].sum()):,} valori finiti")
    add(66, "l'arrotondamento non spiega tutte le differenze di gamma", "MAE > soglia",
        f"{mae_g_mdm:.2e} contro {GAMMA_ROUNDING_FLOOR:.1e}", "confronto", mae_g_mdm > GAMMA_ROUNDING_FLOOR)
    add(66, "IV ricostruita leggermente piu' bassa in quasi ogni seduta", "quasi ogni seduta",
        f"{n_neg_mean}/{len(by_day)}", f">= {NEARLY_EVERY_SHARE:.0%} delle sedute",
        n_neg_mean >= NEARLY_EVERY_SHARE * len(by_day),
        f"media di (propria - vendor) < 0 in {n_neg_mean} sedute, mediana < 0 in {n_neg_median}")
    check = pd.DataFrame(check)

    # --- scrittura
    by_day.to_csv(OUTPUT_DIR / OUT_BY_DAY, index=False)
    summary.to_csv(OUTPUT_DIR / OUT_SUMMARY, index=False)
    bucket.to_csv(OUTPUT_DIR / OUT_BUCKET, index=False)
    zero_ask.to_csv(OUTPUT_DIR / OUT_ZERO_ASK, index=False)
    spacing.to_csv(OUTPUT_DIR / OUT_SPACING, index=False)
    check.to_csv(OUTPUT_DIR / OUT_CHECK, index=False)

    # --- stampa (le opzioni di visualizzazione valgono solo qui)
    with pd.option_context("display.width", 200, "display.max_colwidth", 90):
        print()
        print("RIEPILOGO")
        print(summary.to_string(index=False))
        print()
        print("FASCE (validation set)")
        print(bucket.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
        print()
        print("CONTROLLO CONTRO LA TESI")
        print(check[["pagina", "voce", "valore_tesi", "valore_calcolato", "esito"]].to_string(index=False))
        n_ok = int((check["esito"] == "OK").sum())
        n_diff = int((check["esito"] == "DIVERSO").sum())
        for _, r in check[check["esito"] == "DIVERSO"].iterrows():
            print(f"  DIVERSO p. {r['pagina']}: {r['voce']}: tesi {r['valore_tesi']}, "
                  f"calcolato {r['valore_calcolato']}. {r['nota']}")
        print()
        print(f"Output in {OUTPUT_DIR}")
        print(f"Tempo: {(time.time() - t0) / 60:.1f} minuti")
        print(f"VERDETTO: {n_ok} OK, {n_diff} DIVERSO su {len(check)} voci della tesi; "
              f"integrita' dei dati verificata (righe, contatori, validation set, MAE e ask nulli "
              f"coincidono con il report dello stadio 02).")
        print("=" * 140)


def main():
    oi_constancy_check()
    validation_atm()
    vendor_validation_full_pass()


if __name__ == "__main__":
    main()
