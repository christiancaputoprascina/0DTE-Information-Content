"""
Descrittive della superficie di volatilita' implicita 0DTE e del gamma, e numeri del Capitolo 2.
  Parte 1: superficie (IV_ATM, skew, curvatura), quote di gamma per fascia di z, liquidita' OTM.
  Parte 2: profilo per minuto di H_OI e H_VOL, con il gamma del vendor e quello Black '76.
  Parte 3: gamma del vendor contro gamma Black '76 per fascia di z, minuto e lato.
  Parte 4: superficie della IV su una griglia fissa di z.
  Parte 5: Tabelle 2.1 e 2.2 e numeri del Capitolo 2, confrontati con i valori della tesi.
Input:  CBOE/Data/02_black76_iv_greeks/, 03_mfiv/, 04_gamma_herfindahl/, 07_daily_panel/,
        09_appendix_a_checks/
Output: CBOE/Data/10_surface_descriptives/
"""

import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR, EARLY_CLOSE_DATES, STRESS_DAY, YEAR, output_dir, path, require

OUTPUT_DIR = output_dir("10_surface_descriptives")
OUTPUT_FILE_PREFIX = "10_"

# File per contratto dello stadio 02 (02_Black76_IV_greeks.py).
B76_DIR = DATA_DIR / "02_black76_iv_greeks"
FILE_RE = re.compile(r"^02_0DTE_(\d{4})_(\d{2})_(\d{2})_black76_365\.csv$")
DAILY_PANEL = ("07_daily_panel", "07_daily_regression_panel_2025.csv")

EARLY_CLOSES = set(EARLY_CLOSE_DATES)
EXPECTED_FILES = 250
EXPECTED_REGULAR = 247

# 38 punti di griglia, 09:40-15:50 ogni 10 minuti (parti 1, 3 e 4).
GRID_TIMES = [f"{h:02d}:{m:02d}:00" for h in range(9, 16) for m in range(0, 60, 10)
              if (h, m) >= (9, 40) and (h, m) <= (15, 50)]
assert len(GRID_TIMES) == 38
SMILE_TIMES = ["09:40:00", "12:00:00", "15:00:00"]     # minuti dei sorrisi (Fig. 2.1)

# Fasce di z (chiuse a destra), parti 1 e 3.
BUCKET_EDGES = [-np.inf, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, np.inf]
BUCKET_LABELS = ["(-inf,-3]", "(-3,-2]", "(-2,-1]", "(-1,-0.5]", "(-0.5,0]",
                 "(0,0.5]", "(0.5,1]", "(1,2]", "(2,3]", "(3,inf)"]

K_SIGN_TOL = 1e-9       # |colonna - log(F/K)|, parti 1, 3 e 4


# =====================================================================
# FUNZIONI COMUNI A PIU' PARTI
# =====================================================================

class CheckFailed(Exception):
    pass


def to_float_exact(series: pd.Series) -> pd.Series:
    # La stringa vuota e' il valore mancante scritto da pandas; float() ricostruisce
    # esattamente i numeri scritti da pandas.
    s = series.astype(str).str.strip()
    s = s.where(s != "", "nan")
    return s.astype(float)


def rel_diff(a: float, b: float) -> float:
    if not np.isfinite(a) and not np.isfinite(b):
        return 0.0
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.inf
    return abs(a - b) / max(abs(b), 1e-300)


def q(s, p):
    s = s.dropna()
    return float(s.quantile(p)) if len(s) else np.nan


class Log:
    """Registro dei filtri: righe prima, tolte, dopo (stessi nomi dei passi nelle parti 1 e 4)."""
    def __init__(self):
        self.rows = []

    def step(self, day, branch, name, before_df, after_df):
        nb, na = len(before_df), len(after_df)
        self.rows.append((day, branch, name, nb, nb - na, na))
        return after_df


# =====================================================================
# PARTE 1: SUPERFICIE, GAMMA PER MONEYNESS, LIQUIDITA'
# =====================================================================
# Legge una volta ciascun file per contratto (247 sessioni regolari, 38 punti di griglia) e
# calcola tre rami: A superficie della IV (IV_ATM, skew, curvatura, una deviazione standard in
# punti), B quote di gamma x OI e gamma x volume per fascia di z, C liquidita' OTM.
#
# La colonna black76_log_moneyness_365 contiene log(F/K): si usa k = log(K/F) = -colonna,
# verificando su ogni giornata che la colonna coincida con log(F/K).
#
# IV_ATM interpola in k fra il put e la call OTM piu' vicini a F; z = k / (IV_ATM * sqrt(tau)).
# Le IV a z = -1 e +1 interpolano fra due strike dello stesso lato, senza estrapolazione.
# Il ramo B usa le righe di 04_Gamma_Herfindahl.py e ne deve riprodurre il Herfindahl.

GAMMA_FILES_H = {
    "CBOE_gamma": ("04_gamma_herfindahl", f"04_gamma_cboe_intraday_{YEAR}_all.csv",
                   "h_oi_cboe", "h_vol_cboe", "gamma"),
    "Black76_gamma": ("04_gamma_herfindahl", f"04_gamma_intraday_{YEAR}_all.csv",
                      "h_oi", "h_vol", "black76_gamma_365"),
}

RANK_MEDIAN = 124       # nearest-rank su 247: ceil(0.5 * 247)
RANK_P90 = 223          # nearest-rank su 247: ceil(0.9 * 247)

H_REL_TOL = 1e-9
SPREAD_REL_TOL = 1e-9

USECOLS_SURFACE = [
    "quote_date", "quote_time", "strike", "option_type", "bid", "ask",
    "trade_volume", "open_interest", "gamma", "black76_r",
    "black76_spread_pct_of_mid", "black76_T_365", "black76_F_365",
    "black76_log_moneyness_365", "black76_implied_volatility_365",
    "black76_is_otm_365", "black76_iv_at_bound_365", "black76_iv_converged_365",
    "black76_gamma_365", "black76_near_expiry_flag_365",
    "black76_solver_success_365",
]
NUMERIC_SURFACE = [c for c in USECOLS_SURFACE if c not in ("quote_date", "quote_time", "option_type")]


def bucket_of(z: pd.Series) -> pd.Series:
    return pd.cut(z, bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=True).astype(object)


def herfindahl(exposure: pd.Series) -> float:
    x = exposure[np.isfinite(exposure) & (exposure > 0)]
    if x.empty:
        return np.nan
    w = x / x.sum()
    return float(np.sum(w ** 2))


def rel_diff_surface(a: float, b: float) -> float:
    # come rel_diff, ma accetta anche None
    if (a is None or not np.isfinite(a)) and (b is None or not np.isfinite(b)):
        return 0.0
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.inf
    return abs(a - b) / max(abs(b), 1e-300)


def read_day_surface(path: Path, day: str, log: Log) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=USECOLS_SURFACE, dtype=str, keep_default_na=False)
    missing = [c for c in USECOLS_SURFACE if c not in raw.columns]
    if missing:
        raise CheckFailed(f"{path.name}: colonne mancanti {missing}")
    dates = raw["quote_date"].unique()
    if len(dates) != 1 or dates[0] != day:
        raise CheckFailed(f"{path.name}: quote_date {dates[:3]} diverso da {day}")
    log.rows.append((day, "lettura", "righe nel file", len(raw), 0, len(raw)))
    df = log.step(day, "lettura", "fuori dai 38 punti di griglia", raw,
                  raw[raw["quote_time"].isin(GRID_TIMES)].copy())
    present = set(df["quote_time"].unique())
    if present != set(GRID_TIMES):
        raise CheckFailed(f"{day}: punti di griglia assenti nel file: {sorted(set(GRID_TIMES) - present)}")
    for c in NUMERIC_SURFACE:
        df[c] = pd.to_numeric(df[c], errors="coerce")    # come la pipeline del gamma
    ln_f_over_k = np.log(df["black76_F_365"] / df["strike"])
    col = df["black76_log_moneyness_365"]
    both = np.isfinite(ln_f_over_k) & np.isfinite(col)
    dev = float((col[both] - ln_f_over_k[both]).abs().max()) if both.any() else np.inf
    if not dev <= K_SIGN_TOL:
        dev_other = float((col[both] + ln_f_over_k[both]).abs().max()) if both.any() else np.inf
        raise CheckFailed(f"{day}: black76_log_moneyness_365 non e' log(F/K) (scarto {dev:.3e}; "
                          f"scarto da log(K/F) {dev_other:.3e})")
    df["_k"] = -col        # k = log(K/F)
    ot = df["option_type"].astype(str).str.strip().str.upper()
    df["_type"] = np.where(ot.isin(["C", "CALL"]), "C", np.where(ot.isin(["P", "PUT"]), "P", ""))
    return df


def interp_side(z: np.ndarray, iv: np.ndarray, target: float):
    """Interpolazione lineare in z fra due strike dello stesso lato che racchiudono target."""
    if z.size == 0:
        return np.nan, "no_strikes"
    order = np.argsort(z)
    z, iv = z[order], iv[order]
    if target < 0:          # lato put: z < 0
        if z.max() < target:
            return np.nan, "inside_gap"      # anche il put piu' vicino e' oltre -1
        if z.min() > target:
            return np.nan, "beyond_range"    # -1 oltre il put piu' lontano utilizzabile
    else:                   # lato call: z > 0
        if z.min() > target:
            return np.nan, "inside_gap"
        if z.max() < target:
            return np.nan, "beyond_range"
    return float(np.interp(target, z, iv)), "ok"


def surface_for_day(df: pd.DataFrame, day: str, log: Log):
    a = df
    a = log.step(day, "A", "black76_is_otm_365 != 1", a, a[a["black76_is_otm_365"] == 1])
    a = log.step(day, "A", "black76_iv_converged_365 != 1", a, a[a["black76_iv_converged_365"] == 1])
    a = log.step(day, "A", "black76_iv_at_bound_365 != 0", a, a[a["black76_iv_at_bound_365"] == 0])
    a = log.step(day, "A", "black76_solver_success_365 != 1", a, a[a["black76_solver_success_365"] == 1])
    a = log.step(day, "A", "black76_near_expiry_flag_365 != 0", a, a[a["black76_near_expiry_flag_365"] == 0])
    a = log.step(day, "A", "bid <= 0 o mancante", a, a[a["bid"] > 0])
    iv, k = a["black76_implied_volatility_365"], a["_k"]
    a = log.step(day, "A", "IV o k non finiti, o IV <= 0", a,
                 a[np.isfinite(iv) & np.isfinite(k) & (iv > 0)])
    a = log.step(day, "A", "k = 0 esatto (ne' put ne' call)", a, a[a["_k"] != 0])

    wrong_side = int((((a["_k"] < 0) & (a["_type"] != "P")) |
                      ((a["_k"] > 0) & (a["_type"] != "C"))).sum())

    rows, smile_ready = [], {}
    for t in GRID_TIMES:
        allt = df[df["quote_time"] == t]
        tau = float(np.nanmedian(allt["black76_T_365"])) if allt["black76_T_365"].notna().any() else np.nan
        F = float(np.nanmedian(allt["black76_F_365"])) if allt["black76_F_365"].notna().any() else np.nan
        u = a[a["quote_time"] == t]
        puts = u[u["_k"] < 0]
        calls = u[u["_k"] > 0]
        rec = {"quote_date": day, "grid_time": t, "tau": tau, "F": F,
               "n_otm_put_usable": len(puts), "n_otm_call_usable": len(calls)}
        iv_atm = k_p = k_c = np.nan
        if len(puts) and len(calls):
            ip = puts["_k"].idxmax()
            ic = calls["_k"].idxmin()
            k_p, v_p = puts.at[ip, "_k"], puts.at[ip, "black76_implied_volatility_365"]
            k_c, v_c = calls.at[ic, "_k"], calls.at[ic, "black76_implied_volatility_365"]
            iv_atm = v_p + (0.0 - k_p) * (v_c - v_p) / (k_c - k_p)
        sd = iv_atm * math.sqrt(tau) if np.isfinite(iv_atm) and np.isfinite(tau) and tau > 0 else np.nan
        rec.update(iv_atm=iv_atm, k_atm_put=k_p, k_atm_call=k_c, sd_k=sd,
                   sd_points=sd * F if np.isfinite(sd) and np.isfinite(F) else np.nan)
        if np.isfinite(sd):
            zp = puts["_k"].to_numpy() / sd
            zc = calls["_k"].to_numpy() / sd
            ivm1, st_m1 = interp_side(zp, puts["black76_implied_volatility_365"].to_numpy(), -1.0)
            ivp1, st_p1 = interp_side(zc, calls["black76_implied_volatility_365"].to_numpy(), 1.0)
        else:
            ivm1, st_m1, ivp1, st_p1 = np.nan, "no_atm", np.nan, "no_atm"
        rec.update(iv_zm1=ivm1, iv_zp1=ivp1, zm1_status=st_m1, zp1_status=st_p1,
                   skew=ivm1 - ivp1, curvature=ivm1 + ivp1 - 2.0 * iv_atm)
        rows.append(rec)
        smile_ready[t] = (u, iv_atm, sd)
    return pd.DataFrame(rows), smile_ready, wrong_side


def gamma_universe(df: pd.DataFrame, gamma_col: str) -> pd.Series:
    solver = df["black76_solver_success_365"].fillna(0).astype(int)
    near = df["black76_near_expiry_flag_365"].fillna(0).astype(int)
    bound = df["black76_iv_at_bound_365"].fillna(0).astype(int)
    g, K, tau, r = df[gamma_col], df["strike"], df["black76_T_365"], df["black76_r"]
    return ((solver == 1) & (near == 0) & (bound != 1)
            & np.isfinite(K) & (K > 0) & np.isfinite(tau) & (tau > 0) & np.isfinite(r)
            & df["_type"].isin(["C", "P"]) & np.isfinite(g) & (g > 0))


def gamma_for_day(df, day, surf, gamma_ref, log):
    sd_by_t = surf.set_index("grid_time")["sd_k"]
    share_rows, check_rows = [], []
    for source, (_, _, h_oi_col, h_vol_col, gcol) in GAMMA_FILES_H.items():
        mask = gamma_universe(df, gcol)
        uni = log.step(day, f"B {source}", "fuori dall'universo delle pipeline del gamma", df, df[mask])
        oi = log.step(day, f"B {source}", "OI <= 0 (quote gamma x OI)", uni,
                      uni[np.isfinite(uni["open_interest"]) & (uni["open_interest"] > 0)])
        vol = log.step(day, f"B {source}", "volume <= 0 (quote gamma x volume)", uni,
                       uni[np.isfinite(uni["trade_volume"]) & (uni["trade_volume"] > 0)])
        for t in GRID_TIMES:
            o = oi[oi["quote_time"] == t]
            v = vol[vol["quote_time"] == t]
            eo = o[gcol] * o["open_interest"]
            ev = v[gcol] * v["trade_volume"]
            h_oi = herfindahl(eo.groupby(o["strike"]).sum())
            h_vol = herfindahl(ev.groupby(v["strike"]).sum())
            ref = gamma_ref[source].get((day, t), (np.nan, np.nan, ""))
            check_rows.append({"quote_date": day, "grid_time": t, "gamma_source": source,
                               "h_oi_repro": h_oi, "h_oi_file": ref[0],
                               "reldiff_h_oi": rel_diff_surface(h_oi, ref[0]),
                               "h_vol_repro": h_vol, "h_vol_file": ref[1],
                               "reldiff_h_vol": rel_diff_surface(h_vol, ref[1]),
                               "gamma_flag_file": ref[2]})
            sd = sd_by_t.get(t, np.nan)
            z_o = o["_k"] / sd
            z_v = v["_k"] / sd
            b_o = bucket_of(z_o).fillna("undefined")
            b_v = bucket_of(z_v).fillna("undefined")
            so = eo.groupby(b_o).sum() / eo.sum() if eo.sum() > 0 else pd.Series(dtype=float)
            sv = ev.groupby(b_v).sum() / ev.sum() if ev.sum() > 0 else pd.Series(dtype=float)
            no = o["strike"].groupby(b_o).nunique()
            nv = v["strike"].groupby(b_v).nunique()
            sd_ok = bool(np.isfinite(sd))
            for b in BUCKET_LABELS + ["undefined"]:
                if b == "undefined" and b not in so.index and b not in sv.index:
                    continue
                # senza IV_ATM al minuto le fasce non esistono: quote mancanti, non zero
                blank = (b != "undefined") and not sd_ok
                share_rows.append({"quote_date": day, "grid_time": t, "gamma_source": source,
                                   "bucket": b,
                                   "share_gamma_oi": float(so.get(b, 0.0)) if len(so) and not blank else np.nan,
                                   "share_gamma_vol": float(sv.get(b, 0.0)) if len(sv) and not blank else np.nan,
                                   "n_strikes_oi": int(no.get(b, 0)),
                                   "n_strikes_vol": int(nv.get(b, 0))})
    return pd.DataFrame(share_rows), pd.DataFrame(check_rows)


def liquidity_for_day(df, day, surf, log):
    c = df
    c = log.step(day, "C", "black76_is_otm_365 != 1", c, c[c["black76_is_otm_365"] == 1])
    c = log.step(day, "C", "black76_near_expiry_flag_365 != 0", c, c[c["black76_near_expiry_flag_365"] == 0])
    c = log.step(day, "C", "k non finito", c, c[np.isfinite(c["_k"])])
    sd = c["quote_time"].map(surf.set_index("grid_time")["sd_k"])
    c = c.assign(_z=c["_k"] / sd)
    c = log.step(day, "C", "z non definito (IV_ATM mancante al minuto)", c, c[np.isfinite(c["_z"])])
    c = c.assign(_bucket=bucket_of(c["_z"]),
                 _two=(c["bid"] > 0) & (c["ask"] > 0))
    c = c.assign(_crossed=c["_two"] & (c["bid"] > c["ask"]))
    good = c[c["_two"] & ~c["_crossed"]]
    recomputed = (good["ask"] - good["bid"]) / ((good["ask"] + good["bid"]) / 2.0)
    col = good["black76_spread_pct_of_mid"]
    spread_dev = float(((col - recomputed).abs() / recomputed.abs().clip(lower=1e-300)).max()) if len(good) else 0.0
    n_spread_missing = int(col.isna().sum())

    g = c.groupby(["quote_time", "_bucket"], observed=False)
    out = pd.DataFrame({
        "n_listed_otm": g.size(),
        "n_two_sided": g["_two"].sum(),
        "n_crossed": g["_crossed"].sum(),
    })
    med = good.groupby(["quote_time", "_bucket"], observed=False)["black76_spread_pct_of_mid"].median()
    out["median_rel_spread"] = med
    out["availability"] = out["n_two_sided"] / out["n_listed_otm"].where(out["n_listed_otm"] > 0)
    out = out.reset_index().rename(columns={"quote_time": "grid_time", "_bucket": "bucket"})
    out.insert(0, "quote_date", day)
    out["bucket"] = out["bucket"].astype(str)
    return out, spread_dev, n_spread_missing, int(c["_crossed"].sum())


def aggregate_surface(surface, shares, liq):
    surface = surface.copy()
    surface["sample"] = np.where(surface["quote_date"].str[5:7] == "04", "april", "non_april")
    parts = []
    for sample_name, frame in [("all", surface), ("april", surface[surface["sample"] == "april"]),
                               ("non_april", surface[surface["sample"] == "non_april"])]:
        for t, g in frame.groupby("grid_time"):
            rec = {"sample": sample_name, "grid_time": t, "n_days": int(g["quote_date"].nunique())}
            for v in ["iv_atm", "skew", "curvature", "sd_points"]:
                rec[f"n_{v}"] = int(g[v].notna().sum())
                rec[f"{v}_p25"], rec[f"{v}_median"], rec[f"{v}_p75"] = q(g[v], .25), q(g[v], .5), q(g[v], .75)
            for side in ["zm1_status", "zp1_status"]:
                for st in ["ok", "inside_gap", "beyond_range", "no_strikes", "no_atm"]:
                    rec[f"{side}_{st}"] = int((g[side] == st).sum())
            rec["n_otm_put_usable_median"] = q(g["n_otm_put_usable"].astype(float), .5)
            rec["n_otm_call_usable_median"] = q(g["n_otm_call_usable"].astype(float), .5)
            parts.append(rec)
    surf_agg = pd.DataFrame(parts)

    sh = shares[shares["bucket"] != "undefined"]
    ga = []
    for (src, t, b), g in sh.groupby(["gamma_source", "grid_time", "bucket"]):
        ga.append({"gamma_source": src, "grid_time": t, "bucket": b,
                   "n_days": int(g["share_gamma_oi"].notna().sum()),
                   "share_gamma_oi_p25": q(g["share_gamma_oi"], .25),
                   "share_gamma_oi_median": q(g["share_gamma_oi"], .5),
                   "share_gamma_oi_p75": q(g["share_gamma_oi"], .75),
                   "share_gamma_oi_mean": float(g["share_gamma_oi"].mean()),
                   "share_gamma_vol_p25": q(g["share_gamma_vol"], .25),
                   "share_gamma_vol_median": q(g["share_gamma_vol"], .5),
                   "share_gamma_vol_p75": q(g["share_gamma_vol"], .75),
                   "share_gamma_vol_mean": float(g["share_gamma_vol"].mean()),
                   "n_strikes_oi_median": q(g["n_strikes_oi"].astype(float), .5)})
    gam_agg = pd.DataFrame(ga)
    und = shares[shares["bucket"] == "undefined"]

    la = []
    for (t, b), g in liq.groupby(["grid_time", "bucket"]):
        la.append({"grid_time": t, "bucket": b,
                   "n_days_listed": int((g["n_listed_otm"] > 0).sum()),
                   "availability_p25": q(g["availability"], .25),
                   "availability_median": q(g["availability"], .5),
                   "availability_p75": q(g["availability"], .75),
                   "n_days_spread": int(g["median_rel_spread"].notna().sum()),
                   "rel_spread_p25": q(g["median_rel_spread"], .25),
                   "rel_spread_median": q(g["median_rel_spread"], .5),
                   "rel_spread_p75": q(g["median_rel_spread"], .75),
                   "n_listed_otm_median": q(g["n_listed_otm"].astype(float), .5)})
    liq_agg = pd.DataFrame(la)
    return surf_agg, gam_agg, liq_agg, und


def representative_days():
    panel = pd.read_csv(path(DAILY_PANEL), dtype=str, keep_default_na=False)
    for c in ["quote_date", "RV_t", "is_early_close"]:
        if c not in panel.columns:
            raise CheckFailed(f"pannello giornaliero: manca la colonna {c}")
    reg = panel[panel["is_early_close"].str.strip() == "0"].copy()
    if len(reg) != EXPECTED_REGULAR or set(panel.loc[panel["is_early_close"].str.strip() == "1", "quote_date"]) != EARLY_CLOSES:
        raise CheckFailed(f"pannello giornaliero: {len(reg)} sessioni regolari, attese {EXPECTED_REGULAR}")
    reg["RV_t"] = to_float_exact(reg["RV_t"])
    reg = reg.sort_values(["RV_t", "quote_date"], kind="mergesort").reset_index(drop=True)
    reg["rank"] = np.arange(1, len(reg) + 1)
    pick = {"median_RV": reg.iloc[RANK_MEDIAN - 1], "p90_RV": reg.iloc[RANK_P90 - 1]}
    stress = reg[reg["quote_date"] == STRESS_DAY]
    if stress.empty:
        raise CheckFailed(f"{STRESS_DAY} non fra le sessioni regolari del pannello")
    pick["stress_2025_04_07"] = stress.iloc[0]
    roles = {}
    for role, row in pick.items():
        roles.setdefault(row["quote_date"], []).append(role)
    info = pd.DataFrame([{"day_role": r, "quote_date": row["quote_date"], "RV_t": row["RV_t"],
                          "rank_of_247": int(row["rank"])} for r, row in pick.items()])
    return {d: "+".join(v) for d, v in roles.items()}, info


def load_gamma_reference():
    ref = {}
    for source, (folder, fname, h_oi, h_vol, _) in GAMMA_FILES_H.items():
        g = pd.read_csv(DATA_DIR / folder / fname, dtype=str, keep_default_na=False)
        g = g[g["quote_time"].isin(GRID_TIMES)]
        ho, hv = to_float_exact(g[h_oi]), to_float_exact(g[h_vol])
        ref[source] = {(d, t): (a, b, f) for d, t, a, b, f in
                       zip(g["quote_date"], g["quote_time"], ho, hv, g["gamma_flag"])}
    return ref


def surface_descriptives(only_dates=None):
    """only_dates: sottoinsieme di date da elaborare (di default tutte le sessioni regolari)."""
    t0 = time.time()

    files = {}
    for p in sorted(B76_DIR.glob("02_0DTE_*_black76_365.csv")):
        m = FILE_RE.match(p.name)
        if m and m.group(1) == str(YEAR):
            files[f"{m.group(1)}-{m.group(2)}-{m.group(3)}"] = p
    if only_dates is None and len(files) != EXPECTED_FILES:
        raise CheckFailed(f"file per contratto trovati {len(files)}, attesi {EXPECTED_FILES}")

    roles, rep_info = representative_days()
    print("Giorni rappresentativi:")
    print(rep_info.to_string(index=False))
    gamma_ref = load_gamma_reference()

    days = [d for d in sorted(files) if d not in EARLY_CLOSES]
    if only_dates is not None:
        days = [d for d in days if d in only_dates]
    elif len(days) != EXPECTED_REGULAR:
        raise CheckFailed(f"sessioni regolari fra i file {len(days)}, attese {EXPECTED_REGULAR}")

    log = Log()
    for d in sorted(set(files) & EARLY_CLOSES):
        log.rows.append((d, "lettura", "chiusura anticipata esclusa (file non letto)", 0, 0, 0))

    surf_all, shares_all, liq_all, checks_all, smiles, day_checks = [], [], [], [], [], []
    failure = None
    for i, day in enumerate(days, 1):
        df = read_day_surface(files[day], day, log)
        surf, smile_ready, wrong_side = surface_for_day(df, day, log)
        shares, checks = gamma_for_day(df, day, surf, gamma_ref, log)
        liq, spread_dev, n_spread_missing, n_crossed = liquidity_for_day(df, day, surf, log)
        surf_all.append(surf); shares_all.append(shares); liq_all.append(liq); checks_all.append(checks)

        if day in roles:
            for t in SMILE_TIMES:
                u, iv_atm, sd = smile_ready[t]
                s = u[["strike", "_type", "_k",
                       "black76_implied_volatility_365", "bid", "ask"]].copy()
                s.columns = ["strike", "option_type", "k", "iv", "bid", "ask"]
                s.insert(0, "grid_time", t); s.insert(0, "day_role", roles[day]); s.insert(0, "quote_date", day)
                s["z"] = s["k"] / sd; s["iv_atm"] = iv_atm; s["sd_k"] = sd
                smiles.append(s.sort_values("k"))

        max_h = float(checks[["reldiff_h_oi", "reldiff_h_vol"]].max().max())
        day_checks.append({"quote_date": day, "max_reldiff_h_repro": max_h,
                           "spread_column_max_reldev": spread_dev,
                           "n_spread_column_missing_two_sided": n_spread_missing,
                           "n_crossed_otm": n_crossed, "n_otm_wrong_side_type": wrong_side})
        print(f"[{i:3d}/{len(days)}] {day} | H max scarto {max_h:.1e} | spread col {spread_dev:.1e} "
              f"| {time.time() - t0:6.0f} s", flush=True)
        if max_h > H_REL_TOL:
            bad = checks[(checks["reldiff_h_oi"] > H_REL_TOL) | (checks["reldiff_h_vol"] > H_REL_TOL)]
            failure = f"{day}: Herfindahl ricalcolato diverso dai file del gamma\n{bad.head(10).to_string(index=False)}"
        elif spread_dev > SPREAD_REL_TOL or n_spread_missing > 0:
            failure = (f"{day}: black76_spread_pct_of_mid diverso da (ask - bid) / mid "
                       f"(scarto massimo {spread_dev:.3e}, mancanti {n_spread_missing})")
        elif wrong_side > 0:
            failure = f"{day}: {wrong_side} righe OTM con tipo incoerente con il segno di k"
        if failure:
            break

    checks_df = pd.concat(checks_all, ignore_index=True)
    day_checks_df = pd.DataFrame(day_checks)
    log_df = pd.DataFrame(log.rows, columns=["quote_date", "branch", "step", "rows_before",
                                             "rows_removed", "rows_after"])
    checks_df.merge(day_checks_df, on="quote_date", how="left").to_csv(
        OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}surface_descriptives_checks_by_day_{YEAR}.csv", index=False)
    log_df.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}surface_descriptives_filter_log_by_day_{YEAR}.csv", index=False)
    if failure:
        raise SystemExit(f"\nRISCONTRO FALLITO, lo script si ferma: {failure}")

    surface = pd.concat(surf_all, ignore_index=True)
    shares = pd.concat(shares_all, ignore_index=True)
    liq = pd.concat(liq_all, ignore_index=True)
    surface.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}surface_by_minute_{YEAR}.csv", index=False)
    shares.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}gamma_moneyness_shares_{YEAR}.csv", index=False)
    liq.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}liquidity_by_bucket_{YEAR}.csv", index=False)
    smiles_df = pd.concat(smiles, ignore_index=True) if smiles else pd.DataFrame()
    smiles_df.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}smiles_representative_days_{YEAR}.csv", index=False)

    surf_agg, gam_agg, liq_agg, und = aggregate_surface(surface, shares, liq)
    surf_agg.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}surface_agg_{YEAR}.csv", index=False)
    gam_agg.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}gamma_moneyness_agg_{YEAR}.csv", index=False)
    liq_agg.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}liquidity_agg_{YEAR}.csv", index=False)

    tot = log_df.groupby(["branch", "step"], sort=False)[["rows_before", "rows_removed", "rows_after"]].sum().reset_index()
    tot.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}surface_descriptives_filter_log_{YEAR}.csv", index=False)

    summary = {
        "n_days_processed": len(days),
        "h_oi_max_reldiff_CBOE": float(checks_df.loc[checks_df.gamma_source == "CBOE_gamma", "reldiff_h_oi"].max()),
        "h_vol_max_reldiff_CBOE": float(checks_df.loc[checks_df.gamma_source == "CBOE_gamma", "reldiff_h_vol"].max()),
        "h_oi_max_reldiff_Black76": float(checks_df.loc[checks_df.gamma_source == "Black76_gamma", "reldiff_h_oi"].max()),
        "h_vol_max_reldiff_Black76": float(checks_df.loc[checks_df.gamma_source == "Black76_gamma", "reldiff_h_vol"].max()),
        "n_grid_points_compared": int(len(checks_df)),
        "spread_column_max_reldev": float(day_checks_df["spread_column_max_reldev"].max()),
        "n_crossed_otm_total": int(day_checks_df["n_crossed_otm"].sum()),
        "n_grid_points_iv_atm_missing": int(surface["iv_atm"].isna().sum()),
        "n_grid_points_zm1_ok": int((surface["zm1_status"] == "ok").sum()),
        "n_grid_points_zp1_ok": int((surface["zp1_status"] == "ok").sum()),
        "max_share_gamma_oi_z_undefined": float(und["share_gamma_oi"].max()) if len(und) else 0.0,
        "n_grid_points_with_z_undefined_gamma": int(und[["quote_date", "grid_time"]].drop_duplicates().shape[0]),
        "elapsed_minutes": (time.time() - t0) / 60.0,
    }
    summ = pd.DataFrame([summary])
    for _, r in rep_info.iterrows():
        summ[f"rep_{r['day_role']}"] = f"{r['quote_date']} (rank {r['rank_of_247']}, RV_t {r['RV_t']:.6e})"
    summ.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}surface_descriptives_checks_summary_{YEAR}.csv", index=False)

    print("\nFiltri, totali sull'anno:")
    print(tot.to_string(index=False))
    print("\nRiepilogo:")
    print(summ.T.to_string(header=False))
    print(f"\nScritto in {OUTPUT_DIR}")


# =====================================================================
# PARTE 2: PROFILO PER MINUTO DI H_OI E H_VOL
# =====================================================================
# Per entrambe le fonti di gamma: 247 sessioni regolari, righe con gamma_flag == "ok" fra 09:31
# e 16:00; per minuto, numero di giorni, quartili e mediana di H_OI e H_VOL.
# R1: le diagnostiche giornaliere di 04_Gamma_Herfindahl.py, calcolate su tutte le righe non
# mancanti, vanno riprodotte con tolleranza relativa 1e-12. R2: sulle sole righe ok, la mediana
# del rapporto chiusura/mediana di H_OI deve stare entro l'1 per cento di quella delle
# diagnostiche. Se un riscontro fallisce lo script si ferma.

EXPECTED_DAYS = 250
ROWS_REGULAR_DAY = 390
ROWS_EARLY_CLOSE_DAY = 210

KNOWN_FLAGS = {"ok", "low_oi", "low_vol", "missing_r", "near_expiry",
               "no_valid_cboe_gamma"}

SESSION_START = "09:31:00"
SESSION_END = "16:00:00"

REL_TOL = 1e-12          # R1: riproduzione delle diagnostiche
COHERENCE_TOL = 0.01     # R2: coerenza del rapporto chiusura/mediana

PROFILE_SOURCES = {
    "cboe": {
        "label": "CBOE_gamma",
        "intraday": ("04_gamma_herfindahl", f"04_gamma_cboe_intraday_{YEAR}_all.csv"),
        "diagnostics": ("04_gamma_herfindahl", f"04_gamma_cboe_diagnostics_{YEAR}_all.csv"),
        "h_oi": "h_oi_cboe",
        "h_vol": "h_vol_cboe",
        "diag": {
            "h_oi_min": "h_oi_cboe_min",
            "h_oi_median": "h_oi_cboe_median",
            "h_oi_max": "h_oi_cboe_max",
            "h_oi_close_to_median_ratio": "h_oi_cboe_close_to_median_ratio",
            "h_vol_min": "h_vol_cboe_min",
            "h_vol_median": "h_vol_cboe_median",
            "h_vol_max": "h_vol_cboe_max",
        },
    },
    "indep": {
        "label": "Black76_gamma",
        "intraday": ("04_gamma_herfindahl", f"04_gamma_intraday_{YEAR}_all.csv"),
        "diagnostics": ("04_gamma_herfindahl", f"04_gamma_diagnostics_{YEAR}_all.csv"),
        "h_oi": "h_oi",
        "h_vol": "h_vol",
        "diag": {
            "h_oi_min": "h_oi_min",
            "h_oi_median": "h_oi_median",
            "h_oi_max": "h_oi_max",
            "h_oi_close_to_median_ratio": "h_oi_close_to_median_ratio",
            "h_vol_min": "h_vol_min",
            "h_vol_median": "h_vol_median",
            "h_vol_max": "h_vol_max",
        },
    },
}

COUNT_COLUMNS = ["n_intraday_rows", "n_ok", "n_near_expiry"]

TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_as_strings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def time_to_seconds(series: pd.Series) -> pd.Series:
    parts = series.str.split(":", expand=True).astype(int)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def rel_diff_series(a: pd.Series, b: pd.Series) -> pd.Series:
    a = a.astype(float)
    b = b.astype(float)
    both_nan = a.isna() & b.isna()
    one_nan = a.isna() ^ b.isna()
    out = (a - b).abs() / np.maximum(b.abs(), 1e-300)
    out[both_nan] = 0.0
    out[one_nan] = np.inf
    return out


def check_structure(df: pd.DataFrame, name: str, h_oi: str, h_vol: str) -> None:
    required = ["quote_date", "quote_time", "gamma_flag", h_oi, h_vol]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise CheckFailed(f"[{name}] colonne mancanti: {missing}")

    bad_date = ~df["quote_date"].str.match(DATE_RE)
    if bad_date.any():
        raise CheckFailed(f"[{name}] quote_date fuori formato: "
                          f"{df.loc[bad_date, 'quote_date'].unique()[:5].tolist()}")

    bad_time = ~df["quote_time"].str.match(TIME_RE)
    if bad_time.any():
        raise CheckFailed(f"[{name}] quote_time fuori formato: "
                          f"{df.loc[bad_time, 'quote_time'].unique()[:5].tolist()}")

    flags = set(df["gamma_flag"].unique())
    unknown = flags - KNOWN_FLAGS
    if unknown:
        raise CheckFailed(f"[{name}] valori di gamma_flag non previsti: {sorted(unknown)}")

    dup = df.duplicated(["quote_date", "quote_time"])
    if dup.any():
        raise CheckFailed(f"[{name}] {int(dup.sum())} righe duplicate per (quote_date, quote_time)")

    days = df["quote_date"].unique()
    if len(days) != EXPECTED_DAYS:
        raise CheckFailed(f"[{name}] giorni trovati {len(days)}, attesi {EXPECTED_DAYS}")

    rows_per_day = df.groupby("quote_date").size()
    short = set(rows_per_day[rows_per_day != ROWS_REGULAR_DAY].index)
    if short != EARLY_CLOSES:
        raise CheckFailed(
            f"[{name}] giorni con righe diverse da {ROWS_REGULAR_DAY}: {sorted(short)}; "
            f"attesi esattamente {sorted(EARLY_CLOSES)}"
        )
    bad_early = rows_per_day[list(EARLY_CLOSES)] != ROWS_EARLY_CLOSE_DAY
    if bad_early.any():
        raise CheckFailed(f"[{name}] chiusure anticipate con righe diverse da "
                          f"{ROWS_EARLY_CLOSE_DAY}: {rows_per_day[list(EARLY_CLOSES)].to_dict()}")


def reproduce_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("quote_date", sort=True)
    out = pd.DataFrame({
        "n_intraday_rows": g.size(),
        "n_ok": g["gamma_flag"].apply(lambda s: int((s == "ok").sum())),
        "n_near_expiry": g["gamma_flag"].apply(lambda s: int((s == "near_expiry").sum())),
        "h_oi_min": g["_h_oi"].min(),
        "h_oi_median": g["_h_oi"].median(),
        "h_oi_max": g["_h_oi"].max(),
        "h_vol_min": g["_h_vol"].min(),
        "h_vol_median": g["_h_vol"].median(),
        "h_vol_max": g["_h_vol"].max(),
    })
    # rapporto: ultimo H_OI non mancante in ordine di orario / mediana dei non mancanti
    nonmissing = df[df["_h_oi"].notna()]
    last = nonmissing.groupby("quote_date", sort=True).tail(1).set_index("quote_date")
    out["close_time_pipeline"] = last["quote_time"]
    out["close_flag_pipeline"] = last["gamma_flag"]
    out["h_oi_close_pipeline"] = last["_h_oi"]
    out["h_oi_close_to_median_ratio"] = out["h_oi_close_pipeline"] / out["h_oi_median"]
    out.index.name = "quote_date"
    return out.reset_index()


def compare_with_diagnostics(repro: pd.DataFrame, diag: pd.DataFrame, cfg: dict,
                             name: str) -> pd.DataFrame:
    needed = ["quote_date"] + COUNT_COLUMNS + list(cfg["diag"].values())
    missing = [c for c in needed if c not in diag.columns]
    if missing:
        raise CheckFailed(f"[{name}] colonne mancanti nel file di diagnostica: {missing}")

    d = diag[needed].copy()
    for c in needed[1:]:
        d[c] = to_float_exact(d[c])
    d = d.rename(columns={v: f"diag_{k}" for k, v in cfg["diag"].items()})
    d = d.rename(columns={c: f"diag_{c}" for c in COUNT_COLUMNS})

    m = repro.merge(d, on="quote_date", how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        bad = m.loc[m["_merge"] != "both", ["quote_date", "_merge"]]
        raise CheckFailed(f"[{name}] giorni non allineati fra intraday e diagnostica:\n{bad}")
    m = m.drop(columns="_merge")

    for k in cfg["diag"]:
        m[f"reldiff_{k}"] = rel_diff_series(m[k], m[f"diag_{k}"])
    for c in COUNT_COLUMNS:
        m[f"diff_{c}"] = (m[c].astype(float) - m[f"diag_{c}"]).abs()
    return m


def profile_sample(df: pd.DataFrame, name: str, log: list) -> pd.DataFrame:
    n0 = len(df)
    step = df[~df["quote_date"].isin(EARLY_CLOSES)]
    log.append((name, "tutte le righe del file", n0, 0, n0))
    log.append((name, "esclusione chiusure anticipate", n0, n0 - len(step), len(step)))

    for flag in sorted(KNOWN_FLAGS - {"ok"}):
        n_before = len(step)
        step = step[step["gamma_flag"] != flag]
        log.append((name, f"gamma_flag == {flag}", n_before, n_before - len(step), len(step)))

    n_before = len(step)
    s0, s1 = time_to_seconds(pd.Series([SESSION_START, SESSION_END]))
    step = step[(step["_sec"] >= s0) & (step["_sec"] <= s1)]
    log.append((name, f"fuori da {SESSION_START}-{SESSION_END}", n_before, n_before - len(step), len(step)))

    # righe ok con misura mancante: non tolte, contate (entrano come mancanti nei quantili)
    log.append((name, "righe ok con H_OI mancante (contate, non tolte)", len(step),
                int(step["_h_oi"].isna().sum()), len(step)))
    log.append((name, "righe ok con H_VOL mancante (contate, non tolte)", len(step),
                int(step["_h_vol"].isna().sum()), len(step)))
    return step


def per_day_ok(sample: pd.DataFrame) -> pd.DataFrame:
    g = sample.groupby("quote_date", sort=True)
    last = sample[sample["_h_oi"].notna()].groupby("quote_date", sort=True).tail(1).set_index("quote_date")
    out = pd.DataFrame({
        "n_ok_rows_in_profile": g.size(),
        "h_oi_median_ok": g["_h_oi"].median(),
        "h_vol_median_ok": g["_h_vol"].median(),
    })
    out["close_time_ok"] = last["quote_time"]
    out["h_oi_close_ok"] = last["_h_oi"]
    out["h_oi_close_to_median_ratio_ok"] = out["h_oi_close_ok"] / out["h_oi_median_ok"]
    out.index.name = "quote_date"
    return out.reset_index()


def minute_profile(sample: pd.DataFrame, label: str) -> pd.DataFrame:
    g = sample.groupby("quote_time", sort=True)
    prof = pd.DataFrame({
        "n_days_h_oi": g["_h_oi"].count(),
        "h_oi_p25": g["_h_oi"].quantile(0.25),
        "h_oi_median": g["_h_oi"].median(),
        "h_oi_p75": g["_h_oi"].quantile(0.75),
        "n_days_h_vol": g["_h_vol"].count(),
        "h_vol_p25": g["_h_vol"].quantile(0.25),
        "h_vol_median": g["_h_vol"].median(),
        "h_vol_p75": g["_h_vol"].quantile(0.75),
    }).reset_index()
    prof.insert(0, "gamma_source", label)
    return prof


def gamma_time_profile():
    log, by_day_frames, profiles, summary_rows = [], [], [], []
    failures = []

    for name, cfg in PROFILE_SOURCES.items():
        p_intr = path(cfg["intraday"])
        p_diag = path(cfg["diagnostics"])
        print(f"[{name}] lettura {p_intr}")
        df = read_as_strings(p_intr)
        check_structure(df, name, cfg["h_oi"], cfg["h_vol"])

        df["_h_oi"] = to_float_exact(df[cfg["h_oi"]])
        df["_h_vol"] = to_float_exact(df[cfg["h_vol"]])
        df["_sec"] = time_to_seconds(df["quote_time"])
        df = df.sort_values(["quote_date", "_sec"], kind="mergesort").reset_index(drop=True)

        # R1
        repro = reproduce_diagnostics(df)
        diag = read_as_strings(p_diag)
        m = compare_with_diagnostics(repro, diag, cfg, name)

        # campione del profilo e R2
        sample = profile_sample(df, name, log)
        ok_day = per_day_ok(sample)
        m = m.merge(ok_day, on="quote_date", how="left")
        m.insert(1, "gamma_source", cfg["label"])
        m.insert(2, "is_early_close", m["quote_date"].isin(EARLY_CLOSES).astype(int))
        by_day_frames.append(m)

        rel_cols = [f"reldiff_{k}" for k in cfg["diag"]]
        max_rel = {c: float(m[c].max()) for c in rel_cols}
        max_cnt = {c: float(m[f"diff_{c}"].max()) for c in COUNT_COLUMNS}
        r1_ok = all(v <= REL_TOL for v in max_rel.values()) and all(v == 0 for v in max_cnt.values())
        if not r1_ok:
            bad = m.loc[(m[rel_cols] > REL_TOL).any(axis=1) |
                        (m[[f"diff_{c}" for c in COUNT_COLUMNS]] > 0).any(axis=1),
                        ["quote_date"] + rel_cols + [f"diff_{c}" for c in COUNT_COLUMNS]]
            failures.append(f"[{name}] R1 FALLITO su {len(bad)} giorni. Primi casi:\n"
                            f"{bad.head(10).to_string(index=False)}")

        reg = m[m["is_early_close"] == 0]
        ratio_diag = reg["diag_h_oi_close_to_median_ratio"].astype(float)
        ratio_ok = reg["h_oi_close_to_median_ratio_ok"].astype(float)
        med_diag, med_ok = float(ratio_diag.median()), float(ratio_ok.median())
        r2_gap = abs(med_ok / med_diag - 1.0)
        r2_ok = r2_gap <= COHERENCE_TOL
        if not r2_ok:
            failures.append(f"[{name}] R2 FALLITO: mediana del rapporto {med_ok:.6f} (righe ok) "
                            f"contro {med_diag:.6f} (diagnostica), scarto {r2_gap:.4%}")

        close_times = reg["close_time_ok"].value_counts()
        modal_close = close_times.index[0]
        at_modal = sample[(sample["quote_time"] == modal_close)]
        profile_level_ratio = float(at_modal["_h_oi"].median() / reg["h_oi_median_ok"].median())

        summary_rows.append({
            "gamma_source": cfg["label"],
            "n_days_file": int(df["quote_date"].nunique()),
            "n_regular_days": int(len(reg)),
            **{f"max_{c}": v for c, v in max_rel.items()},
            **{f"max_absdiff_{c}": v for c, v in max_cnt.items()},
            "R1_reproduction_pass": int(r1_ok),
            "ratio_median_diag_regular": med_diag,
            "ratio_mean_diag_regular": float(ratio_diag.mean()),
            "ratio_share_above_2_diag_regular": float((ratio_diag > 2).mean()),
            "ratio_median_ok_regular": med_ok,
            "ratio_mean_ok_regular": float(ratio_ok.mean()),
            "ratio_share_above_2_ok_regular": float((ratio_ok > 2).mean()),
            "R2_gap_rel": r2_gap,
            "R2_coherence_pass": int(r2_ok),
            "close_time_ok_modal": modal_close,
            "n_regular_days_close_not_modal": int((reg["close_time_ok"] != modal_close).sum()),
            "n_regular_days_close_pipeline_differs_from_ok": int(
                (reg["close_time_pipeline"] != reg["close_time_ok"]).sum()),
            "profile_median_h_oi_at_modal_close": float(at_modal["_h_oi"].median()),
            "median_of_daily_ok_median_h_oi": float(reg["h_oi_median_ok"].median()),
            "profile_level_close_over_median": profile_level_ratio,
        })
        profiles.append(minute_profile(sample, cfg["label"]))

        print(f"[{name}] R1 riproduzione diagnostica: {'OK' if r1_ok else 'FALLITO'} "
              f"(max scarto relativo {max(max_rel.values()):.3e})")
        print(f"[{name}] R2 rapporto chiusura/mediana, sessioni regolari: diagnostica "
              f"{med_diag:.4f}, righe ok {med_ok:.4f}, scarto {r2_gap:.3%} "
              f"-> {'OK' if r2_ok else 'FALLITO'}")
        print(f"[{name}] minuto di chiusura modale {modal_close}; giorni diversi "
              f"{int((reg['close_time_ok'] != modal_close).sum())}")

    by_day = pd.concat(by_day_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    log_df = pd.DataFrame(log, columns=["gamma_source_key", "step", "rows_before",
                                        "rows_removed", "rows_after"])

    by_day.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}gamma_time_profile_checks_by_day_{YEAR}.csv", index=False)
    summary.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}gamma_time_profile_checks_summary_{YEAR}.csv", index=False)
    log_df.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}gamma_time_profile_filter_log_{YEAR}.csv", index=False)

    if failures:
        raise SystemExit("\n".join(["", "RISCONTRI FALLITI: il profilo NON viene scritto."] + failures))

    profile = pd.concat(profiles, ignore_index=True)
    profile.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}gamma_time_profile_{YEAR}.csv", index=False)

    print("\nFiltri (righe tolte):")
    print(log_df.to_string(index=False))
    print("\nRiepilogo dei riscontri:")
    print(summary.T.to_string(header=False))
    print(f"\nScritto in {OUTPUT_DIR}")


# =====================================================================
# PARTE 3: GAMMA DEL VENDOR CONTRO GAMMA BLACK '76 PER FASCIA DI Z
# =====================================================================
# Dove e perche' H_OI del gamma del vendor e quello del gamma Black '76 divergono nell'ultima
# ora, mentre H_VOL coincide. Ipotesi: nelle ali il vendor usa una IV, e quindi un gamma, piu'
# alti; e' respinta se alla chiusura i rapporti nelle ali non superano 1.
#
# z e fasce come nella parte 1; universo di ciascuna fonte come in 04_Gamma_Herfindahl.py, con
# open_interest > 0; lati otm, itm e all. exposure_ratio (somma di gamma x OI del vendor su quella
# Black '76) include sia i gamma del vendor arrotondati a zero sia la differenza di livello.
# A ogni minuto la somma di gamma x OI deve coincidere con quella dei file del gamma (1e-9).

SURFACE_FILE = ("10_surface_descriptives", f"10_surface_by_minute_{YEAR}.csv")    # della parte 1
GAMMA_FILES_GOI = {"CBOE_gamma": ("04_gamma_herfindahl", f"04_gamma_cboe_intraday_{YEAR}_all.csv", "g_oi_raw_cboe"),
                   "Black76_gamma": ("04_gamma_herfindahl", f"04_gamma_intraday_{YEAR}_all.csv", "g_oi_raw")}
G_REL_TOL = 1e-9
USECOLS_VENDOR = ["quote_date", "quote_time", "strike", "option_type", "open_interest", "gamma",
                  "implied_volatility", "black76_r", "black76_T_365", "black76_F_365",
                  "black76_log_moneyness_365", "black76_implied_volatility_365", "black76_is_otm_365",
                  "black76_iv_at_bound_365", "black76_gamma_365", "black76_near_expiry_flag_365",
                  "black76_solver_success_365"]
NUMERIC_VENDOR = [c for c in USECOLS_VENDOR if c not in ("quote_date", "quote_time", "option_type")]


def universe(df, gcol):
    solver = df["black76_solver_success_365"].fillna(0).astype(int)
    near = df["black76_near_expiry_flag_365"].fillna(0).astype(int)
    bound = df["black76_iv_at_bound_365"].fillna(0).astype(int)
    g, K, tau, r = df[gcol], df["strike"], df["black76_T_365"], df["black76_r"]
    return ((solver == 1) & (near == 0) & (bound != 1) & np.isfinite(K) & (K > 0)
            & np.isfinite(tau) & (tau > 0) & np.isfinite(r) & df["_type"].isin(["C", "P"])
            & np.isfinite(g) & (g > 0) & np.isfinite(df["open_interest"]) & (df["open_interest"] > 0))


def process_day(path, day, sd_map, gref, log):
    raw = pd.read_csv(path, usecols=USECOLS_VENDOR, dtype=str, keep_default_na=False)
    log.append((day, "righe nel file", len(raw), 0, len(raw)))
    df = raw[raw["quote_time"].isin(GRID_TIMES)].copy()
    log.append((day, "fuori dai 38 punti di griglia", len(raw), len(raw) - len(df), len(df)))
    if set(df["quote_time"]) != set(GRID_TIMES):
        raise CheckFailed(f"{day}: punti di griglia mancanti {sorted(set(GRID_TIMES) - set(df['quote_time']))}")
    for c in NUMERIC_VENDOR:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    lnfk = np.log(df["black76_F_365"] / df["strike"]); col = df["black76_log_moneyness_365"]
    ok = np.isfinite(lnfk) & np.isfinite(col)
    dev = float((col[ok] - lnfk[ok]).abs().max()) if ok.any() else np.inf
    if not dev <= K_SIGN_TOL:
        raise CheckFailed(f"{day}: black76_log_moneyness_365 non e' log(F/K) (scarto {dev:.3e})")
    ot = df["option_type"].astype(str).str.strip().str.upper()
    df["_type"] = np.where(ot.isin(["C", "CALL"]), "C", np.where(ot.isin(["P", "PUT"]), "P", ""))
    df["_k"] = -col
    sd = df["quote_time"].map(lambda t: sd_map.get((day, t), np.nan))
    if sd.isna().any():
        raise CheckFailed(f"{day}: sd_k mancante in surface_by_minute per {sorted(df.loc[sd.isna(), 'quote_time'].unique())[:5]}")
    df["_bucket"] = pd.cut(df["_k"] / sd, bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=True).astype(object)
    df["_side"] = np.where(df["black76_is_otm_365"] == 1, "otm", "itm")

    u_v = universe(df, "gamma"); u_o = universe(df, "black76_gamma_365")
    log.append((day, "universo vendor (con OI > 0)", len(df), int((~u_v).sum()), int(u_v.sum())))
    log.append((day, "universo tesi (con OI > 0)", len(df), int((~u_o).sum()), int(u_o.sum())))
    df["_ev"] = np.where(u_v, df["gamma"] * df["open_interest"], 0.0)
    df["_eo"] = np.where(u_o, df["black76_gamma_365"] * df["open_interest"], 0.0)

    checks = []
    for t, g in df.groupby("quote_time"):
        for src, colname, ecol in [("CBOE_gamma", "g_oi_raw_cboe", "_ev"), ("Black76_gamma", "g_oi_raw", "_eo")]:
            mine = float(g[ecol].sum()); ref = gref[src].get((day, t), np.nan)
            checks.append({"quote_date": day, "grid_time": t, "gamma_source": src,
                           "g_oi_repro": mine, "g_oi_file": ref, "reldiff": rel_diff(mine, ref)})

    o = df[u_o].copy()
    both = o["gamma"].notna() & (o["gamma"] > 0)
    o["_gratio"] = np.where(both, o["gamma"] / o["black76_gamma_365"], np.nan)
    o["_vzero"] = (~both).astype(int)
    viv = o["implied_volatility"]
    o["_ivratio"] = np.where(np.isfinite(viv) & (viv > 0), viv / o["black76_implied_volatility_365"], np.nan)
    o["_ivzero"] = (~(np.isfinite(viv) & (viv > 0))).astype(int)

    rows = []
    for side in ["otm", "itm", "all"]:
        dsel = df if side == "all" else df[df["_side"] == side]
        osel = o if side == "all" else o[o["_side"] == side]
        ev = dsel.groupby(["quote_time", "_bucket"])["_ev"].sum()
        eo = dsel.groupby(["quote_time", "_bucket"])["_eo"].sum()
        go = osel.groupby(["quote_time", "_bucket"])
        stats = pd.DataFrame({
            "n_rows_own_universe": go.size(),
            "gamma_ratio_median": go["_gratio"].median(),
            "share_rows_vendor_gamma_zero": go["_vzero"].mean(),
            "iv_ratio_median": go["_ivratio"].median(),
            "n_vendor_iv_zero": go["_ivzero"].sum(),
        })
        stats["exposure_ratio"] = (ev / eo.where(eo > 0)).reindex(stats.index)
        stats = stats.reset_index().rename(columns={"quote_time": "grid_time", "_bucket": "bucket"})
        stats.insert(0, "side", side); stats.insert(0, "quote_date", day)
        rows.append(stats)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(checks)


def vendor_comparison(only=None):
    """only: sottoinsieme di date da elaborare (di default tutte le sessioni regolari)."""
    t0 = time.time()
    files = {}
    for p in sorted(B76_DIR.glob("02_0DTE_*_black76_365.csv")):
        m = FILE_RE.match(p.name)
        if m and m.group(1) == str(YEAR):
            files[f"{m.group(1)}-{m.group(2)}-{m.group(3)}"] = p
    if only is None and len(files) != EXPECTED_FILES:
        raise CheckFailed(f"file per contratto trovati {len(files)}, attesi {EXPECTED_FILES}")
    surf = pd.read_csv(path(SURFACE_FILE), dtype=str, keep_default_na=False)
    sd_map = dict(zip(zip(surf["quote_date"], surf["grid_time"]), to_float_exact(surf["sd_k"])))
    gref = {}
    for src, (folder, fname, col) in GAMMA_FILES_GOI.items():
        g = pd.read_csv(DATA_DIR / folder / fname, dtype=str, keep_default_na=False)
        g = g[g["quote_time"].isin(GRID_TIMES)]
        gref[src] = dict(zip(zip(g["quote_date"], g["quote_time"]), to_float_exact(g[col])))
    days = [d for d in sorted(files) if d not in EARLY_CLOSES and (only is None or d in only)]
    if only is None and len(days) != EXPECTED_REGULAR:
        raise CheckFailed(f"sessioni regolari {len(days)}, attese {EXPECTED_REGULAR}")

    log, res, chk, failure = [], [], [], None
    for i, day in enumerate(days, 1):
        r, c = process_day(files[day], day, sd_map, gref, log)
        res.append(r); chk.append(c)
        mx = float(c["reldiff"].max())
        print(f"[{i:3d}/{len(days)}] {day} | g_oi max scarto {mx:.1e} | {time.time() - t0:6.0f} s", flush=True)
        if mx > G_REL_TOL:
            failure = f"{day}: somma di gamma x OI diversa dai file del gamma\n{c[c.reldiff > G_REL_TOL].head(10).to_string(index=False)}"
            break
    chk_df = pd.concat(chk, ignore_index=True)
    chk_df.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}vendor_comparison_checks_by_day_{YEAR}.csv", index=False)
    log_df = pd.DataFrame(log, columns=["quote_date", "step", "rows_before", "rows_removed", "rows_after"])
    log_df.groupby("step", sort=False)[["rows_before", "rows_removed", "rows_after"]].sum().reset_index().to_csv(
        OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}vendor_comparison_filter_log_{YEAR}.csv", index=False)
    if failure:
        raise SystemExit(f"\nRISCONTRO FALLITO, lo script si ferma: {failure}")
    full = pd.concat(res, ignore_index=True)
    full.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}vendor_vs_own_by_bucket_{YEAR}.csv", index=False)
    agg = []
    for (side, t, b), g in full.groupby(["side", "grid_time", "bucket"]):
        rec = {"side": side, "grid_time": t, "bucket": b, "n_days": int(len(g))}
        for v in ["exposure_ratio", "gamma_ratio_median", "share_rows_vendor_gamma_zero", "iv_ratio_median"]:
            x = g[v].dropna()
            rec[f"{v}_n"] = int(len(x))
            rec[f"{v}_p25"] = float(x.quantile(.25)) if len(x) else np.nan
            rec[f"{v}_median"] = float(x.median()) if len(x) else np.nan
            rec[f"{v}_p75"] = float(x.quantile(.75)) if len(x) else np.nan
        rec["n_rows_own_universe_median"] = float(g["n_rows_own_universe"].median())
        rec["n_vendor_iv_zero_total"] = int(g["n_vendor_iv_zero"].sum())
        agg.append(rec)
    pd.DataFrame(agg).to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}vendor_vs_own_agg_{YEAR}.csv", index=False)
    summ = pd.DataFrame([{
        "n_days_processed": len(days),
        "g_oi_max_reldiff_CBOE": float(chk_df.loc[chk_df.gamma_source == "CBOE_gamma", "reldiff"].max()),
        "g_oi_max_reldiff_Black76": float(chk_df.loc[chk_df.gamma_source == "Black76_gamma", "reldiff"].max()),
        "n_grid_points_compared": int(len(chk_df)),
        "elapsed_minutes": (time.time() - t0) / 60.0}])
    summ.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}vendor_comparison_checks_summary_{YEAR}.csv", index=False)
    print("\nRiepilogo:"); print(summ.T.to_string(header=False))
    print(f"\nScritto in {OUTPUT_DIR}")


# =====================================================================
# PARTE 4: SUPERFICIE DELLA IV SU UNA GRIGLIA FISSA DI Z (FIG. 2.1)
# =====================================================================
# Mappa di calore della superficie: mediana fra le sessioni della IV per z e ora del giorno.
# Righe, IV_ATM e z sono quelli del ramo A della parte 1; la IV e' valutata su 53 nodi di z da
# -8 a +5 (passo 0.25), lo stesso intervallo della figura.
#
# Valore a un nodo: interpolazione lineare in z sulla curva OTM (fra il put e la call piu' vicini
# e' la regola di IV_ATM), senza estrapolazione oltre lo strike piu' lontano. Si calcola anche
# IV(z) / IV_ATM, la forma al netto del livello.
#
# Per le tre sessioni rappresentative si salvano tutti i punti usati dalla superficie, da cui
# 17_Figures.py disegna le superfici 3D passando per gli strike veri.
#
# Riscontri, ognuno ferma lo script: R1 colonna = log(F/K); R2 tipi coerenti con il segno di k,
# nessun z duplicato; R3-R4 valori a z = 0 e +-1 uguali alla parte 1; R5 filtri uguali alla
# parte 1; R7 punti ai minuti degli smile uguali alla figura A1 (tolleranza relativa 1e-12).

REFERENCE_FOLDER = "10_surface_descriptives"                          # uscite della parte 1
REF_SURFACE = f"10_surface_by_minute_{YEAR}.csv"
REF_FILTER_LOG = f"10_surface_descriptives_filter_log_{YEAR}.csv"
REF_SMILES = f"10_smiles_representative_days_{YEAR}.csv"
# Valore fisso della colonna r6 nelle uscite (riscontro non eseguito).
R6_NOT_RUN = f"R6 non eseguito: riscontro esterno sul {YEAR} non disponibile"

Z_NODES = np.round(np.arange(-8.0, 5.0 + 1e-9, 0.25), 2)
assert len(Z_NODES) == 53 and Z_NODES[0] == -8.0 and Z_NODES[-1] == 5.0
I_ZERO = int(np.where(Z_NODES == 0.0)[0][0])
I_M1 = int(np.where(Z_NODES == -1.0)[0][0])
I_P1 = int(np.where(Z_NODES == 1.0)[0][0])
COVERAGE_SHARE = 0.5          # cella mostrata se valore in >= meta' delle sessioni

REPRO_TOL = 1e-12             # differenza relativa, riscontri R3-R4

USECOLS_HEATMAP = [
    "quote_date", "quote_time", "strike", "option_type", "bid",
    "black76_T_365", "black76_F_365", "black76_log_moneyness_365",
    "black76_implied_volatility_365", "black76_is_otm_365",
    "black76_iv_at_bound_365", "black76_iv_converged_365",
    "black76_near_expiry_flag_365", "black76_solver_success_365",
]
NUMERIC_HEATMAP = [c for c in USECOLS_HEATMAP if c not in ("quote_date", "quote_time", "option_type")]


def read_day_heatmap(path: Path, day: str, log: Log) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=USECOLS_HEATMAP, dtype=str, keep_default_na=False)
    dates = raw["quote_date"].unique()
    if len(dates) != 1 or dates[0] != day:
        raise CheckFailed(f"{path.name}: quote_date {dates[:3]} diverso da {day}")
    log.rows.append((day, "lettura", "righe nel file", len(raw), 0, len(raw)))
    df = log.step(day, "lettura", "fuori dai 38 punti di griglia", raw,
                  raw[raw["quote_time"].isin(GRID_TIMES)].copy())
    present = set(df["quote_time"].unique())
    if present != set(GRID_TIMES):
        raise CheckFailed(f"{day}: punti di griglia assenti nel file: {sorted(set(GRID_TIMES) - present)}")
    for c in NUMERIC_HEATMAP:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ln_f_over_k = np.log(df["black76_F_365"] / df["strike"])
    col = df["black76_log_moneyness_365"]
    both = np.isfinite(ln_f_over_k) & np.isfinite(col)
    dev = float((col[both] - ln_f_over_k[both]).abs().max()) if both.any() else np.inf
    if not dev <= K_SIGN_TOL:
        dev_other = float((col[both] + ln_f_over_k[both]).abs().max()) if both.any() else np.inf
        raise CheckFailed(f"R1 {day}: black76_log_moneyness_365 non e' log(F/K) (scarto {dev:.3e}; "
                          f"scarto da log(K/F) {dev_other:.3e})")
    df["_k"] = -col
    ot = df["option_type"].astype(str).str.strip().str.upper()
    df["_type"] = np.where(ot.isin(["C", "CALL"]), "C", np.where(ot.isin(["P", "PUT"]), "P", ""))
    return df


def branch_a(df: pd.DataFrame, day: str, log: Log) -> pd.DataFrame:
    a = df
    a = log.step(day, "A", "black76_is_otm_365 != 1", a, a[a["black76_is_otm_365"] == 1])
    a = log.step(day, "A", "black76_iv_converged_365 != 1", a, a[a["black76_iv_converged_365"] == 1])
    a = log.step(day, "A", "black76_iv_at_bound_365 != 0", a, a[a["black76_iv_at_bound_365"] == 0])
    a = log.step(day, "A", "black76_solver_success_365 != 1", a, a[a["black76_solver_success_365"] == 1])
    a = log.step(day, "A", "black76_near_expiry_flag_365 != 0", a, a[a["black76_near_expiry_flag_365"] == 0])
    a = log.step(day, "A", "bid <= 0 o mancante", a, a[a["bid"] > 0])
    iv, k = a["black76_implied_volatility_365"], a["_k"]
    a = log.step(day, "A", "IV o k non finiti, o IV <= 0", a,
                 a[np.isfinite(iv) & np.isfinite(k) & (iv > 0)])
    a = log.step(day, "A", "k = 0 esatto (ne' put ne' call)", a, a[a["_k"] != 0])
    wrong_side = int((((a["_k"] < 0) & (a["_type"] != "P")) |
                      ((a["_k"] > 0) & (a["_type"] != "C"))).sum())
    if wrong_side:
        raise CheckFailed(f"R2 {day}: {wrong_side} righe OTM con tipo incoerente con il segno di k")
    return a


def nodes_for_minute(puts: pd.DataFrame, calls: pd.DataFrame, sd: float) -> np.ndarray:
    z = np.concatenate([puts["_k"].to_numpy(), calls["_k"].to_numpy()]) / sd
    v = np.concatenate([puts["black76_implied_volatility_365"].to_numpy(),
                        calls["black76_implied_volatility_365"].to_numpy()])
    order = np.argsort(z, kind="mergesort")
    z, v = z[order], v[order]
    if np.any(np.diff(z) <= 0):
        raise CheckFailed("R2: z duplicati o non ordinati in un minuto")
    out = np.full(len(Z_NODES), np.nan)
    inside = (Z_NODES >= z[0]) & (Z_NODES <= z[-1])
    out[inside] = np.interp(Z_NODES[inside], z, v)
    return out


def surface_nodes_for_day(df: pd.DataFrame, a: pd.DataFrame, day: str, keep_points: bool = False):
    rows, minute_rows, point_parts = [], [], []
    for t in GRID_TIMES:
        allt = df[df["quote_time"] == t]
        tau = float(np.nanmedian(allt["black76_T_365"])) if allt["black76_T_365"].notna().any() else np.nan
        u = a[a["quote_time"] == t]
        puts = u[u["_k"] < 0]
        calls = u[u["_k"] > 0]
        iv_atm = np.nan
        if len(puts) and len(calls):
            ip = puts["_k"].idxmax()
            ic = calls["_k"].idxmin()
            k_p, v_p = puts.at[ip, "_k"], puts.at[ip, "black76_implied_volatility_365"]
            k_c, v_c = calls.at[ic, "_k"], calls.at[ic, "black76_implied_volatility_365"]
            iv_atm = v_p + (0.0 - k_p) * (v_c - v_p) / (k_c - k_p)
        sd = iv_atm * math.sqrt(tau) if np.isfinite(iv_atm) and np.isfinite(tau) and tau > 0 else np.nan
        vals = nodes_for_minute(puts, calls, sd) if np.isfinite(sd) else np.full(len(Z_NODES), np.nan)
        if keep_points:
            pts = u[["strike", "_type", "_k", "black76_implied_volatility_365"]].copy()
            pts.columns = ["strike", "option_type", "k", "iv"]
            pts["z"] = pts["k"] / sd                      # come nel file degli smile
            pts["iv_atm"] = iv_atm
            pts["sd_k"] = sd
            pts.insert(0, "grid_time", t)
            pts.insert(0, "quote_date", day)
            point_parts.append(pts.sort_values("k"))
        minute_rows.append({"quote_date": day, "grid_time": t, "iv_atm": iv_atm, "sd_k": sd,
                            "n_otm_put_usable": len(puts), "n_otm_call_usable": len(calls),
                            "iv_node_0": vals[I_ZERO], "iv_node_m1": vals[I_M1], "iv_node_p1": vals[I_P1]})
        for z, v in zip(Z_NODES, vals):
            rows.append((day, t, float(z), v, v / iv_atm if np.isfinite(v) and np.isfinite(iv_atm) else np.nan))
    nodes = pd.DataFrame(rows, columns=["quote_date", "grid_time", "z", "iv", "iv_rel"])
    points = pd.concat(point_parts, ignore_index=True) if point_parts else None
    return nodes, pd.DataFrame(minute_rows), points


def load_reference_surface() -> pd.DataFrame:
    path = DATA_DIR / REFERENCE_FOLDER / REF_SURFACE
    if not path.exists():
        raise CheckFailed(f"manca il file di riscontro {path}")
    ref = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = ["quote_date", "grid_time", "iv_atm", "n_otm_put_usable", "n_otm_call_usable",
            "iv_zm1", "iv_zp1", "zm1_status", "zp1_status"]
    missing = [c for c in need if c not in ref.columns]
    if missing:
        raise CheckFailed(f"{REF_SURFACE}: colonne mancanti {missing}")
    for c in ["iv_atm", "iv_zm1", "iv_zp1"]:
        ref[c] = to_float_exact(ref[c])
    for c in ["n_otm_put_usable", "n_otm_call_usable"]:
        ref[c] = to_float_exact(ref[c]).astype(int)
    return ref.set_index(["quote_date", "grid_time"])


def check_against_reference(minutes: pd.DataFrame, ref: pd.DataFrame, day: str) -> dict:
    worst = {"r3_iv_atm": 0.0, "r4_zm1": 0.0, "r4_zp1": 0.0, "n_r4_zm1": 0, "n_r4_zp1": 0}
    for _, m in minutes.iterrows():
        key = (m["quote_date"], m["grid_time"])
        if key not in ref.index:
            raise CheckFailed(f"R3 {day} {m['grid_time']}: punto assente in {REF_SURFACE}")
        r = ref.loc[key]
        if int(m["n_otm_put_usable"]) != int(r["n_otm_put_usable"]) or \
           int(m["n_otm_call_usable"]) != int(r["n_otm_call_usable"]):
            raise CheckFailed(f"R3 {day} {m['grid_time']}: strike utilizzabili {m['n_otm_put_usable']}/"
                              f"{m['n_otm_call_usable']} contro {r['n_otm_put_usable']}/{r['n_otm_call_usable']}")
        d0 = max(rel_diff(m["iv_atm"], r["iv_atm"]), rel_diff(m["iv_node_0"], r["iv_atm"]))
        if d0 > REPRO_TOL:
            raise CheckFailed(f"R3 {day} {m['grid_time']}: IV_ATM {float(m['iv_atm'])!r} / nodo 0 "
                              f"{float(m['iv_node_0'])!r} contro {float(r['iv_atm'])!r}")
        worst["r3_iv_atm"] = max(worst["r3_iv_atm"], d0)
        for node, col, st, wk in [("iv_node_m1", "iv_zm1", "zm1_status", "zm1"),
                                  ("iv_node_p1", "iv_zp1", "zp1_status", "zp1")]:
            if str(r[st]).strip() == "ok":
                d = rel_diff(m[node], r[col])
                if d > REPRO_TOL:
                    raise CheckFailed(f"R4 {day} {m['grid_time']}: nodo {wk} {float(m[node])!r} "
                                      f"contro {float(r[col])!r}")
                worst[f"r4_{wk}"] = max(worst[f"r4_{wk}"], d)
                worst[f"n_r4_{wk}"] += 1
    return worst


def check_filter_log(log_df: pd.DataFrame) -> str:
    path = DATA_DIR / REFERENCE_FOLDER / REF_FILTER_LOG
    if not path.exists():
        return f"R5 non eseguito: manca {path}"
    ref = pd.read_csv(path, dtype=str, keep_default_na=False)
    ref = ref[ref["branch"].isin(["lettura", "A"])].copy()
    for c in ["rows_before", "rows_removed", "rows_after"]:
        ref[c] = to_float_exact(ref[c]).astype(np.int64)
    ref = ref.set_index(["branch", "step"])[["rows_before", "rows_removed", "rows_after"]]
    mine = log_df.groupby(["branch", "step"], sort=False)[["rows_before", "rows_removed", "rows_after"]].sum()
    mine = mine.astype(np.int64)
    for key, r in mine.iterrows():
        if key not in ref.index:
            raise CheckFailed(f"R5: passo {key} assente nel registro di riferimento")
        if tuple(r) != tuple(ref.loc[key]):
            raise CheckFailed(f"R5: passo {key}: {tuple(r)} contro {tuple(ref.loc[key])}")
    extra = [k for k in ref.index if k not in mine.index]
    if extra:
        raise CheckFailed(f"R5: passi del riferimento non ritrovati: {extra}")
    return f"R5 superato: totali dei filtri identici a {REFERENCE_FOLDER}"


def load_smiles():
    """Le tre sessioni rappresentative e i punti della figura A1."""
    path = DATA_DIR / REFERENCE_FOLDER / REF_SMILES
    if not path.exists():
        raise CheckFailed(f"manca il file degli smile {path}")
    sm = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = ["quote_date", "day_role", "grid_time", "option_type", "k", "iv", "z", "iv_atm"]
    missing = [c for c in need if c not in sm.columns]
    if missing:
        raise CheckFailed(f"{REF_SMILES}: colonne mancanti {missing}")
    for c in ["k", "iv", "z", "iv_atm"]:
        sm[c] = to_float_exact(sm[c])
    days = sorted(sm["quote_date"].unique())
    if len(days) != 3 or sorted(sm["grid_time"].unique()) != SMILE_TIMES:
        raise CheckFailed(f"{REF_SMILES}: attese 3 sessioni e i minuti {SMILE_TIMES}")
    return days, sm


def check_points_against_smiles(points: pd.DataFrame, sm: pd.DataFrame) -> str:
    """R7: ai minuti degli smile i punti della superficie sono gli stessi strike, con gli
    stessi valori, della figura A1."""
    worst, n = 0.0, 0
    for (day, t), ref in sm.groupby(["quote_date", "grid_time"], sort=True):
        mine = points[(points["quote_date"] == day) & (points["grid_time"] == t)].sort_values("k")
        ref = ref.sort_values("k")
        if len(mine) != len(ref):
            raise CheckFailed(f"R7 {day} {t}: {len(mine)} punti contro {len(ref)} della figura A1")
        if list(mine["option_type"]) != list(ref["option_type"].str.strip().str.upper()):
            raise CheckFailed(f"R7 {day} {t}: tipi di opzione diversi da quelli della figura A1")
        for c in ["k", "z", "iv", "iv_atm"]:
            a = mine[c].to_numpy(dtype=float)
            b = ref[c].to_numpy(dtype=float)
            d = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
            d = float(d.max()) if len(d) else 0.0
            if not d <= REPRO_TOL:
                raise CheckFailed(f"R7 {day} {t}: colonna {c}, scarto relativo {d:.3e}")
            worst = max(worst, d)
        n += len(ref)
    return f"R7 superato: {n} punti ai minuti degli smile identici alla figura A1, max {worst:.1e}"


def aggregate_heatmap(nodes: pd.DataFrame) -> pd.DataFrame:
    nodes = nodes.copy()
    nodes["sample"] = np.where(nodes["quote_date"].str[5:7] == "04", "april", "non_april")
    parts = []
    for sample_name, frame in [("all", nodes), ("non_april", nodes[nodes["sample"] == "non_april"]),
                               ("april", nodes[nodes["sample"] == "april"])]:
        n_sessions = int(frame["quote_date"].nunique())
        need = int(math.ceil(COVERAGE_SHARE * n_sessions))
        for (t, z), g in frame.groupby(["grid_time", "z"], sort=True):
            n = int(g["iv"].notna().sum())
            parts.append({"sample": sample_name, "grid_time": t, "z": z,
                          "n_sessions": n_sessions, "n_days": n,
                          "coverage_ok": bool(n >= need),
                          "iv_p25": q(g["iv"], .25), "iv_median": q(g["iv"], .5), "iv_p75": q(g["iv"], .75),
                          "iv_rel_p25": q(g["iv_rel"], .25), "iv_rel_median": q(g["iv_rel"], .5),
                          "iv_rel_p75": q(g["iv_rel"], .75)})
    return pd.DataFrame(parts)


def surface_iv_heatmap(only_dates=None):
    """only_dates: sottoinsieme di date da elaborare (di default tutte le sessioni regolari)."""
    t0 = time.time()
    rep_days, smiles = load_smiles()
    files = {}
    for p in sorted(B76_DIR.glob("02_0DTE_*_black76_365.csv")):
        m = FILE_RE.match(p.name)
        if m and m.group(1) == str(YEAR):
            files[f"{m.group(1)}-{m.group(2)}-{m.group(3)}"] = p
    if only_dates is None and len(files) != EXPECTED_FILES:
        raise CheckFailed(f"file per contratto trovati {len(files)}, attesi {EXPECTED_FILES}")
    days = [d for d in sorted(files) if d not in EARLY_CLOSES]
    if only_dates is not None:
        days = [d for d in days if d in only_dates]
        if not days:
            raise CheckFailed("nessuna delle date richieste e' una sessione regolare con file")
    elif len(days) != EXPECTED_REGULAR:
        raise CheckFailed(f"sessioni regolari fra i file {len(days)}, attese {EXPECTED_REGULAR}")

    ref = load_reference_surface()
    log = Log()
    for d in sorted(set(files) & EARLY_CLOSES):
        log.rows.append((d, "lettura", "chiusura anticipata esclusa (file non letto)", 0, 0, 0))

    all_nodes, worst_rows, all_points = [], [], []
    for i, day in enumerate(days, 1):
        df = read_day_heatmap(files[day], day, log)
        a = branch_a(df, day, log)
        nodes, minutes, points = surface_nodes_for_day(df, a, day, keep_points=day in rep_days)
        if points is not None:
            all_points.append(points)
        worst = check_against_reference(minutes, ref, day)
        worst["quote_date"] = day
        worst_rows.append(worst)
        all_nodes.append(nodes)
        print(f"[{i:3d}/{len(days)}] {day} | R3 {worst['r3_iv_atm']:.1e} | R4 {max(worst['r4_zm1'], worst['r4_zp1']):.1e} "
              f"| {time.time() - t0:6.0f} s", flush=True)

    log_df = pd.DataFrame(log.rows, columns=["quote_date", "branch", "step", "rows_before",
                                             "rows_removed", "rows_after"])
    r5 = check_filter_log(log_df) if only_dates is None else "R5 non eseguito: run di prova su poche date"

    nodes = pd.concat(all_nodes, ignore_index=True)
    r6 = R6_NOT_RUN

    missing_rep = sorted(set(rep_days) - set(days))
    if missing_rep and only_dates is None:
        raise CheckFailed(f"sessioni rappresentative non elaborate: {missing_rep}")
    if missing_rep:                          # prova su poche date senza i tre giorni
        r7 = "R7 non eseguito: run di prova senza le tre sessioni rappresentative"
        print(r7, flush=True)
    else:
        points = pd.concat(all_points, ignore_index=True)
        r7 = check_points_against_smiles(points, smiles)
        print(r7, flush=True)
        points.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}iv_surface_points_repdays_{YEAR}.csv", index=False)
        pts_summary = {"days": ",".join(rep_days), "n_points": len(points),
                       "n_minutes_with_atm": int(points.groupby(["quote_date", "grid_time"])["iv_atm"].first().notna().sum()),
                       "r3_max_reldiff_iv_atm": float(pd.DataFrame(worst_rows)["r3_iv_atm"].max()),
                       "r6": r6, "r7": r7}
        pd.DataFrame([pts_summary]).to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}iv_surface_points_checks_{YEAR}.csv", index=False)
    nodes.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}iv_surface_by_z_by_day_{YEAR}.csv", index=False)
    agg = aggregate_heatmap(nodes)
    agg.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}iv_surface_by_z_agg_{YEAR}.csv", index=False)
    tot = log_df.groupby(["branch", "step"], sort=False)[["rows_before", "rows_removed", "rows_after"]].sum().reset_index()
    tot.to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}iv_surface_filter_log_{YEAR}.csv", index=False)

    w = pd.DataFrame(worst_rows)
    allg = agg[agg["sample"] == "all"]
    summary = {
        "n_days_processed": len(days),
        "full_run": only_dates is None,
        "n_z_nodes": len(Z_NODES), "z_min": float(Z_NODES[0]), "z_max": float(Z_NODES[-1]),
        "r3_max_reldiff_iv_atm": float(w["r3_iv_atm"].max()),
        "r4_max_reldiff_zm1": float(w["r4_zm1"].max()), "r4_n_compared_zm1": int(w["n_r4_zm1"].sum()),
        "r4_max_reldiff_zp1": float(w["r4_zp1"].max()), "r4_n_compared_zp1": int(w["n_r4_zp1"].sum()),
        "r5": r5,
        "r6": r6,
        "r7": r7,
        "n_cells_all": int(len(allg)), "n_cells_all_coverage_ok": int(allg["coverage_ok"].sum()),
        "elapsed_minutes": (time.time() - t0) / 60.0,
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}iv_surface_checks_summary_{YEAR}.csv", index=False)

    print("\nFiltri, totali:")
    print(tot.to_string(index=False))
    print("\nRiepilogo:")
    print(pd.Series(summary).to_string())
    print(f"\nScritto in {OUTPUT_DIR}")


# =====================================================================
# PARTE 5: TABELLE 2.1 E 2.2 E NUMERI DEL CAPITOLO 2
# =====================================================================
# Ricava le Tabelle 2.1 e 2.2 e i numeri del testo del Capitolo 2 dagli output delle parti 1-4
# e di altri script, senza nuove stime, e li confronta con i valori stampati nella tesi.

MFIV_EOS = ("03_mfiv", "03_mfiv_end_of_session_2025_all.csv")
SURF_AGG = ("10_surface_descriptives", "10_surface_agg_2025.csv")
SURF_MIN = ("10_surface_descriptives", "10_surface_by_minute_2025.csv")
GAM_AGG = ("10_surface_descriptives", "10_gamma_moneyness_agg_2025.csv")
GAM_SHARES = ("10_surface_descriptives", "10_gamma_moneyness_shares_2025.csv")
SMILES = ("10_surface_descriptives", "10_smiles_representative_days_2025.csv")
CHECKS_SURFACE = ("10_surface_descriptives", "10_surface_descriptives_checks_summary_2025.csv")
PROFILE_SUMMARY = ("10_surface_descriptives", "10_gamma_time_profile_checks_summary_2025.csv")
PROFILE_BY_DAY = ("10_surface_descriptives", "10_gamma_time_profile_checks_by_day_2025.csv")
VENDOR_AGG = ("10_surface_descriptives", "10_vendor_vs_own_agg_2025.csv")
POINTS_HEATMAP = ("10_surface_descriptives", "10_iv_surface_points_repdays_2025.csv")
SPACING_NEAR_F = ("09_appendix_a_checks", "09_strike_spacing_near_forward_2025.csv")

OUT_T21 = "10_table_2_1_2025.csv"
OUT_T22 = "10_table_2_2_2025.csv"
OUT_HALF_SD = "10_half_sd_strike_count_by_time_2025.csv"
OUT_TEXT = "10_text_numbers_2025.csv"
OUT_CHECK = "10_thesis_check.csv"

REP_DAYS = {"median_RV": "2025-09-25", "p90_RV": "2025-04-15", "max_RV": "2025-04-07"}
TABLE_TIMES = ["09:40:00", "12:00:00", "15:00:00", "15:50:00"]
Z_RANGE = (-8.0, 5.0)                  # intervallo della Fig. 2.1 (17_Figures.py)
JT_BENCHMARK_SD = 0.35                 # Jiang e Tian (2005), Sezione 3.1
STRIKE_SPACING = 5.0
# "Strikes 5 points apart near the money" e' confermato se fra gli 8 strike quotati piu' vicini
# al forward la distanza non scende mai sotto 5 punti e vale 5 ovunque in almeno il 99% dei
# punti (seduta x minuto); le eccezioni sono elencate nella nota.
SPACING_SHARE_MIN = 0.99
SEVERAL_MIN = 2.0                      # "several times" letto come almeno 2 volte

# Fasce con segno -> fasce di |z| della Tabella 2.2
FOLD = {"(-0.5,0]": "0 to 0.5", "(0,0.5]": "0 to 0.5",
        "(-1,-0.5]": "0.5 to 1", "(0.5,1]": "0.5 to 1",
        "(-2,-1]": "1 to 2", "(1,2]": "1 to 2",
        "(-3,-2]": "2 to 3", "(2,3]": "2 to 3",
        "(-inf,-3]": "Beyond 3", "(3,inf)": "Beyond 3"}
ABS_BUCKETS = ["0 to 0.5", "0.5 to 1", "1 to 2", "2 to 3", "Beyond 3"]
NEAR_BUCKETS = ["(-0.5,0]", "(0,0.5]"]
SOURCE_NAMES = {"Vendor": "CBOE_gamma", "Black76": "Black76_gamma"}

# ----- Valori come stampati nella tesi -----
# Tabella 2.1: (riga, campione, colonna, decimali) -> valori a 09:40, 12:00, 15:00, 15:50
THESIS_T21 = [
    ("IV_ATM, 247 regular sessions", "all", "iv_atm_median", 3, [0.220, 0.189, 0.201, 0.300]),
    ("IV_ATM, 226 sessions outside April", "non_april", "iv_atm_median", 3, [0.209, 0.183, 0.189, 0.280]),
    ("IV_ATM, 21 April sessions", "april", "iv_atm_median", 3, [0.511, 0.472, 0.444, 0.640]),
    ("Skew, IV(-1) - IV(+1)", "all", "skew_median", 3, [0.049, 0.038, 0.021, 0.015]),
    ("Curvature, IV(-1) + IV(+1) - 2 IV_ATM", "all", "curvature_median", 3, [0.009, 0.016, 0.022, 0.024]),
    ("Usable OTM puts", "all", "n_otm_put_usable_median", 0, [62, 48, 21, 6]),
    ("Usable OTM calls", "all", "n_otm_call_usable_median", 0, [25, 20, 10, 4]),
    ("One standard deviation, index points", "all", "sd_points_median", 1, [36.9, 25.5, 13.2, 8.2]),
]
THESIS_T21_MINIMA = [("all", 0.180, "14:10"), ("non_april", 0.175, "13:40"), ("april", 0.427, "14:50")]
THESIS_T21_N_1550 = 246
THESIS_T21_SIZES = {"all": 247, "non_april": 226, "april": 21}

# Tabella 2.2: (pannello, fonte, ora) -> valori per 0-0.5, 0.5-1, 1-2, 2-3, oltre 3
THESIS_T22 = {
    ("A", "Vendor", "09:40:00"): [0.352, 0.278, 0.256, 0.068, 0.046],
    ("A", "Vendor", "15:50:00"): [0.290, 0.226, 0.232, 0.090, 0.161],
    ("A", "Black76", "09:40:00"): [0.350, 0.277, 0.259, 0.069, 0.045],
    ("A", "Black76", "15:50:00"): [0.313, 0.248, 0.253, 0.079, 0.107],
    ("B", "Vendor", "09:40:00"): [0.437, 0.266, 0.251, 0.037, 0.009],
    ("B", "Vendor", "15:50:00"): [0.477, 0.303, 0.188, 0.026, 0.006],
    ("B", "Black76", "09:40:00"): [0.437, 0.265, 0.251, 0.037, 0.009],
    ("B", "Black76", "15:50:00"): [0.476, 0.305, 0.190, 0.025, 0.005],
}
THESIS_T22_STRIKES = {("Vendor", "09:40:00"): 4, ("Vendor", "15:50:00"): 1,
                      ("Black76", "09:40:00"): 4, ("Black76", "15:50:00"): 1}
PANEL_COL = {"A": "share_gamma_oi", "B": "share_gamma_vol"}

THESIS_TEXT = {
    "sd_points_0940": 37, "sd_points_1550": 8,
    "atm_factor": 6,
    "smile_strikes_outside": 27, "smile_strikes_total": 959,
    "stress_wing_max": 3.4, "stress_wing_time": "10:40", "stress_12_outer": 2.7,
    "d1_term_0940": 0.003,
    "half_sd_morning": 4, "half_sd_afternoon": 1,
    "wing_ratio_1550": 1.65,
    "hoi_median_b76": 0.0661, "hoi_median_vendor": 0.0658, "hoi_corr": 0.998,
}


def fmt_dec(x, d):
    return f"{x:.{d}f}"


def same_dec(x, t, d):
    """Uguale alla precisione stampata (confronto fra stringhe arrotondate)."""
    return fmt_dec(float(x), d) == fmt_dec(float(t), d)


def hhmm(t):
    return str(t)[:5]


def surface_text_numbers():
    t0 = time.time()
    print("=" * 140)
    print("PARTE 5: SURFACE TEXT NUMBERS: Tabelle 2.1 e 2.2 e numeri del testo del Capitolo 2")
    print("=" * 140)

    for pair in (MFIV_EOS, DAILY_PANEL, SURF_AGG, SURF_MIN, GAM_AGG, GAM_SHARES, SMILES, CHECKS_SURFACE,
                 PROFILE_SUMMARY, PROFILE_BY_DAY, VENDOR_AGG, POINTS_HEATMAP, SPACING_NEAR_F):
        require(path(pair))

    surf_agg = pd.read_csv(path(SURF_AGG))
    surf_min = pd.read_csv(path(SURF_MIN))
    gam_agg = pd.read_csv(path(GAM_AGG))
    gam_sh = pd.read_csv(path(GAM_SHARES))
    smiles = pd.read_csv(path(SMILES))
    checks21 = pd.read_csv(path(CHECKS_SURFACE)).iloc[0]      # della parte 1
    prof_sum = pd.read_csv(path(PROFILE_SUMMARY)).set_index("gamma_source")
    prof_day = pd.read_csv(path(PROFILE_BY_DAY))
    vendor_agg = pd.read_csv(path(VENDOR_AGG))
    pts = pd.read_csv(path(POINTS_HEATMAP))
    spacing = pd.read_csv(path(SPACING_NEAR_F))
    panel = pd.read_csv(path(DAILY_PANEL))
    eos = pd.read_csv(path(MFIV_EOS))

    if surf_min["quote_date"].nunique() != EXPECTED_REGULAR:
        raise CheckFailed(f"10_surface_by_minute: {surf_min['quote_date'].nunique()} sedute, attese {EXPECTED_REGULAR}")
    if set(surf_min["quote_date"]) & EARLY_CLOSES:
        raise CheckFailed("10_surface_by_minute contiene chiusure anticipate")

    check, text = [], []

    def add(pagina, voce, tesi, calc, precisione, ok, nota=""):
        check.append({"pagina": pagina, "voce": voce, "valore_tesi": tesi,
                      "valore_calcolato": calc, "precisione": precisione,
                      "esito": "OK" if ok else "DIVERSO", "nota": nota})

    def note(voce, valore, fonte):
        text.append({"voce": voce, "valore": valore, "fonte_e_definizione": fonte})

    # --- p. 7: STRIP NON VALIDA VICINO ALLA CHIUSURA; DISTANZA FRA STRIKE IN SD
    eos_reg = eos[~eos["quote_date"].isin(EARLY_CLOSES)]
    if len(eos_reg) != EXPECTED_REGULAR:
        raise CheckFailed(f"03 end of session: {len(eos_reg)} sedute regolari")
    n_fail = int((eos_reg["mfiv_flag"] != "ok").sum())
    note("sedute regolari con strip non valida all'ultimo minuto (15:55)", n_fail,
         "03_mfiv_end_of_session_2025_all.csv, mfiv_flag != ok")
    add(7, "near the close the strip fails its validity rule on most sessions", "most sessions",
        f"{n_fail}/{len(eos_reg)}", "> 50% delle sedute", n_fail > len(eos_reg) / 2, "alle 15:55")

    sd = surf_agg[surf_agg["sample"] == "all"].set_index("grid_time")["sd_points_median"]
    spacing_sd_1500 = STRIKE_SPACING / sd["15:00:00"]
    note("distanza di 5 punti in deviazioni standard alle 15:00", spacing_sd_1500,
         "5 / sd_points_median (21_surface_agg, all)")
    add(7, "strike spacing exceeds the numerical benchmark (0.35 SD) of Section 3.1", "> 0.35 SD",
        round(spacing_sd_1500, 3), "confronto", spacing_sd_1500 > JT_BENCHMARK_SD,
        f"5 punti / {sd['15:00:00']:.1f} punti alle 15:00")

    # --- p. 10: UNA DEVIAZIONE STANDARD; DISTANZA FRA STRIKE
    for t, key in (("09:40:00", "sd_points_0940"), ("15:50:00", "sd_points_1550")):
        add(10, f"one standard deviation, median index points at {hhmm(t)}", THESIS_TEXT[key],
            round(float(sd[t]), 3), "intero", same_dec(sd[t], THESIS_TEXT[key], 0))

    sp_reg = spacing[~spacing["quote_date"].isin(EARLY_CLOSES)]
    if sp_reg["quote_date"].nunique() != EXPECTED_REGULAR:
        raise CheckFailed("09_strike_spacing: numero di sedute regolari inatteso")
    all5 = (sp_reg["gap_min"] == STRIKE_SPACING) & (sp_reg["gap_max"] == STRIKE_SPACING)
    share5 = float(all5.mean())
    never_finer = bool((sp_reg["gap_min"] == STRIKE_SPACING).all())
    exc = sp_reg[~all5].groupby("quote_date").agg(n=("grid_time", "size"), gmax=("gap_max", "max"),
                                                  t0=("grid_time", "min"))
    exc_txt = "; ".join(f"{d} da {hhmm(r.t0)} ({r.n} punti, distanza max {r.gmax:g})" for d, r in exc.iterrows())
    note("quota di (seduta, punto di griglia) con strike a 5 punti fra gli 8 piu' vicini al forward",
         share5, "09_strike_spacing_near_forward_2025.csv, gap_min = gap_max = 5, sedute regolari; "
         f"eccezioni: {exc_txt or 'nessuna'}")
    add(10, "strikes 5 points apart near the money", 5,
        f"{share5:.4f} dei {len(sp_reg)} punti con tutte le distanze = 5; distanza minima sempre 5: {never_finer}",
        f"distanza minima 5 ovunque e quota >= {SPACING_SHARE_MIN:.0%}",
        never_finer and share5 >= SPACING_SHARE_MIN,
        "4 strike quotati per lato attorno al forward, 38 punti di griglia, 247 sedute; "
        f"eccezioni (uno strike non quotato vicino al forward): {exc_txt or 'nessuna'}")

    # --- p. 11: TABELLA 2.1
    t21_rows = []
    for label, sample, col, dec, vals in THESIS_T21:
        s = surf_agg[surf_agg["sample"] == sample].set_index("grid_time")
        for t, tv in zip(TABLE_TIMES, vals):
            cv = float(s.loc[t, col])
            t21_rows.append({"riga": label, "campione": sample, "grid_time": t, "valore": cv})
            add(11, f"Tab. 2.1 {label}, {hhmm(t)}", tv, round(cv, 6), f"{dec} decimali",
                same_dec(cv, tv, dec))
    for sample, tv, tt in THESIS_T21_MINIMA:
        s = surf_agg[surf_agg["sample"] == sample]
        i = s["iv_atm_median"].idxmin()
        cv, ct = float(s.loc[i, "iv_atm_median"]), hhmm(s.loc[i, "grid_time"])
        add(11, f"Tab. 2.1 nota: minimo di IV_ATM ({sample})", f"{tv:.3f} at {tt}",
            f"{cv:.6f} at {ct}", "3 decimali e minuto", same_dec(cv, tv, 3) and ct == tt)
        t21_rows.append({"riga": "minimo IV_ATM", "campione": sample, "grid_time": ct + ":00", "valore": cv})
    s = surf_agg[surf_agg["sample"] == "all"].set_index("grid_time")
    n_sk, n_cu = int(s.loc["15:50:00", "n_skew"]), int(s.loc["15:50:00", "n_curvature"])
    add(11, "Tab. 2.1 nota: skew e curvature alle 15:50 su 246 sedute", THESIS_T21_N_1550,
        f"{n_sk}/{n_cu}", "intero", n_sk == n_cu == THESIS_T21_N_1550)
    for sample, n in THESIS_T21_SIZES.items():
        nd = sorted(surf_agg.loc[surf_agg["sample"] == sample, "n_days"].unique())
        add(11, f"Tab. 2.1 numero di sedute ({sample})", n, ";".join(map(str, nd)), "intero", nd == [n])

    # --- p. 11: SEDUTE RAPPRESENTATIVE E LORO SMILE
    reg = panel[panel["is_early_close"] == 0].sort_values(["RV_t", "quote_date"], kind="mergesort").reset_index(drop=True)
    if len(reg) != EXPECTED_REGULAR:
        raise CheckFailed(f"pannello giornaliero: {len(reg)} sedute regolari")
    rank_med, rank_p90 = math.ceil(0.5 * len(reg)), math.ceil(0.9 * len(reg))
    picked = {"median_RV": (reg.loc[rank_med - 1, "quote_date"], rank_med),
              "p90_RV": (reg.loc[rank_p90 - 1, "quote_date"], rank_p90),
              "max_RV": (reg.loc[len(reg) - 1, "quote_date"], len(reg))}
    for role, (d, rk) in picked.items():
        in21 = str(checks21.get(f"rep_{role}" if role != "max_RV" else "rep_stress_2025_04_07", ""))
        note(f"seduta rappresentativa {role}", f"{d} (rango {rk} di {len(reg)})",
             "07_daily_regression_panel_2025.csv, RV_t, nearest rank; parte 1: " + in21)
        add(11, f"seduta scelta per RV giornaliera: {role}", REP_DAYS[role], f"{d} (rango {rk})", "data",
            d == REP_DAYS[role] and in21.startswith(REP_DAYS[role]))

    rep = surf_min[surf_min["quote_date"].isin(REP_DAYS.values())].copy()
    rep["call_wing_net"] = rep["iv_zp1"] - rep["iv_atm"]
    atm = rep.pivot(index="grid_time", columns="quote_date", values="iv_atm")
    cw = rep.pivot(index="grid_time", columns="quote_date", values="call_wing_net")
    smile_atm = smiles.groupby(["quote_date", "grid_time"])["iv_atm"].first().unstack("quote_date")
    if (smile_atm.loc[SMILE_TIMES, list(REP_DAYS.values())] - atm.loc[SMILE_TIMES, list(REP_DAYS.values())]).abs().max().max() > 1e-12:
        raise CheckFailed("IV_ATM dello smile (parte 1) diversa da quella di 10_surface_by_minute")
    for t in SMILE_TIMES:
        ratio = float(atm.loc[t].max() / atm.loc[t].min())
        note(f"rapporto max/min IV_ATM fra le tre sedute alle {hhmm(t)}", ratio,
             "10_surface_by_minute_2025.csv, iv_atm")
        add(11, f"ATM level differs by a factor of about six ({hhmm(t)})", THESIS_TEXT["atm_factor"],
            round(ratio, 2), "intero", round(ratio) == THESIS_TEXT["atm_factor"],
            f"{atm.loc[t].idxmax()} / {atm.loc[t].idxmin()}")
    for role, d in REP_DAYS.items():
        a0, a1 = float(atm.loc["15:00:00", d]), float(atm.loc["15:50:00", d])
        add(11, f"IV_ATM rises in the last fifty minutes: {d}", "sale", f"{a0:.3f} -> {a1:.3f}",
            "15:00 -> 15:50", a1 > a0)
    for role, d in REP_DAYS.items():
        c0, c1 = float(cw.loc["09:40:00", d]), float(cw.loc["15:00:00", d])
        add(11, f"call wing net of level rises 09:40 -> 15:00: {d}", "sale", f"{c0:+.4f} -> {c1:+.4f}",
            "IV(+1) - IV_ATM", c1 > c0)
    allcw = surf_min.assign(cwn=surf_min["iv_zp1"] - surf_min["iv_atm"]).groupby("grid_time")["cwn"].median()
    add(11, "call wing net of level rises 09:40 -> 15:00: median across sessions", "sale",
        f"{allcw['09:40:00']:+.4f} -> {allcw['15:00:00']:+.4f}", "IV(+1) - IV_ATM, mediana",
        allcw["15:00:00"] > allcw["09:40:00"])

    # --- p. 12: FIG. 2.1 E SUPERFICIE DEL 7 APRILE
    n_smile = len(smiles)
    n_out = int(((smiles["z"] < Z_RANGE[0]) | (smiles["z"] > Z_RANGE[1])).sum())
    n_pts_smile = int(pts["grid_time"].isin(SMILE_TIMES).sum())
    note("strike OTM utilizzabili negli smile della Fig. 2.1", n_smile,
         f"10_smiles_representative_days_2025.csv; 24 ai tre minuti: {n_pts_smile}")
    note("di cui fuori da z in [-8, 5]", n_out, "10_smiles_representative_days_2025.csv, z")
    add(12, "Fig. 2.1: usable OTM strikes in the smiles", THESIS_TEXT["smile_strikes_total"], n_smile, "intero",
        n_smile == THESIS_TEXT["smile_strikes_total"] == n_pts_smile)
    add(12, "Fig. 2.1: of which outside z in [-8, 5]", THESIS_TEXT["smile_strikes_outside"], n_out, "intero",
        n_out == THESIS_TEXT["smile_strikes_outside"])

    st = pts[(pts["quote_date"] == STRESS_DAY) & (pts["z"] >= Z_RANGE[0]) & (pts["z"] <= Z_RANGE[1])]
    i = st["iv"].idxmax()
    top = st.loc[i]
    n_times_top = int(st.loc[st["strike"] == top["strike"], "grid_time"].nunique())
    s12 = st[(st["grid_time"] == "12:00:00") & (st["option_type"] == "P")].sort_values("z")
    outer12 = s12.iloc[0]
    note("7 aprile: punto piu' alto della superficie (z in [-8, 5])",
         f"{top['iv']:.4f} alle {hhmm(top['grid_time'])}, strike {top['strike']:.0f} {top['option_type']}, "
         f"z {top['z']:.2f}, utilizzabile in {n_times_top} punto/i di griglia",
         "10_iv_surface_points_repdays_2025.csv")
    note("7 aprile: strike piu' esterno dello smile delle 12:00",
         f"{outer12['iv']:.4f}, strike {outer12['strike']:.0f} P, z {outer12['z']:.2f}",
         "10_iv_surface_points_repdays_2025.csv")
    add(12, "7 April: far put wing reaches 3.4 at 10:40", f"{THESIS_TEXT['stress_wing_max']} at "
        f"{THESIS_TEXT['stress_wing_time']}", f"{top['iv']:.4f} at {hhmm(top['grid_time'])}",
        "1 decimale e minuto", same_dec(top["iv"], THESIS_TEXT["stress_wing_max"], 1)
        and hhmm(top["grid_time"]) == THESIS_TEXT["stress_wing_time"])
    add(12, "7 April: on a strike usable at that grid point only", 1, n_times_top, "intero",
        n_times_top == 1, f"strike {top['strike']:.0f} {top['option_type']}")
    add(12, "7 April: outermost strike of the 12:00 smile", THESIS_TEXT["stress_12_outer"],
        round(float(outer12["iv"]), 4), "1 decimale", same_dec(outer12["iv"], THESIS_TEXT["stress_12_outer"], 1),
        f"strike {outer12['strike']:.0f} P")

    # --- p. 14: d1 E "SEVERAL TIMES"
    d1 = surf_min.assign(t=surf_min["sd_k"] / 2.0).groupby("grid_time")["t"].median()
    note("secondo termine di d1 (mediana di sd_k / 2) alle 09:40", float(d1["09:40:00"]),
         "10_surface_by_minute_2025.csv, sd_k = IV_ATM * sqrt(tau)")
    add(14, "d1 second term at 09:40", THESIS_TEXT["d1_term_0940"], round(float(d1["09:40:00"]), 6), "3 decimali",
        same_dec(d1["09:40:00"], THESIS_TEXT["d1_term_0940"], 3))
    later = d1.drop("09:40:00")
    add(14, "d1 second term smaller after 09:40", "minore", f"max dopo 09:40 = {later.max():.5f}", "confronto",
        bool((later < d1["09:40:00"]).all()))
    for src_lab, src in SOURCE_NAMES.items():
        ratio = float(prof_sum.loc[src, "ratio_median_ok_regular"])
        lvl = float(prof_sum.loc[src, "profile_level_close_over_median"])
        note(f"H_OI alla chiusura / mediana giornaliera, mediana sulle sedute ({src_lab})", ratio,
             "10_gamma_time_profile_checks_summary_2025.csv, ratio_median_ok_regular; "
             f"livello del profilo alla chiusura / mediana: {lvl:.3f}")
        add(14, f"H_OI several times more concentrated near expiry ({src_lab})", "several times",
            round(ratio, 2), f">= {SEVERAL_MIN:g} volte", ratio >= SEVERAL_MIN,
            "mediana sulle 247 sedute di H_OI(15:55) / mediana giornaliera di H_OI")

    # --- p. 15: STRIKE NELLA FASCIA DI MEZZA DEVIAZIONE STANDARD
    hs = gam_agg[gam_agg["bucket"].isin(NEAR_BUCKETS)].pivot_table(
        index="grid_time", columns=["gamma_source", "bucket"], values="n_strikes_oi_median")
    hs_out = hs.copy()
    hs_out.columns = [f"{a}_{b}" for a, b in hs_out.columns]
    hs_out = hs_out.reset_index()
    both_max = hs.max(axis=1)                 # entrambe le fasce, entrambe le fonti
    first_le2 = hhmm(both_max[both_max <= 2].index.min())
    first_1 = hhmm(both_max[both_max <= 1].index.min())
    stays_1 = bool((both_max.loc[both_max[both_max <= 1].index.min():] == 1).all())
    note("fascia di mezza SD: prima ora con mediana <= 2 strike in entrambe le fasce e fonti", first_le2,
         "10_gamma_moneyness_agg_2025.csv, n_strikes_oi_median")
    note("fascia di mezza SD: prima ora con mediana 1 strike in entrambe le fasce e fonti", first_1,
         f"idem; poi resta 1 fino alle 15:50: {stays_1}")
    v0940, v1550 = hs.loc["09:40:00"], hs.loc["15:50:00"]
    add(15, "half-SD bucket: median strikes in the morning (09:40)", THESIS_TEXT["half_sd_morning"],
        ";".join(f"{x:g}" for x in v0940), "intero",
        bool((v0940 == THESIS_TEXT["half_sd_morning"]).all()), "entrambe le fasce e fonti")
    add(15, "half-SD bucket: only one by the afternoon (15:50)", THESIS_TEXT["half_sd_afternoon"],
        ";".join(f"{x:g}" for x in v1550), "intero",
        bool((v1550 == THESIS_TEXT["half_sd_afternoon"]).all()),
        f"scende a 2 alle {first_le2} e a 1 alle {first_1} (entrambe le fasce e fonti)")

    # --- p. 16: TABELLA 2.2
    sh = gam_sh[gam_sh["grid_time"].isin(["09:40:00", "15:50:00"])].copy()
    if sh.groupby(["quote_date", "grid_time", "gamma_source"]).size().ne(10).any():
        raise CheckFailed("10_gamma_moneyness_shares: non tutte le 10 fasce presenti per giornata")
    sh["abs_bucket"] = sh["bucket"].map(FOLD)
    if sh["abs_bucket"].isna().any():
        raise CheckFailed(f"fasce sconosciute: {sorted(sh.loc[sh['abs_bucket'].isna(), 'bucket'].unique())}")
    per_day = sh.groupby(["quote_date", "grid_time", "gamma_source", "abs_bucket"])[
        ["share_gamma_oi", "share_gamma_vol"]].sum()
    fold_day = per_day.groupby(["grid_time", "gamma_source", "abs_bucket"]).mean()
    ga = gam_agg[gam_agg["grid_time"].isin(["09:40:00", "15:50:00"])].copy()
    ga["abs_bucket"] = ga["bucket"].map(FOLD)
    fold_agg = ga.groupby(["grid_time", "gamma_source", "abs_bucket"])[
        ["share_gamma_oi_mean", "share_gamma_vol_mean"]].sum()
    n_days_t22 = sorted(sh.groupby(["grid_time", "gamma_source"])["quote_date"].nunique().unique())

    t22_rows = []
    max_dev = 0.0
    for (panel_id, src_lab, t), vals in THESIS_T22.items():
        src = SOURCE_NAMES[src_lab]
        col = PANEL_COL[panel_id]
        for b, tv in zip(ABS_BUCKETS, vals):
            v_day = float(fold_day.loc[(t, src, b), col])
            v_agg = float(fold_agg.loc[(t, src, b), col + "_mean"])
            max_dev = max(max_dev, abs(v_day - v_agg))
            t22_rows.append({"panel": panel_id, "abs_z_bucket": b, "gamma_source": src_lab,
                             "grid_time": t, "mean_share": v_day, "sum_of_signed_bucket_means": v_agg,
                             "valore_tesi": tv})
            add(16, f"Tab. 2.2 panel {panel_id} {src_lab} {hhmm(t)} |z| {b}", tv, round(v_day, 6), "3 decimali",
                same_dec(v_day, tv, 3))
    if max_dev > 1e-12:
        raise CheckFailed(f"Tabella 2.2: piegatura per giornata e somma delle medie differiscono di {max_dev:.2e}")
    for (src_lab, t), tv in THESIS_T22_STRIKES.items():
        sub = gam_agg[(gam_agg["gamma_source"] == SOURCE_NAMES[src_lab]) & (gam_agg["grid_time"] == t)
                      & (gam_agg["bucket"].isin(NEAR_BUCKETS))]
        vals = sorted(sub["n_strikes_oi_median"].unique())
        t22_rows.append({"panel": "A", "abs_z_bucket": "strikes per near-money bucket", "gamma_source": src_lab,
                         "grid_time": t, "mean_share": np.nan, "sum_of_signed_bucket_means": np.nan,
                         "valore_tesi": tv, "n_strikes_median": ";".join(f"{x:g}" for x in vals)})
        add(16, f"Tab. 2.2 strikes per near-money bucket, {src_lab} {hhmm(t)}", tv,
            ";".join(f"{x:g}" for x in vals), "intero", vals == [tv], "fasce (-0.5,0] e (0,0.5]")
    note("Tabella 2.2: sedute per cella", ";".join(map(str, n_days_t22)), "10_gamma_moneyness_shares_2025.csv")

    # --- p. 17: ALA PUT LONTANA, MEDIANE DI H_OI, CORRELAZIONE
    w = vendor_agg[(vendor_agg["side"] == "all") & (vendor_agg["grid_time"] == "15:50:00")
                   & (vendor_agg["bucket"] == "(-inf,-3]")]
    if len(w) != 1:
        raise CheckFailed("10_vendor_vs_own_agg: riga side=all, 15:50, (-inf,-3] non unica")
    wr = float(w["exposure_ratio_median"].iloc[0])
    note("ala put lontana (z <= -3) alle 15:50: esposizione gamma x OI vendor / Black '76", wr,
         "10_vendor_vs_own_agg_2025.csv, exposure_ratio_median, side all (mediana su "
         f"{int(w['exposure_ratio_n'].iloc[0])} sedute)")
    add(17, "far put wing (z < -3) vendor / Black '76 gamma x OI exposure at 15:50", THESIS_TEXT["wing_ratio_1550"],
        round(wr, 4), "2 decimali", same_dec(wr, THESIS_TEXT["wing_ratio_1550"], 2))

    pd_reg = prof_day[prof_day["is_early_close"] == 0]
    hp = pd_reg.pivot(index="quote_date", columns="gamma_source", values="h_oi_median_ok")
    if len(hp) != EXPECTED_REGULAR or hp.isna().any().any():
        raise CheckFailed("10_gamma_time_profile_checks_by_day: serie giornaliere di H_OI incomplete")
    for src_lab, src, key in (("Black76", "Black76_gamma", "hoi_median_b76"), ("Vendor", "CBOE_gamma", "hoi_median_vendor")):
        v_sum = float(prof_sum.loc[src, "median_of_daily_ok_median_h_oi"])
        v_day = float(hp[src].median())
        if abs(v_sum - v_day) > 1e-15:
            raise CheckFailed(f"mediana di H_OI {src_lab}: summary {v_sum} contro by_day {v_day}")
        note(f"mediana sulle sedute della mediana giornaliera di H_OI ({src_lab})", v_sum,
             "10_gamma_time_profile_checks_summary_2025.csv, median_of_daily_ok_median_h_oi")
        add(17, f"median of the daily-median H_OI ({src_lab})", THESIS_TEXT[key], round(v_sum, 6), "4 decimali",
            same_dec(v_sum, THESIS_TEXT[key], 4))
    corr = float(hp["Black76_gamma"].corr(hp["CBOE_gamma"]))
    note("correlazione (Pearson) delle serie giornaliere di H_OI, Black '76 / vendor", corr,
         "10_gamma_time_profile_checks_by_day_2025.csv, h_oi_median_ok, 247 sedute regolari")
    add(17, "the two daily H_OI series correlate at", THESIS_TEXT["hoi_corr"], round(corr, 5), "3 decimali",
        same_dec(corr, THESIS_TEXT["hoi_corr"], 3), "Pearson, 247 sedute regolari")

    # --- scrittura
    check = pd.DataFrame(check)
    pd.DataFrame(t21_rows).to_csv(OUTPUT_DIR / OUT_T21, index=False)
    pd.DataFrame(t22_rows).to_csv(OUTPUT_DIR / OUT_T22, index=False)
    hs_out.to_csv(OUTPUT_DIR / OUT_HALF_SD, index=False)
    pd.DataFrame(text).to_csv(OUTPUT_DIR / OUT_TEXT, index=False)
    check.to_csv(OUTPUT_DIR / OUT_CHECK, index=False)

    # --- stampa (le opzioni di visualizzazione valgono solo qui)
    with pd.option_context("display.width", 220, "display.max_colwidth", 110):
        print()
        print("NUMERI DEL TESTO")
        print(pd.DataFrame(text)[["voce", "valore"]].to_string(index=False))
        print()
        n_ok = int((check["esito"] == "OK").sum())
        n_diff = int((check["esito"] == "DIVERSO").sum())
        print("CONTROLLO CONTRO LA TESI (solo le voci fuori dalle Tabelle 2.1 e 2.2; le tabelle sono nel CSV)")
        tab = check["voce"].str.startswith("Tab. ")
        print(check.loc[~tab, ["pagina", "voce", "valore_tesi", "valore_calcolato", "esito"]].to_string(index=False))
        print(f"  Tabelle 2.1 e 2.2: {int((check.loc[tab, 'esito'] == 'OK').sum())} OK, "
              f"{int((check.loc[tab, 'esito'] == 'DIVERSO').sum())} DIVERSO su {int(tab.sum())} celle")
        for _, r in check[check["esito"] == "DIVERSO"].iterrows():
            print(f"  DIVERSO p. {r['pagina']}: {r['voce']}: tesi {r['valore_tesi']}, "
                  f"calcolato {r['valore_calcolato']}. {r['nota']}")
        print()
        print(f"Output in {OUTPUT_DIR}")
        print(f"Tempo: {time.time() - t0:.1f} secondi")
        print(f"VERDETTO: {n_ok} OK, {n_diff} DIVERSO su {len(check)} voci della tesi.")
        print("=" * 140)


def main():
    surface_descriptives()
    gamma_time_profile()
    vendor_comparison()
    surface_iv_heatmap()
    surface_text_numbers()


if __name__ == "__main__":
    main()
