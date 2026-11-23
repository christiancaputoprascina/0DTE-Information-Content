"""
Figure della tesi in PDF (per il LaTeX) e in PNG, costruite solo dai CSV degli script precedenti,
senza stime nuove. Prima di disegnare ogni figura controlla i dati e, dove puo', che riproducano
numeri gia' salvati (pendenze di Mincer-Zarnowitz, statistiche del log-rapporto, punti dei
sorrisi); se un controllo fallisce lo script si ferma. L'elenco delle figure e' in FIGURES.
Input:  CBOE/Data/07_daily_panel/, 10_surface_descriptives/, 14_daily_mz_robustness/,
        16_intraday_robustness/
Output: CBOE/Data/17_figures/
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATA_DIR, EARLY_CLOSE_DATES, FOMC_DATES, output_dir

OUTPUT_DIR = output_dir("17_figures")


# =====================================================================
# FIGURE DELLA TESI
# =====================================================================
# Palette unica: blu = fonte della tesi / sessioni normali / mattina; rosso = fonte del fornitore
# / aprile / chiusura. Le mappe di calore usano una scala sequenziale dal blu chiaro al rosso
# scuro, con luminosita' monotona (leggibile anche in bianco e nero): rosso = valore alto.
from matplotlib.colors import LinearSegmentedColormap
BLUE, PURPLE, RED = "#1f4e79", "#7b4a9b", "#b2182b"
BR_SEQ = LinearSegmentedColormap.from_list(
    "blue_red_seq", ["#f0f5fa", "#b3cde8", "#7a9ed1", "#9a6bb0", "#bd3440", "#7a0b18"])

SOURCES = [("CBOE_gamma", "Vendor (Cboe) gamma", RED, "-"),
           ("Black76_gamma", "Black '76 gamma", BLUE, "--")]

# Impostazioni grafiche comuni a tutte le figure.
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9,
                     "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def minutes_since_open(t: pd.Series) -> np.ndarray:
    parts = t.str.split(":", expand=True).astype(int)
    return (parts[0] * 60 + parts[1] - (9 * 60 + 30)).to_numpy()


def time_axis(ax):
    ticks = [0, 90, 180, 270, 360]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["09:30", "11:00", "12:30", "14:00", "15:30"])
    ax.set_xlim(0, 390)
    ax.set_xlabel("Time of day (New York)")


def check_profile(p: pd.DataFrame):
    need = {"gamma_source", "quote_time", "n_days_h_oi", "h_oi_p25", "h_oi_median", "h_oi_p75",
            "n_days_h_vol", "h_vol_p25", "h_vol_median", "h_vol_p75"}
    missing = need - set(p.columns)
    assert not missing, f"colonne mancanti: {missing}"
    counts = p.groupby("gamma_source").size().to_dict()
    assert counts == {"Black76_gamma": 385, "CBOE_gamma": 385}, counts
    assert p["quote_time"].min() == "09:31:00" and p["quote_time"].max() == "15:55:00"
    assert ((p["h_oi_p25"] <= p["h_oi_median"]) & (p["h_oi_median"] <= p["h_oi_p75"])).all()
    assert ((p["h_vol_p25"] <= p["h_vol_median"]) & (p["h_vol_median"] <= p["h_vol_p75"])).all()


def figure_b1(csv_dir: Path, out_dir: Path):
    p = pd.read_csv(csv_dir / "10_gamma_time_profile_2025.csv", keep_default_na=False)
    check_profile(p)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for ax, meas, title in [(axes[0], "h_oi", r"(a) $H^{OI}$ (open-interest-weighted)"),
                            (axes[1], "h_vol", r"(b) $H^{VOL}$ (volume-weighted)")]:
        g = {src: p[p["gamma_source"] == src].sort_values("quote_time") for src, *_ in SOURCES}
        x = minutes_since_open(g["Black76_gamma"]["quote_time"])
        b = g["Black76_gamma"]; c = g["CBOE_gamma"]
        ax.fill_between(x, b[f"{meas}_p25"], b[f"{meas}_p75"], color=BLUE, alpha=0.18, lw=0,
                        label="Black '76 gamma, interquartile range")
        ax.plot(x, c[f"{meas}_p25"], color=RED, lw=0.6, ls=":", label="Vendor (Cboe) gamma, interquartile range")
        ax.plot(x, c[f"{meas}_p75"], color=RED, lw=0.6, ls=":")
        ax.plot(x, b[f"{meas}_median"], color=BLUE, lw=1.2, ls="--", label="Black '76 gamma, median")
        ax.plot(x, c[f"{meas}_median"], color=RED, lw=1.0, ls="-", label="Vendor (Cboe) gamma, median")
        time_axis(ax)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, None)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        if meas == "h_oi":
            ins = ax.inset_axes([0.12, 0.48, 0.42, 0.42])
            sel = x >= 330
            ins.plot(x[sel], b[f"{meas}_median"].to_numpy()[sel], color=BLUE, lw=1.1, ls="--")
            ins.plot(x[sel], c[f"{meas}_median"].to_numpy()[sel], color=RED, lw=1.0)
            ins.set_xticks([330, 360, 385]); ins.set_xticklabels(["15:00", "15:30", "15:55"], fontsize=6)
            ins.tick_params(axis="y", labelsize=6)
            ins.set_title("medians, 15:00-15:55", fontsize=6, pad=2)
            for sp in ["top", "right"]:
                ins.spines[sp].set_visible(True)
    axes[0].set_ylabel("Herfindahl index across strikes")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"17_fig_B1_gamma_time_profile.{ext}", dpi=200)
    plt.close(fig)


BUCKETS = ["(-inf,-3]", "(-3,-2]", "(-2,-1]", "(-1,-0.5]", "(-0.5,0]",
           "(0,0.5]", "(0.5,1]", "(1,2]", "(2,3]", "(3,inf)"]
BUCKET_TICKS = ["z <= -3", "-3 to -2", "-2 to -1", "-1 to -0.5", "-0.5 to 0",
                "0 to 0.5", "0.5 to 1", "1 to 2", "2 to 3", "z > 3"]
GRID = [f"{h:02d}:{m:02d}:00" for h in range(9, 16) for m in range(0, 60, 10) if (9, 40) <= (h, m) <= (15, 50)]
ROLE_TITLES = {"median_RV": "median-RV session", "p90_RV": "90th-pct-RV session",
               "stress_2025_04_07": "stress session"}
SMILE_TIMES = [("09:40:00", BLUE, "o"), ("12:00:00", PURPLE, "s"), ("15:00:00", RED, "^")]
Z_RANGE = (-8.0, 5.0)


def need(df, cols, name):
    missing = set(cols) - set(df.columns)
    assert not missing, f"{name}: colonne mancanti: {missing}"


def grid_axis(ax):
    lab = ["10:00:00", "11:00:00", "12:00:00", "13:00:00", "14:00:00", "15:00:00"]
    ax.set_xticks([GRID.index(t) for t in lab])
    ax.set_xticklabels([t[:5] for t in lab])


def save(fig, out_dir, name, tight=True):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight" if tight else None)
    plt.close(fig)


def figure_a1(csv_dir: Path, out_dir: Path):
    s = pd.read_csv(csv_dir / "10_smiles_representative_days_2025.csv", keep_default_na=False)
    need(s, ["quote_date", "day_role", "grid_time", "k", "iv", "z", "iv_atm", "option_type"], "smiles")
    assert set(s["day_role"]) == set(ROLE_TITLES), set(s["day_role"])
    assert set(s["grid_time"]) == {t for t, _, _ in SMILE_TIMES}
    assert not (((s["k"] < 0) & (s["option_type"] != "P")) | ((s["k"] > 0) & (s["option_type"] != "C"))).any()
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6))
    n_out = 0
    for ax, role in zip(axes, ["median_RV", "p90_RV", "stress_2025_04_07"]):
        d = s[s["day_role"] == role]
        for t, col, mk in SMILE_TIMES:
            g = d[d["grid_time"] == t].sort_values("z")
            inside = g[(g["z"] >= Z_RANGE[0]) & (g["z"] <= Z_RANGE[1])]
            n_out += len(g) - len(inside)
            ax.plot(inside["z"], inside["iv"], color=col, lw=0.9, marker=mk, ms=2.2, label=t[:5])
            ax.plot([0], [g["iv_atm"].iloc[0]], marker="x", color=col, ms=5, mew=1.2)
        ax.set_xlim(*Z_RANGE)
        ax.axvline(0, color="grey", lw=0.4)
        ax.set_title(f"{ROLE_TITLES[role]}, {d['quote_date'].iloc[0]}", loc="left", fontsize=7.5)
        ax.set_xlabel("standardized moneyness z")
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Black '76 implied vol. (annualized)")
    axes[0].legend(frameon=False, title="time", title_fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, out_dir, "17_fig_A1_smiles_representative_days")
    return f"strike fuori da z in {Z_RANGE}: {n_out} su {len(s)}"


def figure_a2(csv_dir: Path, out_dir: Path):
    a = pd.read_csv(csv_dir / "10_surface_agg_2025.csv", keep_default_na=False)
    need(a, ["sample", "grid_time", "iv_atm_p25", "iv_atm_median", "iv_atm_p75", "n_days"], "surface_agg")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(GRID))
    for smp, col, ls, lab in [("non_april", BLUE, "-", "non-April sessions"), ("april", RED, "--", "April sessions")]:
        g = a[a["sample"] == smp].set_index("grid_time").loc[GRID]
        ax.fill_between(x, g["iv_atm_p25"].astype(float), g["iv_atm_p75"].astype(float), color=col, alpha=0.15, lw=0)
        ax.plot(x, g["iv_atm_median"].astype(float), color=col, ls=ls, lw=1.2, label=f"{lab} (n = {int(g['n_days'].iloc[0])})")
    grid_axis(ax); ax.set_xlabel("Time of day (New York)")
    ax.set_ylabel("ATM implied vol. (annualized)"); ax.set_ylim(0, None)
    ax.legend(frameon=False); ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    save(fig, out_dir, "17_fig_A2_iv_atm_by_time")


def figure_a3(csv_dir: Path, out_dir: Path):
    a = pd.read_csv(csv_dir / "10_surface_agg_2025.csv", keep_default_na=False)
    need(a, ["sample", "grid_time", "skew_p25", "skew_median", "skew_p75", "curvature_p25", "curvature_median", "curvature_p75"], "surface_agg")
    g = a[a["sample"] == "all"].set_index("grid_time").loc[GRID]
    x = np.arange(len(GRID))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for ax, v, title in [(axes[0], "skew", "(a) skew: IV(z=-1) - IV(z=+1)"),
                         (axes[1], "curvature", "(b) curvature: IV(z=-1) + IV(z=+1) - 2 IV_ATM")]:
        ax.fill_between(x, g[f"{v}_p25"].astype(float), g[f"{v}_p75"].astype(float), color=BLUE, alpha=0.15, lw=0)
        ax.plot(x, g[f"{v}_median"].astype(float), color=BLUE, lw=1.2)
        ax.axhline(0, color="grey", lw=0.4)
        grid_axis(ax); ax.set_xlabel("Time of day (New York)")
        ax.set_title(title, loc="left", fontsize=8); ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("implied vol. (annualized)")
    fig.tight_layout()
    save(fig, out_dir, "17_fig_A3_skew_curvature_by_time")


def moneyness_matrix(df, value, source=None):
    d = df if source is None else df[df["gamma_source"] == source]
    m = d.pivot(index="bucket", columns="grid_time", values=value)
    return m.reindex(index=BUCKETS, columns=GRID).astype(float).to_numpy()


def bucket_axis(ax):
    ax.set_yticks(range(len(BUCKETS))); ax.set_yticklabels(BUCKET_TICKS, fontsize=7)
    ax.set_ylabel("standardized moneyness z")


def heat_pair(g, value, cbar_label, name, out_dir):
    need(g, ["gamma_source", "grid_time", "bucket", value], "gamma_moneyness_agg")
    mats = {src: moneyness_matrix(g, value, src) for src in ["CBOE_gamma", "Black76_gamma"]}
    vmax = float(np.nanmax([np.nanmax(v) for v in mats.values()]))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True, constrained_layout=True)
    for ax, src, title in [(axes[0], "CBOE_gamma", "(a) Vendor (Cboe) gamma"),
                           (axes[1], "Black76_gamma", "(b) Black '76 gamma")]:
        im = ax.imshow(mats[src], aspect="auto", origin="lower", cmap=BR_SEQ, vmin=0, vmax=vmax,
                       interpolation="nearest")
        grid_axis(ax); ax.set_xlabel("Time of day (New York)")
        ax.set_title(title, loc="left", fontsize=8)
    bucket_axis(axes[0])
    cb = fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02); cb.set_label(cbar_label, fontsize=7)
    save(fig, out_dir, name)


def figure_b2(csv_dir: Path, out_dir: Path):
    g = pd.read_csv(csv_dir / "10_gamma_moneyness_agg_2025.csv", keep_default_na=False)
    heat_pair(g, "share_gamma_oi_median", "share of gamma x OI (median across sessions)",
              "17_fig_B2_gamma_oi_share_by_moneyness", out_dir)


def figure_b3(csv_dir: Path, out_dir: Path):
    g = pd.read_csv(csv_dir / "10_gamma_moneyness_agg_2025.csv", keep_default_na=False)
    heat_pair(g, "share_gamma_vol_median", "share of gamma x volume (median across sessions)",
              "17_fig_B3_gamma_vol_share_by_moneyness", out_dir)


def figure_c12(csv_dir: Path, out_dir: Path):
    from matplotlib.colors import LogNorm
    l = pd.read_csv(csv_dir / "10_liquidity_agg_2025.csv", keep_default_na=False)
    need(l, ["grid_time", "bucket", "rel_spread_median", "availability_median"], "liquidity_agg")
    rs, av = moneyness_matrix(l, "rel_spread_median"), moneyness_matrix(l, "availability_median")
    assert np.nanmin(av) >= 0 and np.nanmax(av) <= 1
    missing_quote = 1.0 - av        # rosso = meno liquido in entrambi i pannelli
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, constrained_layout=True)
    im0 = axes[0].imshow(rs, aspect="auto", origin="lower", cmap=BR_SEQ, norm=LogNorm(vmin=0.01, vmax=0.7),
                         interpolation="nearest")
    im1 = axes[1].imshow(missing_quote, aspect="auto", origin="lower", cmap=BR_SEQ, vmin=0, vmax=1,
                         interpolation="nearest")
    for ax, title in [(axes[0], "(a) median relative spread, two-sided quotes"),
                      (axes[1], "(b) share of listed OTM strikes without a two-sided quote")]:
        grid_axis(ax); ax.set_xlabel("Time of day (New York)"); ax.set_title(title, loc="left", fontsize=8)
    bucket_axis(axes[0])
    c0 = fig.colorbar(im0, ax=axes[0], fraction=0.06, pad=0.03); c0.set_label("(ask - bid) / mid", fontsize=7)
    c0.ax.tick_params(labelsize=7)
    c1 = fig.colorbar(im1, ax=axes[1], fraction=0.06, pad=0.03); c1.set_label("1 - availability", fontsize=7)
    c1.ax.tick_params(labelsize=7)
    save(fig, out_dir, "17_fig_C12_liquidity_by_moneyness")


def figure_b4(csv_dir: Path, out_dir: Path):
    """Fornitore contro Black '76, per fascia di z e minuto. Scala divergente perche' il
    rapporto ha un punto neutro, 1: blu = fornitore sotto, rosso = fornitore sopra.
    Il pannello (c) mostra la IV dei contratti ITM."""
    from matplotlib.colors import TwoSlopeNorm
    a = pd.read_csv(csv_dir / "10_vendor_vs_own_agg_2025.csv", keep_default_na=False)
    need(a, ["side", "grid_time", "bucket", "iv_ratio_median_median", "exposure_ratio_median"], "vendor_vs_own_agg")
    panels = [(moneyness_matrix(a[a["side"] == "all"], "exposure_ratio_median"),
               "(a) gamma x OI, all contracts"),
              (moneyness_matrix(a[a["side"] == "otm"], "iv_ratio_median_median"),
               "(b) implied vol., OTM contracts"),
              (moneyness_matrix(a[a["side"] == "itm"], "iv_ratio_median_median"),
               "(c) implied vol., ITM contracts")]
    br_div = LinearSegmentedColormap.from_list("blue_white_red", [BLUE, "#f7f7f7", RED])
    br_div.set_bad("#9e9e9e")      # celle senza dati: grigio, distinguibile dal bianco (= 1)
    allv = np.concatenate([m[np.isfinite(m) & (m > 0)] for m, _ in panels])
    hi = float(np.exp(min(max(np.max(np.abs(np.log(allv))), np.log(1.02)), np.log(2.0))))
    norm = TwoSlopeNorm(vmin=1.0 / hi, vcenter=1.0, vmax=hi)     # stessa scala nei tre pannelli
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True, constrained_layout=True)
    for ax, (m, title) in zip(axes, panels):
        im = ax.imshow(m, aspect="auto", origin="lower", cmap=br_div, norm=norm, interpolation="nearest")
        grid_axis(ax); ax.set_xlabel("Time of day (New York)"); ax.set_title(title, loc="left", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=6.5)
    bucket_axis(axes[0])
    cb = fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02)
    cb.set_label("Vendor / Black '76, median across sessions", fontsize=7); cb.ax.tick_params(labelsize=7)
    save(fig, out_dir, "17_fig_B4_vendor_vs_thesis_by_moneyness")


# ---------------------------------------------------------------------
# FIGURE DEI RISULTATI
# ---------------------------------------------------------------------

REGIMES = [("calmo", "calm", BLUE), ("intermedio", "intermediate", PURPLE), ("agitato", "agitated", RED)]


def figure_mz(csv_dir: Path, out_dir: Path):
    """Varianza realizzata contro implicita di sessione, per regime. Il regime (terzili della
    media di RV_t delle cinque sessioni precedenti) e' ricalcolato e deve riprodurre le
    pendenze di Mincer-Zarnowitz di 14_Daily_MZ_robustness.py, altrimenti la figura non
    viene scritta."""
    d = pd.read_csv(csv_dir / "07_daily_regression_panel_2025.csv", keep_default_na=False)
    need(d, ["quote_date", "RV_t", "MFIV_session_t"], "daily panel")
    mz_path = DATA_DIR / "14_daily_mz_robustness" / "14_regime_and_rv5_robustness_mincer_zarnowitz.csv"
    mz = pd.read_csv(mz_path, keep_default_na=False)
    need(mz, ["sample", "form", "n", "slope"], "mincer_zarnowitz")
    for c in ["RV_t", "MFIV_session_t"]:
        d[c] = pd.to_numeric(d[c].replace("", np.nan))
    d = d.sort_values("quote_date").reset_index(drop=True)
    d["regime"] = pd.qcut(d["RV_t"].rolling(5).mean().shift(1), 3, labels=[r for r, _, _ in REGIMES])
    # riscontro: le pendenze per regime devono coincidere con il file
    for r, _, _ in REGIMES:
        s = d[d["regime"] == r]
        for form in ["levels", "log"]:
            x, y = s["MFIV_session_t"], s["RV_t"]
            if form == "log":
                x, y = np.log(x), np.log(y)
            b = np.polyfit(x, y, 1)[0]
            row = mz[(mz["sample"] == r) & (mz["form"] == form)].iloc[0]
            assert len(s) == int(row["n"]), f"regime {r}: {len(s)} giorni, file {row['n']}"
            assert abs(b - float(row["slope"])) < 1e-6, f"regime {r} {form}: pendenza {b} contro {row['slope']}"
    full = mz[(mz["sample"] == "intero") & (mz["form"] == "log")].iloc[0]
    b1, b0 = np.polyfit(np.log(d["MFIV_session_t"]), np.log(d["RV_t"]), 1)
    assert abs(b1 - float(full["slope"])) < 1e-6
    april = d["quote_date"].str[5:7] == "04"

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lo = float(min(d["MFIV_session_t"].min(), d["RV_t"].min())) * 0.8
    hi = float(max(d["MFIV_session_t"].max(), d["RV_t"].max())) * 1.25
    g = np.array([lo, hi])
    ax.plot(g, g, color="grey", lw=0.8, ls="--", label="45-degree line")
    ax.plot(g, np.exp(b0) * g ** b1, color="black", lw=1.0,
            label=f"log fit, full sample (slope {float(full['slope']):.2f})")
    none = d["regime"].isna()
    ax.scatter(d.loc[none, "MFIV_session_t"], d.loc[none, "RV_t"], s=10, color="#9e9e9e",
               label="first five sessions (no regime)")
    for r, lab, col in REGIMES:
        slope = float(mz[(mz["sample"] == r) & (mz["form"] == "log")]["slope"].iloc[0])
        sel = (d["regime"] == r) & ~april
        ax.scatter(d.loc[sel, "MFIV_session_t"], d.loc[sel, "RV_t"], s=12, color=col, alpha=0.85,
                   label=f"{lab} regime (log slope {slope:.2f})")
    ax.scatter(d.loc[april, "MFIV_session_t"], d.loc[april, "RV_t"], s=22, facecolors="none",
               edgecolors=RED, linewidths=0.9, marker="^", label="April 2025 (agitated regime)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("implied variance of the session, MFIV$^{ses}$ (log scale)")
    ax.set_ylabel("realized variance of the session, RV (log scale)")
    ax.grid(lw=0.3, alpha=0.5, which="major")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, out_dir, "17_fig_R1_realized_vs_implied_by_regime")
    return f"regimi riprodotti: pendenze uguali al file; aprile {int(april.sum())} sessioni"


def figure_effects(csv_dir: Path, out_dir: Path):
    """Effetti di H_OI e H_VOL per deviazione standard, per orizzonte intraday.
    Colore = fonte del gamma (rosso fornitore, blu tesi); pieno = MFIV^ann, vuoto = MFIV
    totale. Scala verticale comune fra livelli e log nella stessa riga, perche' il
    confronto fra le ampiezze degli intervalli e' il punto della figura."""
    e = pd.read_csv(csv_dir / "16_gamma_effect_size_ci.csv", keep_default_na=False)
    need(e, ["design", "form", "regressor", "source", "horizon", "term", "effect_pct",
             "ci_low_pct", "ci_high_pct"], "gamma_effect_size_ci")
    e = e[e["design"] == "intraday"].copy()
    for c in ["horizon", "effect_pct", "ci_low_pct", "ci_high_pct"]:
        e[c] = pd.to_numeric(e[c])
    assert len(e) == 64, len(e)
    assert ((e["ci_low_pct"] <= e["effect_pct"]) & (e["effect_pct"] <= e["ci_high_pct"])).all()
    key = ["form", "regressor", "source", "horizon", "term"]
    assert not e.duplicated(key).any()
    horizons = [5.0, 15.0, 30.0, 60.0]
    combos = [("cboe", "MFIV_ann", -0.21), ("indep", "MFIV_ann", -0.07), ("cboe", "MFIV", 0.07), ("indep", "MFIV", 0.21)]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8), sharey="row", sharex=True)
    for i, term in enumerate(["H_OI", "H_VOL"]):
        for j, form in enumerate(["levels", "log"]):
            ax = axes[i, j]
            ax.axhline(0, color="grey", lw=0.6)
            for src, reg, off in combos:
                col = RED if src == "cboe" else BLUE
                sub = e[(e["term"] == term) & (e["form"] == form) & (e["source"] == src) & (e["regressor"] == reg)]
                sub = sub.set_index("horizon").loc[horizons]
                x = np.arange(len(horizons)) + off
                ax.errorbar(x, sub["effect_pct"], yerr=[sub["effect_pct"] - sub["ci_low_pct"], sub["ci_high_pct"] - sub["effect_pct"]],
                            fmt="o", ms=3.5, color=col, mfc=col if reg == "MFIV_ann" else "white", mec=col,
                            elinewidth=0.9, capsize=1.5,
                             label=("Vendor (Cboe)" if src == "cboe" else "Black '76") + " gamma, "
                                  + ("MFIV$^{ann}$" if reg == "MFIV_ann" else "total MFIV"))
            ax.set_xticks(range(len(horizons))); ax.set_xticklabels([f"{int(h)}" for h in horizons])
            ax.grid(axis="y", lw=0.3, alpha=0.5)
            tname = "$H^{OI}$" if term == "H_OI" else "$H^{VOL}$"
            ax.set_title(f"{tname}, {'levels (primary form)' if form == 'levels' else 'log (robustness form)'}",
                         loc="left", fontsize=8)
            if i == 1:
                ax.set_xlabel("forward horizon h (minutes)")
            if j == 0:
                ax.set_ylabel("effect per within-effects s.d. (%)", fontsize=7.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, out_dir, "17_fig_R2_convexity_effects_by_horizon")

def figure_s1(csv_dir: Path, out_dir: Path):
    """Superficie di volatilita' implicita 0DTE: mediana fra le 247 sessioni regolari, per
    nodo di moneyness standardizzata z e minuto di griglia.
    (a) livello: IV annualizzata, scala sequenziale (rosso = alto).
    (b) forma: IV(z) / IV_ATM, scala divergente centrata in 1 (come B4).
    Celle con valore in meno di meta' delle sessioni (coverage_ok falso): grigio."""
    from matplotlib.colors import TwoSlopeNorm
    a = pd.read_csv(csv_dir / "10_iv_surface_by_z_agg_2025.csv", keep_default_na=False)
    need(a, ["sample", "grid_time", "z", "n_sessions", "n_days", "coverage_ok",
             "iv_median", "iv_rel_median"], "iv_surface_by_z_agg")
    g = a[a["sample"] == "all"].copy()
    g["z"] = pd.to_numeric(g["z"]).round(2)
    zs = sorted(g["z"].unique())
    assert zs[0] == Z_RANGE[0] and zs[-1] == Z_RANGE[1], zs
    assert set(g["grid_time"]) == set(GRID), "minuti di griglia diversi dai 38 attesi"
    assert len(g) == len(zs) * 38, len(g)
    ok = g["coverage_ok"].astype(str).str.strip().str.lower() == "true"
    n_sessions = int(pd.to_numeric(g["n_sessions"]).iloc[0])
    mats = {}
    for v in ["iv_median", "iv_rel_median"]:
        x = pd.to_numeric(g[v], errors="coerce").where(ok)
        m = g.assign(_v=x).pivot(index="z", columns="grid_time", values="_v")
        mats[v] = m.reindex(index=zs, columns=GRID).astype(float).to_numpy()
    z0 = zs.index(0.0)
    rel0 = mats["iv_rel_median"][z0]
    assert np.all(np.abs(rel0[np.isfinite(rel0)] - 1.0) < 1e-9), "IV / IV_ATM a z = 0 diverso da 1"

    seq = BR_SEQ.copy(); seq.set_bad("#9e9e9e")
    div = LinearSegmentedColormap.from_list("blue_white_red", [BLUE, "#f7f7f7", RED])
    div.set_bad("#9e9e9e")
    lv = mats["iv_median"]
    rl = mats["iv_rel_median"]
    good = rl[np.isfinite(rl) & (rl > 0)]
    hi = float(np.exp(min(max(np.max(np.abs(np.log(good))), np.log(1.02)), np.log(3.0))))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True, constrained_layout=True)
    im0 = axes[0].imshow(lv, aspect="auto", origin="lower", cmap=seq, interpolation="nearest",
                         vmin=float(np.nanmin(lv)), vmax=float(np.nanmax(lv)))
    im1 = axes[1].imshow(rl, aspect="auto", origin="lower", cmap=div, interpolation="nearest",
                         norm=TwoSlopeNorm(vmin=1.0 / hi, vcenter=1.0, vmax=hi))
    for ax, title in [(axes[0], "(a) median implied vol. (annualized)"),
                      (axes[1], "(b) median IV / IV_ATM (shape)")]:
        grid_axis(ax); ax.set_xlabel("Time of day (New York)")
        ax.set_title(title, loc="left", fontsize=8)
        ax.axhline(z0, color="black", lw=0.3, alpha=0.6)
    ticks = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0]
    axes[0].set_yticks([zs.index(t) for t in ticks])
    axes[0].set_yticklabels([f"{t:+.0f}" if t != 0 else "0" for t in ticks])
    axes[0].set_ylabel("standardized moneyness z")
    c0 = fig.colorbar(im0, ax=axes[0], fraction=0.06, pad=0.03)
    c0.set_label("Black '76 implied vol.", fontsize=7); c0.ax.tick_params(labelsize=7)
    c1 = fig.colorbar(im1, ax=axes[1], fraction=0.06, pad=0.03)
    c1.set_label("IV / IV_ATM", fontsize=7); c1.ax.tick_params(labelsize=7)
    save(fig, out_dir, "17_fig_S1_iv_surface_by_z_and_time")
    return (f"celle mostrate {int(ok.sum())} su {len(g)} (valore in almeno meta' delle "
            f"{n_sessions} sessioni)")


# ---------------------------------------------------------------------
# SUPERFICI 3D, una figura per superficie. S2: tre sessioni di A1 x tre terzi della giornata,
# ognuno con il suo sorriso; S2D: le stesse sessioni sull'intera giornata; S3: superficie
# mediana sulle 247 sessioni regolari. S2 e S2D passano per gli strike veri, S3 usa i nodi di z.
# ---------------------------------------------------------------------

S_WINDOWS = [  # terzili della giornata usati nella tesi
    ("1_open", "09:40-11:30", "09:40:00", "11:30:00", "09:40:00"),
    ("2_midday", "11:40-13:40", "11:40:00", "13:40:00", "12:00:00"),
    ("3_close", "13:50-15:50", "13:50:00", "15:50:00", "15:00:00"),
]
S_ROLES = ["median_RV", "p90_RV", "stress_2025_04_07"]
SMILE_COLOR = {t: col for t, col, _ in SMILE_TIMES}


def _minutes(t: str) -> int:
    h, m, _ = map(int, t.split(":"))
    return h * 60 + m - (9 * 60 + 30)


def _window_grid(start: str, end: str):
    return [t for t in GRID if start <= t <= end]


# Cornice delle superfici 3D: stessa misura e stessa posizione in ogni file (niente ritaglio
# "tight"); ogni superficie ha la sua scala verticale.
SURF_FIGSIZE = (5.8, 4.3)              # pollici
SURF_AX_RECT = [-0.07, 0.0, 1.0, 1.0]   # riquadro 3D in coordinate della figura
SURF_ZOOM = 1.03
SURF_ZLABEL_XY = (0.905, 0.56)         # etichetta dell'asse verticale, coordinate della figura
SURF_LEGEND_Y = 0.985


def _surface_axes(ax, zs, times, z_lo, z_hi):
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ax.set_xlim(zs[0], zs[-1])
    ax.set_ylim(_minutes(times[0]), _minutes(times[-1]))
    pad = 0.02 * (z_hi - z_lo) if z_hi > z_lo else 0.01
    ax.set_zlim(z_lo - pad, z_hi + pad)
    ax.set_xticks([-8, -6, -4, -2, 0, 2, 4])
    ticks = [t for t in times if t[3:5] in ("00", "30")]
    if len(ticks) > 7:                       # giornata intera: un'etichetta all'ora
        ticks = [t for t in ticks if t[3:5] == "00"]
    ax.set_yticks([_minutes(t) for t in ticks])
    ax.set_yticklabels([t[:5] for t in ticks])
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=7, pad=0)
    ax.set_xlabel("standardized moneyness z", fontsize=8, labelpad=2)
    ax.set_ylabel("time of day (New York)", fontsize=8, labelpad=4)
    # etichetta dell'asse verticale in coordinate 2D: quella 3D (set_zlabel) finisce
    # fuori dal riquadro salvato con questa vista
    ax.figure.text(SURF_ZLABEL_XY[0], SURF_ZLABEL_XY[1], "implied vol. (annualized)",
                   rotation=90, fontsize=8, va="center", ha="left")
    ax.view_init(elev=24, azim=-58)
    try:                                     # zoom esiste da matplotlib 3.6
        ax.set_box_aspect((1.3, 1.0, 0.8), zoom=SURF_ZOOM)
    except TypeError:
        ax.set_box_aspect((1.3, 1.0, 0.8))
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0)); pane.set_edgecolor((0.8, 0.8, 0.8, 1))


def _draw_surface(ax, zs, times, Z, norm):
    """Quadrilateri SOLO dove i quattro vertici hanno un valore: nessuna cella mancante
    viene riempita o interpolata, e il risultato non dipende da come la versione di
    matplotlib tratta i NaN in plot_surface."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ys = np.array([_minutes(t) for t in times], dtype=float)
    polys, cols = [], []
    for i in range(len(zs) - 1):
        for j in range(len(ys) - 1):
            v = Z[i:i + 2, j:j + 2]
            if not np.all(np.isfinite(v)):
                continue
            polys.append([(zs[i], ys[j], v[0, 0]), (zs[i], ys[j + 1], v[0, 1]),
                          (zs[i + 1], ys[j + 1], v[1, 1]), (zs[i + 1], ys[j], v[1, 0])])
            cols.append(BR_SEQ(norm(float(v.mean()))))
    coll = Poly3DCollection(polys, facecolors=cols, edgecolors=(0.25, 0.25, 0.25, 0.3), linewidths=0.2)
    ax.add_collection3d(coll)
    return len(polys), (len(zs) - 1) * (len(ys) - 1)


def _smile_line(ax, zs, times, Z, t, label_fmt):
    j = times.index(t)
    ok = np.isfinite(Z[:, j])
    ax.plot(np.asarray(zs)[ok], np.full(int(ok.sum()), _minutes(t)), Z[ok, j],
            color=SMILE_COLOR[t], lw=1.8, zorder=10, label=label_fmt.format(t=t[:5]))


def _matrix(df, value, times):
    m = df.pivot(index="z", columns="grid_time", values=value)
    zs = sorted(m.index)
    return zs, m.reindex(index=zs, columns=times).astype(float).to_numpy()


def _one_surface(zs, times, Z, smile_times, title, name, out_dir,
                 label_fmt="smile at {t}",
                 footer="cells with no usable strike left empty.", exact_lines=None):
    """exact_lines: {grid_time: (z, iv)} con i punti della figura A1; se dato, le linee
    degli smile usano quei punti invece dei nodi della superficie."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra la proiezione 3d)
    from matplotlib.colors import Normalize
    if not np.isfinite(Z).any():
        return f"{name}: nessun valore, figura non scritta"
    lo, hi = float(np.nanmin(Z)), float(np.nanmax(Z))
    if exact_lines:
        for t in smile_times:
            if t in exact_lines and len(exact_lines[t][1]):
                lo = min(lo, float(np.min(exact_lines[t][1])))
                hi = max(hi, float(np.max(exact_lines[t][1])))
    fig = plt.figure(figsize=SURF_FIGSIZE)
    ax = fig.add_axes(SURF_AX_RECT, projection="3d")
    n, n_all = _draw_surface(ax, zs, times, Z, Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9))
    for t in smile_times:
        if t not in times:
            continue
        if exact_lines and t in exact_lines:
            zz, vv = exact_lines[t]
            ax.plot(zz, np.full(len(zz), _minutes(t)), vv, color=SMILE_COLOR[t], lw=1.8,
                    zorder=10, label=label_fmt.format(t=t[:5]))
        else:
            _smile_line(ax, zs, times, Z, t, label_fmt)
    _surface_axes(ax, zs, times, lo, hi)
    # Titolo e nota stanno nella didascalia LaTeX, non nell'immagine: `title` e `footer`
    # non sono usati.
    if smile_times:
        h, lab = ax.get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, fontsize=7.5, loc="upper center", ncol=len(h),
                   bbox_to_anchor=(0.5, SURF_LEGEND_Y), handlelength=1.6, columnspacing=1.4)
    save(fig, out_dir, name, tight=False)
    return f"{name}: {n}/{n_all} celle"


def _representative_days(csv_dir: Path):
    smiles_path = DATA_DIR / "10_surface_descriptives" / "10_smiles_representative_days_2025.csv"
    sm = pd.read_csv(smiles_path, keep_default_na=False)
    need(sm, ["quote_date", "day_role"], "smiles")
    roles = sm[["day_role", "quote_date"]].drop_duplicates()
    assert set(roles["day_role"]) == set(ROLE_TITLES), set(roles["day_role"])
    return dict(zip(roles["day_role"], roles["quote_date"]))


def _smile_points(csv_dir: Path, day: str):
    """I punti della figura A1 per un giorno: {grid_time: (z, iv)}, z dentro Z_RANGE,
    ordinati per z (gli stessi che figure_a1 disegna)."""
    path = DATA_DIR / "10_surface_descriptives" / "10_smiles_representative_days_2025.csv"
    sm = pd.read_csv(path, keep_default_na=False)
    need(sm, ["quote_date", "grid_time", "z", "iv"], "smiles")
    sm = sm[sm["quote_date"] == day].copy()
    sm["z"] = pd.to_numeric(sm["z"]); sm["iv"] = pd.to_numeric(sm["iv"])
    out = {}
    for t, g in sm.groupby("grid_time"):
        g = g[(g["z"] >= Z_RANGE[0]) & (g["z"] <= Z_RANGE[1])].sort_values("z")
        out[t] = (g["z"].to_numpy(), g["iv"].to_numpy())
    return out


CELL = 0.25          # maglia disegnata ogni 0.25 in z (come i nodi), ogni 10 minuti in t


def _points(csv_dir: Path):
    p = pd.read_csv(csv_dir / "10_iv_surface_points_repdays_2025.csv", keep_default_na=False,
                    dtype={"quote_date": str, "grid_time": str, "option_type": str})
    need(p, ["quote_date", "grid_time", "option_type", "k", "z", "iv", "iv_atm"], "iv_surface_points_repdays")
    for c in ["k", "z", "iv", "iv_atm"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    return p


def _slices(p_day, times):
    """{minuto: (z, iv)}: gli strike OTM utilizzabili con z dentro Z_RANGE, ordinati per z,
    piu' il punto (0, IV_ATM), che sta sul segmento fra il put e la call piu' vicini (e' la
    stessa interpolazione che definisce IV_ATM, quindi non cambia la spezzata). None se al
    minuto manca IV_ATM."""
    out = {}
    for t in times:
        g = p_day[p_day["grid_time"] == t]
        atm = g["iv_atm"].dropna()
        if g.empty or atm.empty:
            out[t] = None
            continue
        g = g[(g["z"] >= Z_RANGE[0]) & (g["z"] <= Z_RANGE[1])].sort_values("z")
        put, call = g[g["z"] < 0], g[g["z"] > 0]
        if put.empty or call.empty:
            out[t] = None
            continue
        z = np.concatenate([put["z"].to_numpy(float), [0.0], call["z"].to_numpy(float)])
        v = np.concatenate([put["iv"].to_numpy(float), [float(atm.iloc[0])], call["iv"].to_numpy(float)])
        assert np.all(np.diff(z) > 0), (t, "z non crescenti")
        out[t] = (z, v)
    return out


def _check_lines_on_slices(slices, lines, day):
    """Le linee (punti di A1) devono essere esattamente le sezioni della superficie ai
    minuti degli smile: stessi strike, stessi valori."""
    for t, (zz, vv) in lines.items():
        sl = slices.get(t)
        assert sl is not None, f"{day} {t}: nessuna sezione della superficie al minuto dello smile"
        z, v = sl
        keep = z != 0.0
        ok = (keep.sum() == len(zz) and np.allclose(z[keep], zz, rtol=1e-12, atol=0.0)
              and np.allclose(v[keep], vv, rtol=1e-12, atol=0.0))
        assert ok, (f"{day} {t}: la sezione della superficie non coincide con lo smile di A1 "
                    f"({int(keep.sum())} punti contro {len(zz)})")


def _half(z, v, sign):
    """Meta' sezione dal punto ATM verso l'esterno: distanze d = sign*z >= 0, crescenti."""
    m = sign * z >= 0
    d, w = sign * z[m], v[m]
    o = np.argsort(d, kind="mergesort")
    return d[o], w[o]


def _strip_half(dA, vA, yA, dB, vB, yB, sign):
    """Poligoni fra due minuti consecutivi (yA, yB) su un lato. Dentro l'estensione comune:
    una cella ogni 0.25 in z, con i lati lungo le due sezioni che passano per tutti gli
    strike della cella (stesso z sui due minuti). Oltre l'estensione comune: la sezione piu'
    estesa, cella per cella, chiusa sull'ultimo punto dell'altra."""
    def path(d, w, lo, hi):
        dd = np.concatenate([[lo], d[(d > lo) & (d < hi)], [hi]])
        return dd, np.interp(dd, d, w)
    polys, vals = [], []
    DA, DB = float(dA[-1]), float(dB[-1])
    c = min(DA, DB)
    edges = np.arange(0.0, max(DA, DB) + CELL, CELL)
    for e0, e1 in zip(edges[:-1], edges[1:]):
        lo, hi = float(e0), min(float(e1), c)
        if hi <= lo:
            break
        da, va = path(dA, vA, lo, hi)
        db, vb = path(dB, vB, lo, hi)
        polys.append([(sign * x, yA, y) for x, y in zip(da, va)] +
                     [(sign * x, yB, y) for x, y in zip(db[::-1], vb[::-1])])
        vals.append(float(np.mean(np.concatenate([va, vb]))))
    if DA != DB:
        (dW, vW, yW), (dN, vN, yN) = (((dA, vA, yA), (dB, vB, yB)) if DA > DB
                                      else ((dB, vB, yB), (dA, vA, yA)))
        apex = (sign * c, yN, float(np.interp(c, dN, vN)))
        DW = float(dW[-1])
        for e0, e1 in zip(edges[:-1], edges[1:]):
            lo, hi = max(float(e0), c), min(float(e1), DW)
            if hi <= lo:
                continue
            dw, vw = path(dW, vW, lo, hi)
            polys.append([(sign * x, yW, y) for x, y in zip(dw, vw)] + [apex])
            vals.append(float(np.mean(np.append(vw, apex[2]))))
    return polys, vals


def _draw_surface_from_slices(ax, slices, times, norm):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys, vals, n_strips = [], [], 0
    for t0, t1 in zip(times[:-1], times[1:]):
        A, B = slices.get(t0), slices.get(t1)
        if A is None or B is None:
            continue
        n_strips += 1
        for sign in (-1.0, 1.0):
            p, w = _strip_half(*_half(*A, sign), _minutes(t0), *_half(*B, sign), _minutes(t1), sign)
            polys += p
            vals += w
    coll = Poly3DCollection(polys, facecolors=[BR_SEQ(norm(x)) for x in vals],
                            edgecolors=(0.25, 0.25, 0.25, 0.3), linewidths=0.2)
    ax.add_collection3d(coll)
    return n_strips, len(times) - 1, len(polys)


def _one_surface_points(slices, times, smile_times, name, out_dir, lines):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra la proiezione 3d)
    from matplotlib.colors import Normalize
    vals = [sl[1] for sl in slices.values() if sl is not None]
    if not vals:
        return f"{name}: nessun valore, figura non scritta"
    allv = np.concatenate(vals)
    lo, hi = float(allv.min()), float(allv.max())
    for t in smile_times:
        if t in lines and len(lines[t][1]):
            lo, hi = min(lo, float(np.min(lines[t][1]))), max(hi, float(np.max(lines[t][1])))
    fig = plt.figure(figsize=SURF_FIGSIZE)
    ax = fig.add_axes(SURF_AX_RECT, projection="3d")
    n_strips, n_all, n_poly = _draw_surface_from_slices(
        ax, slices, times, Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9))
    for t in smile_times:
        if t in times and t in lines:
            zz, vv = lines[t]
            ax.plot(zz, np.full(len(zz), _minutes(t)), vv, color=SMILE_COLOR[t], lw=1.8,
                    zorder=10, label=f"smile at {t[:5]}")
    _surface_axes(ax, list(Z_RANGE), times, lo, hi)
    if smile_times:
        h, lab = ax.get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, fontsize=7.5, loc="upper center", ncol=len(h),
                   bbox_to_anchor=(0.5, SURF_LEGEND_Y), handlelength=1.6, columnspacing=1.4)
    save(fig, out_dir, name, tight=False)
    return f"{name}: {n_strips}/{n_all} intervalli di 10 minuti, {n_poly} poligoni"


def figure_s2(csv_dir: Path, out_dir: Path):
    """Nove superfici 3D: le tre sessioni di A1 per i tre terzi della giornata, una
    figura ciascuna, ognuna con il sorriso di A1 che cade nel suo terzo; dagli strike veri."""
    day_of = _representative_days(csv_dir)
    p = _points(csv_dir)
    notes = []
    for role in S_ROLES:
        day = day_of[role]
        pday = p[p["quote_date"] == day]
        lines = _smile_points(csv_dir, day)
        _check_lines_on_slices(_slices(pday, list(lines)), lines, day)
        for wkey, wlab, start, end, smile_t in S_WINDOWS:
            times = _window_grid(start, end)
            notes.append(_one_surface_points(_slices(pday, times), times, [smile_t],
                                             f"17_fig_S2_iv_surface_3d_{day}_{wkey}", out_dir, lines))
    return "; ".join(notes)


def figure_s2d(csv_dir: Path, out_dir: Path):
    """Tre superfici 3D, una per sessione di A1, sull'intera giornata (38 punti), con
    i tre sorrisi di A1; dagli strike veri."""
    day_of = _representative_days(csv_dir)
    p = _points(csv_dir)
    notes = []
    for role in S_ROLES:
        day = day_of[role]
        pday = p[p["quote_date"] == day]
        slices = _slices(pday, GRID)
        lines = _smile_points(csv_dir, day)
        _check_lines_on_slices(slices, lines, day)
        notes.append(_one_surface_points(slices, GRID, [t for t, _, _ in SMILE_TIMES],
                                         f"17_fig_S2D_iv_surface_3d_{day}_full_day", out_dir, lines))
    return "linee = sezioni della superficie ai minuti degli smile (verificato); " + "; ".join(notes)


def figure_s3(csv_dir: Path, out_dir: Path):
    """Superficie 3D mediana fra le 247 sessioni regolari (campione all), regola di
    copertura di S1: una cella entra solo se ha un valore in almeno meta' delle
    sessioni. Ai bordi la mediana e' calcolata solo sui giorni le cui quotazioni
    arrivano a quel nodo: leggerla come tale."""
    a = pd.read_csv(csv_dir / "10_iv_surface_by_z_agg_2025.csv", keep_default_na=False)
    need(a, ["sample", "grid_time", "z", "n_sessions", "coverage_ok", "iv_median"], "iv_surface_by_z_agg")
    g = a[a["sample"] == "all"].copy()
    g["z"] = pd.to_numeric(g["z"]).round(2)
    ok = g["coverage_ok"].astype(str).str.strip().str.lower() == "true"
    g["iv_shown"] = pd.to_numeric(g["iv_median"], errors="coerce").where(ok)
    zs, Z = _matrix(g, "iv_shown", GRID)
    assert zs[0] == Z_RANGE[0] and zs[-1] == Z_RANGE[1] and Z.shape == (len(zs), 38)
    n_sessions = int(pd.to_numeric(g["n_sessions"]).iloc[0])
    return _one_surface(zs, GRID, Z, [t for t, _, _ in SMILE_TIMES],
                        f"median across the {n_sessions} regular sessions",
                        "17_fig_S3_iv_surface_3d_median", out_dir,
                        label_fmt="median smile at {t}",
                        footer="cells with a value in fewer than half the sessions left empty.")

# ---------------------------------------------------------------------
# R3: volatilita' implicita di apertura e realizzata della sessione, 2025.
# ---------------------------------------------------------------------
R3_LOG_SCALE = False     # True: asse verticale logaritmico (il divario diventa una distanza costante)
FOMC_2025 = list(FOMC_DATES)
EARLY_CLOSES_2025 = set(EARLY_CLOSE_DATES)
MINUTES_PER_YEAR = 525600.0

# File di controllo di R3 e R4, scritti da 14_Daily_MZ_robustness.py.
MZ_EX_ANTE_CHECK = DATA_DIR / "14_daily_mz_robustness" / "14_mz_ex_ante_2025.csv"
JENSEN_CHECK = DATA_DIR / "14_daily_mz_robustness" / "14_log_gap_jensen_2025.csv"


def _check_mz_ex_ante(rv: np.ndarray, mfiv: np.ndarray) -> str:
    """La pendenza in livelli di RV_t su MFIV_0931 (campione intero) deve coincidere con
    quella del file delle regressioni ex ante; se il file manca il controllo e' saltato."""
    if not MZ_EX_ANTE_CHECK.exists():
        print(f"AVVISO: manca {MZ_EX_ANTE_CHECK}: controllo della pendenza ex ante (R3) SALTATO")
        return f"verifica sul file {MZ_EX_ANTE_CHECK.name} SALTATA: file non trovato"
    mz = pd.read_csv(MZ_EX_ANTE_CHECK, keep_default_na=False)
    need(mz, ["regressor", "sample", "form", "n", "slope"], "mz_ex_ante")
    row = mz[(mz["regressor"] == "MFIV_0931") & (mz["sample"] == "intero") & (mz["form"] == "levels")]
    assert len(row) == 1, "riga MFIV_0931 / intero / levels assente o doppia"
    row = row.iloc[0]
    assert len(rv) == int(row["n"]), f"{len(rv)} sessioni, file {row['n']}"
    b = float(np.polyfit(mfiv, rv, 1)[0])
    assert abs(b - float(row["slope"])) < 1e-6, f"pendenza {b} contro {row['slope']} del file"
    return f"pendenza ex ante in livelli {b:.4f}, uguale a {MZ_EX_ANTE_CHECK.name}"


def figure_r3(csv_dir: Path, out_dir: Path):
    """Volatilita' implicita all'apertura, MFIV_d(09:31), e realizzata della sessione, RV_d,
    giorno per giorno nel 2025, in percento: 100 * sqrt(varianza / tau_open), tau_open in anni
    di calendario. MFIV_d(09:31) e' l'unico prezzo ex ante della varianza della sessione;
    MFIV_session_t, media delle quotazioni della sessione stessa, non lo e'."""
    import matplotlib.dates as mdates
    from matplotlib.ticker import ScalarFormatter
    d = pd.read_csv(csv_dir / "07_daily_regression_panel_2025.csv", keep_default_na=False)
    need(d, ["quote_date", "RV_t", "MFIV_0931", "tau_open", "is_early_close"], "daily panel")
    for c in ["RV_t", "MFIV_0931", "tau_open"]:
        d[c] = pd.to_numeric(d[c].replace("", np.nan))
    d = d.sort_values("quote_date").reset_index(drop=True)
    assert len(d) == 250 and d["quote_date"].is_unique, f"{len(d)} righe o date ripetute"
    assert d["quote_date"].str[:4].eq("2025").all(), "date fuori dal 2025"
    v = d[["RV_t", "MFIV_0931", "tau_open"]]
    assert v.notna().all().all(), "valori mancanti in RV_t, MFIV_0931 o tau_open"
    assert (v > 0).all().all(), "valori non positivi in RV_t, MFIV_0931 o tau_open"
    early = d["is_early_close"].astype(str).str.strip().str.lower().isin(["1", "1.0", "true"])
    assert set(d.loc[early, "quote_date"]) == EARLY_CLOSES_2025, sorted(d.loc[early, "quote_date"])
    minutes = d["tau_open"] * MINUTES_PER_YEAR
    assert np.allclose(minutes[~early], 389.0, rtol=1e-6) and np.allclose(minutes[early], 209.0, rtol=1e-6), \
        "tau_open non e' in anni di calendario o non vale 389 (209) minuti"
    note = _check_mz_ex_ante(d["RV_t"].to_numpy(float), d["MFIV_0931"].to_numpy(float))

    iv = 100.0 * np.sqrt(d["MFIV_0931"] / d["tau_open"])
    rv = 100.0 * np.sqrt(d["RV_t"] / d["tau_open"])
    x = pd.to_datetime(d["quote_date"])
    april = d["quote_date"].str[5:7] == "04"
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.axvspan(x[april].min() - pd.Timedelta(hours=12), x[april].max() + pd.Timedelta(hours=12),
               color="#9e9e9e", alpha=0.18, lw=0, label=f"April 2025 ({int(april.sum())} sessions)")
    for i, f in enumerate(FOMC_2025):
        ax.axvline(pd.Timestamp(f), color="#6f6f6f", lw=0.6, ls=":", zorder=1,
                   label="FOMC decision day" if i == 0 else None)
    ax.plot(x, iv, color=BLUE, lw=1.1, zorder=3, label=r"implied, opening price $\mathrm{MFIV}_d$(09:31)")
    ax.plot(x, rv, color=RED, lw=0.9, zorder=2, label=r"realized, session $RV_d$")
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    if R3_LOG_SCALE:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
    else:
        ax.set_ylim(0, None)
    ax.set_ylabel("annualized volatility (%)")
    ax.set_xlabel("2025")
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, out_dir, "17_fig_R3_implied_vs_realized_vol_2025")
    return (f"{len(d)} sessioni, aprile {int(april.sum())}; mediane: implicita {np.median(iv):.1f}, "
            f"realizzata {np.median(rv):.1f}; {note}")


# ---------------------------------------------------------------------
# R4: distribuzione del log-rapporto giornaliero fra varianza realizzata e implicita.
# ---------------------------------------------------------------------
R4_MEASURES = [("MFIV_0931", r"(a) opening price $\mathrm{MFIV}_d$(09:31)"),
               ("MFIV_session_t", r"(b) $\mathrm{MFIV}^{ses}_d$ (calibration)")]
R4_BIN = 0.1


def _check_jensen(d: pd.DataFrame, april: pd.Series) -> str:
    """Le statistiche del log-rapporto devono coincidere con quelle del file della Jensen,
    per entrambe le misure e per i due campioni; se il file manca il controllo e' saltato."""
    if not JENSEN_CHECK.exists():
        print(f"AVVISO: manca {JENSEN_CHECK}: controllo delle statistiche del log-rapporto (R4) SALTATO")
        return f"verifica sul file {JENSEN_CHECK.name} SALTATA: file non trovato"
    j = pd.read_csv(JENSEN_CHECK, keep_default_na=False)
    need(j, ["regressor", "sample", "n", "mean_log_ratio", "var_log_ratio", "geometric_ratio",
             "mean_of_ratios", "median_ratio"], "mz_ex_ante_jensen")
    ddofs = set()
    for reg, _ in R4_MEASURES:
        for sample, sel in [("intero", np.ones(len(d), bool)), ("senza_aprile", ~april.to_numpy())]:
            row = j[(j["regressor"] == reg) & (j["sample"] == sample)]
            assert len(row) == 1, f"riga {reg} / {sample} assente o doppia"
            row = row.iloc[0]
            ratio = d.loc[sel, "RV_t"].to_numpy(float) / d.loc[sel, reg].to_numpy(float)
            lr = np.log(ratio)
            assert len(lr) == int(row["n"]), f"{reg} {sample}: {len(lr)} sessioni, file {row['n']}"
            for name, val in [("mean_log_ratio", lr.mean()), ("geometric_ratio", np.exp(lr.mean())),
                              ("mean_of_ratios", ratio.mean()), ("median_ratio", np.median(ratio))]:
                assert abs(val - float(row[name])) < 1e-9, f"{reg} {sample} {name}: {val} contro {row[name]}"
            ok = [k for k in (0, 1) if abs(lr.var(ddof=k) - float(row["var_log_ratio"])) < 1e-9]
            assert ok, f"{reg} {sample} var_log_ratio: {lr.var(ddof=0)} / {lr.var(ddof=1)} contro {row['var_log_ratio']}"
            ddofs.update(ok)
    return (f"statistiche del log-rapporto uguali a {JENSEN_CHECK.name} "
            f"(varianza con ddof {sorted(ddofs)})")


def figure_r4(csv_dir: Path, out_dir: Path):
    """Distribuzione di ln(RV_d / X_d) sulle 250 sessioni: (a) X_d = MFIV_0931, divario ex ante;
    (b) X_d = MFIV_session_t, diagnostica di calibrazione. E' il divario ex post, non il premio
    per il rischio di varianza."""
    d = pd.read_csv(csv_dir / "07_daily_regression_panel_2025.csv", keep_default_na=False)
    cols = ["RV_t"] + [m for m, _ in R4_MEASURES]
    need(d, ["quote_date"] + cols, "daily panel")
    for c in cols:
        d[c] = pd.to_numeric(d[c].replace("", np.nan))
    d = d.sort_values("quote_date").reset_index(drop=True)
    assert len(d) == 250 and d["quote_date"].is_unique, f"{len(d)} righe o date ripetute"
    assert d["quote_date"].str[:4].eq("2025").all(), "date fuori dal 2025"
    assert d[cols].notna().all().all(), "valori mancanti"
    assert (d[cols] > 0).all().all(), "valori non positivi"
    april = d["quote_date"].str[5:7] == "04"
    note = _check_jensen(d, april)

    lrs = {m: np.log(d["RV_t"].to_numpy(float) / d[m].to_numpy(float)) for m, _ in R4_MEASURES}
    allv = np.concatenate(list(lrs.values()))
    lo = np.floor(allv.min() / R4_BIN) * R4_BIN
    hi = np.ceil(allv.max() / R4_BIN) * R4_BIN
    bins = np.arange(lo, hi + R4_BIN / 2, R4_BIN)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    parts = []
    for ax, (m, title) in zip(axes, R4_MEASURES):
        lr = lrs[m]
        ax.hist([lr[~april.to_numpy()], lr[april.to_numpy()]], bins=bins, stacked=True,
                color=[BLUE, RED], edgecolor="white", linewidth=0.3,
                label=[f"other sessions ({int((~april).sum())})", f"April 2025 ({int(april.sum())})"])
        mu = float(lr.mean())
        ax.axvline(mu, color="black", lw=1.1, label="mean log ratio")
        ax.axvline(0.0, color="grey", lw=0.8, ls="--", label="no gap (ratio 1)")
        ax.set_title(f"{title}\ngeometric ratio {np.exp(mu):.3f}", loc="left", fontsize=8)
        ax.set_xlabel(r"$\ln(RV_d / X_d)$")
        ax.set_xlim(lo, hi)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        parts.append(f"{m}: media {mu:.4f}, scarto {lr.std(ddof=1):.3f}")
    axes[0].set_ylabel("sessions")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out_dir, "17_fig_R4_log_gap_distribution_2025")
    return "; ".join(parts) + f"; {note}"


# figura: (funzione, cartella del CSV principale, CSV principale).
FIGURES = {
    "A1": (figure_a1, "10_surface_descriptives", "10_smiles_representative_days_2025.csv"),
    "A2": (figure_a2, "10_surface_descriptives", "10_surface_agg_2025.csv"),
    "A3": (figure_a3, "10_surface_descriptives", "10_surface_agg_2025.csv"),
    "B1": (figure_b1, "10_surface_descriptives", "10_gamma_time_profile_2025.csv"),
    "B2": (figure_b2, "10_surface_descriptives", "10_gamma_moneyness_agg_2025.csv"),
    "B3": (figure_b3, "10_surface_descriptives", "10_gamma_moneyness_agg_2025.csv"),
    "C12": (figure_c12, "10_surface_descriptives", "10_liquidity_agg_2025.csv"),
    "B4": (figure_b4, "10_surface_descriptives", "10_vendor_vs_own_agg_2025.csv"),
    "S1": (figure_s1, "10_surface_descriptives", "10_iv_surface_by_z_agg_2025.csv"),
    "S2": (figure_s2, "10_surface_descriptives", "10_iv_surface_points_repdays_2025.csv"),
    "S2D": (figure_s2d, "10_surface_descriptives", "10_iv_surface_points_repdays_2025.csv"),
    "S3": (figure_s3, "10_surface_descriptives", "10_iv_surface_by_z_agg_2025.csv"),
    "R1": (figure_mz, "07_daily_panel", "07_daily_regression_panel_2025.csv"),
    "R2": (figure_effects, "16_intraday_robustness", "16_gamma_effect_size_ci.csv"),
    "R3": (figure_r3, "07_daily_panel", "07_daily_regression_panel_2025.csv"),
    "R4": (figure_r4, "07_daily_panel", "07_daily_regression_panel_2025.csv"),
}


def descriptive_figures():
    """Tutte le figure di FIGURES, in ordine; quelle il cui CSV manca sono saltate."""
    for key in FIGURES:
        func, folder, fname = FIGURES[key]
        csv_dir = DATA_DIR / folder
        if not (csv_dir / fname).exists():
            print(f"figura {key} saltata: manca {csv_dir / fname}")
            continue
        note = func(csv_dir, OUTPUT_DIR)
        print(f"figura {key} scritta in {OUTPUT_DIR}" + (f" ({note})" if note else ""))


def main():
    descriptive_figures()


if __name__ == "__main__":
    main()
