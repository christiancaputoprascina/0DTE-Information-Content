"""
Firma di volatilita' di SPX e SPY e numeri del testo che ne derivano (Appendice B e Tab. 3.3).
  Parte 1: varianza realizzata a 1-30 minuti rispetto a quella a un minuto, SPX e SPY, nelle 247
           sessioni regolari: tabella, valori per giornata, grafico.
  Parte 2: autocorrelazione, sessioni agitate, divario SPX-SPY, Tab. B.1 e conteggi dei
           rendimenti della Tab. 3.3, confrontati con i valori della tesi.
Input:  CBOE/Data/05_spx_minute_bars/, 06_spy_minute_bars/, 07_daily_panel/
Output: CBOE/Data/12_rv_signature/
"""

import time

import numpy as np
import pandas as pd

from common import DATA_DIR, EARLY_CLOSE_DATES, output_dir, path, require

OUTPUT_FOLDER = "12_rv_signature"
OUTPUT_DIR = output_dir(OUTPUT_FOLDER)

K_LIST = [1, 2, 3, 5, 10, 15, 20, 30]


# =====================================================================
# PARTE 1: FIRMA DI VOLATILITA', SPX E SPY A CONFRONTO
# =====================================================================
# Il confronto con lo SPY distingue due spiegazioni di una firma dell'SPX crescente: ritardo
# dell'indice calcolato (SPY piatto mentre l'SPX sale) o proprieta' del mercato (SPY che sale
# come l'SPX); se lo SPY sale ma meno, entrambe.
#
# Lo SPY non ha il ritardo dell'indice, ma il rimbalzo fra denaro e lettera spinge la sua firma
# verso il basso. "Piatto": media dei rapporti giornalieri entro il 5 per cento di 1 fino a 15 minuti.
#
# Il file SPY e' in ora italiana: la conversione usa i fusi orari, perche' l'ora legale cambia in
# date diverse, e si verifica che ogni sessione regolare vada dalle 09:31 alle 16:00 di New York.
# Le barre sono etichettate con l'ora di chiusura, come nel file SPX.

SPX_DIR = DATA_DIR / "05_spx_minute_bars"
SPX_FILE_PATTERN = "05_SPX_{:02d}_RV.csv"

# File SPY pulito scritto da 06_SPY_minute_bars.py.
SPY_FILE = DATA_DIR / "06_spy_minute_bars" / "06_SPY_2025_1min_clean_RV.csv"
SPY_SOURCE_TZ = "Europe/Rome"
MARKET_TZ = "America/New_York"

OUTPUT_TABLE_NAME = "12_rv_signature_spx_spy_2025.csv"
OUTPUT_DAILY_NAME = "12_rv_signature_spx_spy_daily_2025.csv"
OUTPUT_PLOT_NAME = "12_rv_signature_spx_spy_2025.png"

SESSION_START = 9 * 60 + 31
SESSION_END = 16 * 60

RULE_MAX_K = 15
RULE_TOL = 0.05

ANCHOR_DATE = "2025-06-16"
ANCHOR_RV_DISPLAY = "1.977605e-05"


def find_spy_file():
    if SPY_FILE.exists():
        return SPY_FILE
    raise FileNotFoundError(
        f"File SPY non trovato:\n{SPY_FILE}\n"
        "Lancia prima 06_SPY_minute_bars.py.")


def load_spx() -> pd.DataFrame:
    frames = []
    for m in range(1, 13):
        f = SPX_DIR / SPX_FILE_PATTERN.format(m)
        if not f.exists():
            raise FileNotFoundError(f"File SPX mancante:\n{f}")
        frames.append(pd.read_csv(f, usecols=["Underlying", "Date", "time", "close"]))
    d = pd.concat(frames, ignore_index=True)
    d = d[d["Underlying"].astype(str).str.strip() == "SPX"].copy()
    parts = d["time"].astype(str).str.strip().str.split(":", expand=True)
    return pd.DataFrame({
        "date": pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d"),
        "minute": parts[0].astype(int) * 60 + parts[1].astype(int),
        "close": pd.to_numeric(d["close"], errors="coerce"),
    })


def load_spy(spy_path):
    d = pd.read_csv(spy_path, usecols=["Date", "Time Interval", "Close"])
    iv = d["Time Interval"].astype(str).str.split("-", expand=True)
    end_local = pd.to_datetime(d["Date"].astype(str) + " " + iv[1].str.strip(),
                               format="%Y-%m-%d %H:%M")
    end_mkt = (end_local.dt.tz_localize(SPY_SOURCE_TZ)
               .dt.tz_convert(MARKET_TZ).dt.tz_localize(None))
    out = pd.DataFrame({
        "date": end_mkt.dt.strftime("%Y-%m-%d"),
        "minute": end_mkt.dt.hour * 60 + end_mkt.dt.minute,
        "close": pd.to_numeric(d["Close"], errors="coerce"),
    })
    return out


def signature_daily(df: pd.DataFrame, label: str):
    df = df[(df["minute"] >= SESSION_START) & (df["minute"] <= SESSION_END)
            & (df["close"] > 0)].copy()
    dup = int(df.duplicated(["date", "minute"]).sum())
    if dup:
        raise ValueError(f"{label}: {dup} coppie data-minuto duplicate.")

    rows, early, gaps, first_last = [], [], [], []
    for date, g in df.groupby("date", sort=True):
        g = g.sort_values("minute")
        first_last.append((date, int(g["minute"].min()), int(g["minute"].max())))
        if g["minute"].max() < SESSION_END:
            early.append(date)
            continue
        s = g.set_index("minute")["close"].reindex(range(SESSION_START, SESSION_END + 1)).ffill()
        if s.isna().any():
            gaps.append(date)
            continue
        lp = np.log(s.to_numpy())
        n = len(lp)
        r1 = np.diff(lp)
        rec = {"date": date, "series": label,
               "n_missing_minutes": int(n - g["minute"].nunique()),
               "acf1_1min": float(np.corrcoef(r1[:-1], r1[1:])[0, 1])}
        for k in K_LIST:
            vals = []
            for o in range(k):
                pts = np.unique(np.concatenate(([0], np.arange(o, n, k), [n - 1])))
                vals.append(float(np.sum(np.diff(lp[pts]) ** 2)))
            rec[f"rv_{k}"] = float(np.mean(vals))
        rows.append(rec)
    return pd.DataFrame(rows), early, gaps, pd.DataFrame(first_last, columns=["date", "first", "last"])


def signature_table(daily: pd.DataFrame, label: str) -> pd.DataFrame:
    base = daily["rv_1"]
    no_apr = ~daily["date"].str.startswith("2025-04")
    out = []
    for k in K_LIST:
        r = daily[f"rv_{k}"] / base
        out.append({
            "series": label, "k_minutes": k,
            "mean_daily_ratio": float(r.mean()),
            "se_mean_daily_ratio": float(r.std(ddof=1) / np.sqrt(len(r))),
            "median_daily_ratio": float(r.median()),
            "ratio_of_means": float(daily[f"rv_{k}"].mean() / base.mean()),
            "ratio_of_means_no_april": float(daily.loc[no_apr, f"rv_{k}"].mean()
                                             / base[no_apr].mean()),
        })
    return pd.DataFrame(out)


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def is_flat(t: pd.DataFrame) -> bool:
    w = t[t["k_minutes"] <= RULE_MAX_K]
    return bool(((w["mean_daily_ratio"] - 1).abs() < RULE_TOL).all())


def rv_signature():
    print("=" * 130)
    print("FIRMA DI VOLATILITA': SPX E SPY, 2025")
    print("=" * 130)

    spy_path = find_spy_file()
    print(f"SPX: {SPX_DIR}")
    print(f"SPY: {spy_path}  (orario di origine {SPY_SOURCE_TZ}, convertito in {MARKET_TZ})")

    spx_d, spx_early, spx_gaps, _ = signature_daily(load_spx(), "SPX")
    spy_d, spy_early, spy_gaps, spy_fl = signature_daily(load_spy(spy_path), "SPY")

    print(f"SPX: {len(spx_d)} sessioni; chiusure anticipate {spx_early}; con buchi {len(spx_gaps)}")
    print(f"SPY: {len(spy_d)} sessioni; chiusure anticipate {spy_early}; con buchi {len(spy_gaps)}")

    daily = pd.concat([spx_d, spy_d], ignore_index=True)
    daily.to_csv(OUTPUT_DIR / OUTPUT_DAILY_NAME, index=False)

    # ---------------------------------------------------------------- controlli
    print()
    print("CONTROLLI")
    checks = []

    a = spx_d[spx_d["date"] == ANCHOR_DATE]
    got = f"{a['rv_1'].iloc[0]:.6e}" if len(a) else "assente"
    checks.append(("SPX: RV a un minuto del 16 giugno riproduce il pannello",
                   got == ANCHOR_RV_DISPLAY, f"atteso {ANCHOR_RV_DISPLAY}, ottenuto {got}"))

    regular = spy_fl[~spy_fl["date"].isin(spy_early)]
    bad_open = regular[regular["first"] != SESSION_START]
    bad_close = regular[regular["last"] != SESSION_END]
    checks.append(("SPY: ogni sessione regolare apre alle 09:31 e chiude alle 16:00 di New York "
                   "(verifica del fuso orario)",
                   len(bad_open) == 0 and len(bad_close) == 0,
                   f"{len(regular)} sessioni, prima barra fuori posto in {len(bad_open)}, "
                   f"ultima in {len(bad_close)}"
                   + (f"; esempi {bad_open['date'].head(3).tolist() + bad_close['date'].head(3).tolist()}"
                      if len(bad_open) or len(bad_close) else "")))

    same = set(spx_d["date"]) == set(spy_d["date"])
    checks.append(("SPX e SPY sulle stesse 247 sessioni regolari",
                   same and len(spx_d) == 247,
                   f"SPX {len(spx_d)}, SPY {len(spy_d)}, insiemi uguali: {same}"))

    for name, ok, det in checks:
        print(f"  [{'OK' if ok else 'FALLITO'}] {name}: {det}")

    if not all(ok for _, ok, _ in checks):
        print()
        print("CONTROLLI FALLITI: i confronti che seguono non sono affidabili.")

    # ---------------------------------------------------------------- tabelle
    tab = pd.concat([signature_table(spx_d, "SPX"), signature_table(spy_d, "SPY")], ignore_index=True)
    tab.to_csv(OUTPUT_DIR / OUTPUT_TABLE_NAME, index=False)

    print()
    print("FIRMA (rapporto rispetto al campionamento a un minuto)")
    print("  mean_daily_ratio: giornata tipica; ratio_of_means: pesato dai giorni volatili")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print()
    print("AUTOCORRELAZIONE DI PRIMO ORDINE DEI RENDIMENTI A UN MINUTO")
    for lab, d in (("SPX", spx_d), ("SPY", spy_d)):
        x = d["acf1_1min"]
        print(f"  {lab}: media {x.mean():+.4f}, mediana {x.median():+.4f}, "
              f"errore standard {x.std(ddof=1)/np.sqrt(len(x)):.4f}")

    print()
    print("LA SOTTOSTIMA E' MAGGIORE NEI GIORNI AGITATI? (verifica della deduzione)")
    for lab, d in (("SPX", spx_d), ("SPY", spy_d)):
        rho = spearman(d["rv_5"] / d["rv_1"], d["rv_1"])
        q = pd.qcut(d["rv_1"], 5, labels=False)
        by_q = (d["rv_5"] / d["rv_1"]).groupby(q).mean()
        print(f"  {lab}: correlazione di rango fra RV(5)/RV(1) e RV(1): {rho:+.3f}")
        print(f"        RV(5)/RV(1) medio per quintile di RV(1), dal piu' calmo al piu' agitato: "
              + ", ".join(f"{v:.3f}" for v in by_q.values))

    m = spx_d[["date", "rv_1", "rv_5"]].merge(spy_d[["date", "rv_1", "rv_5"]], on="date",
                                              suffixes=("_spx", "_spy"))
    print()
    print("SPX CONTRO SPY, STESSA GIORNATA")
    print(f"  RV(1) SPX / RV(1) SPY, media dei rapporti giornalieri: "
          f"{(m['rv_1_spx']/m['rv_1_spy']).mean():.4f}")
    print(f"  RV(5) SPX / RV(5) SPY, media dei rapporti giornalieri: "
          f"{(m['rv_5_spx']/m['rv_5_spy']).mean():.4f}")
    print("  A 5 minuti il ritardo dell'indice conta molto meno: se il rapporto e' vicino a 1")
    print("  a 5 minuti ma sotto 1 a un minuto, l'SPX perde varianza al minuto.")

    # ---------------------------------------------------------------- lettura secondo la regola
    spx_t, spy_t = tab[tab["series"] == "SPX"], tab[tab["series"] == "SPY"]
    spx_rise = float(spx_t.loc[spx_t["k_minutes"] == 5, "mean_daily_ratio"].iloc[0]) - 1
    spy_rise = float(spy_t.loc[spy_t["k_minutes"] == 5, "mean_daily_ratio"].iloc[0]) - 1

    print()
    print("LETTURA SECONDO LA REGOLA FISSATA PRIMA (giornata tipica, fino a 15 minuti, 5%)")
    print(f"  SPX piatto: {is_flat(spx_t)}; a 5 minuti {spx_rise:+.1%}")
    print(f"  SPY piatto: {is_flat(spy_t)}; a 5 minuti {spy_rise:+.1%}")
    if spy_rise <= 0.005 and spx_rise > 0.005:
        print("  -> (a) Il ritardo e' dell'INDICE: lo SPY non sale.")
    elif spy_rise >= 0.8 * spx_rise:
        print("  -> (b) Sale anche lo SPY, quasi quanto l'SPX: e' una proprieta' del MERCATO.")
    else:
        print(f"  -> Entrambe: lo SPY sale ma meno. Quota attribuibile all'indice, a 5 minuti: "
              f"circa {(spx_rise - spy_rise) / spx_rise:.0%}.")

    # ---------------------------------------------------------------- grafico
    # Backend Agg: nessuna finestra.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for lab, t, mk in (("SPX", spx_t, "o"), ("SPY", spy_t, "s")):
            ax.errorbar(t["k_minutes"], t["mean_daily_ratio"], yerr=2 * t["se_mean_daily_ratio"],
                        marker=mk, capsize=3, label=f"{lab}, giornata tipica (+-2 e.s.)")
            ax.plot(t["k_minutes"], t["ratio_of_means"], ls="--", marker=mk, alpha=0.6,
                    label=f"{lab}, rapporto delle medie")
        ax.axhline(1.0, color="black", lw=0.8)
        ax.axhspan(1 - RULE_TOL, 1 + RULE_TOL, color="grey", alpha=0.15, label=f"+-{RULE_TOL:.0%}")
        ax.set_xscale("log")
        ax.set_xticks(K_LIST)
        ax.set_xticklabels([str(k) for k in K_LIST])
        ax.set_xlabel("intervallo di campionamento (minuti)")
        ax.set_ylabel("varianza realizzata / varianza a 1 minuto")
        ax.set_title("Firma di volatilita' 2025, 247 sessioni regolari: SPX e SPY")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / OUTPUT_PLOT_NAME, dpi=150)
        print()
        print(f"Grafico: {OUTPUT_DIR / OUTPUT_PLOT_NAME}")
    except ImportError:
        print("matplotlib non disponibile: grafico non prodotto.")

    print(f"Tabella: {OUTPUT_TABLE_NAME}; valori giornalieri: {OUTPUT_DAILY_NAME}")
    print("=" * 130)

    if not all(ok for _, ok, _ in checks):
        raise SystemExit("Parte 1: controlli FALLITI (vedi sopra).")


# =====================================================================
# PARTE 2: FIRMA DI VOLATILITA' E FINESTRE DELLA RV, I NUMERI DEL TESTO
# =====================================================================
# Ricalcola dal file per giornata della parte 1 i numeri del testo (autocorrelazione dei
# rendimenti a un minuto, sessioni agitate, divario SPX-SPY) e la Tabella B.1, conta i rendimenti
# della Tabella 3.3 dal pannello di 07_Daily_panel.py e confronta tutto con i valori della tesi.
#
# Sessioni agitate: Spearman fra RV(5)/RV(1) e RV(1) positivo e quintile piu' agitato sopra il
# piu' calmo. Divario SPX-SPY: a 5 minuti |rapporto - 1| minore che a 1 minuto.
#
# Esito OK se il valore arrotondato alla precisione stampata coincide; un DIVERSO non ferma lo
# script, lo fermano solo i problemi di integrita' dei dati.

SIGNATURE_DAILY = (OUTPUT_FOLDER, OUTPUT_DAILY_NAME)
SIGNATURE_TABLE = (OUTPUT_FOLDER, OUTPUT_TABLE_NAME)
DAILY_PANEL = ("07_daily_panel", "07_daily_regression_panel_2025.csv")

OUT_NUMBERS = "12_rv_signature_text_numbers_2025.csv"
OUT_QUINTILES = "12_rv5_rv1_by_rv1_quintile_2025.csv"
OUT_THESIS = "12_thesis_check.csv"

TABLE_B1_K = [2, 3, 5, 10, 15, 20, 30]
N_QUINTILES = 5
REGULAR_SESSIONS_EXPECTED = 247
ALL_SESSIONS_EXPECTED = 250
EARLY_CLOSES = set(EARLY_CLOSE_DATES)
TABLE_TOL = 1e-12

# Tabella B.1 come stampata: k -> (media, s.e., mediana)
TABLE_B1_THESIS = {
    "SPX": {2: ("1.021", "0.005", "1.024"), 3: ("1.026", "0.007", "1.026"),
            5: ("1.032", "0.009", "1.024"), 10: ("1.026", "0.014", "1.002"),
            15: ("1.026", "0.017", "0.983"), 20: ("1.023", "0.019", "0.969"),
            30: ("1.020", "0.023", "0.955")},
    "SPY": {2: ("0.983", "0.005", "0.978"), 3: ("0.970", "0.006", "0.963"),
            5: ("0.961", "0.009", "0.952"), 10: ("0.943", "0.012", "0.917"),
            15: ("0.939", "0.015", "0.894"), 20: ("0.935", "0.017", "0.890"),
            30: ("0.931", "0.020", "0.874")},
}


class IntegrityError(RuntimeError):
    pass


def integrity(cond: bool, message: str):
    if not cond:
        raise IntegrityError(message)


def _norm(s: str) -> str:
    return str(s).replace(",", "").replace("−", "-").replace("+", "").strip()


def single(values, label):
    u = sorted(values.dropna().unique())
    integrity(len(u) == 1, f"{label}: valori non unici {u}")
    return u[0]


def rv_signature_text_numbers():
    thesis_rows = []
    numbers = []

    # ---------------------------------------------------------------- confronto con la tesi
    def check(pagina, voce, valore_tesi, valore_calcolato, precisione, nota="", esito=None):
        """precisione: 'dN' (N decimali), 'intero' o 'testo' (con esito esplicito)."""
        if esito is None:
            if precisione == "intero":
                shown = f"{int(round(float(valore_calcolato))):,}"
                ok = int(round(float(valore_calcolato))) == int(float(_norm(valore_tesi)))
            elif precisione.startswith("d"):
                nd = int(precisione[1:])
                shown = f"{float(valore_calcolato):+.{nd}f}" if str(valore_tesi).strip()[:1] in "+-−" \
                    else f"{float(valore_calcolato):.{nd}f}"
                ok = float(_norm(shown)) == float(_norm(valore_tesi))
            else:
                raise ValueError("Per precisione 'testo' passare esito esplicito.")
        else:
            ok = bool(esito)
            shown = str(valore_calcolato)
        thesis_rows.append({
            "pagina": pagina, "voce": voce, "valore_tesi": valore_tesi,
            "valore_calcolato": shown, "precisione": precisione,
            "esito": "OK" if ok else "DIVERSO", "nota": nota,
        })

    def record(sezione, serie, voce, valore, nota=""):
        numbers.append({"sezione": sezione, "serie": serie, "voce": voce, "valore": valore, "nota": nota})

    # ---------------------------------------------------------------- lettura e integrita'
    t0 = time.time()
    print("=" * 120)
    print("FIRMA DI VOLATILITA' E FINESTRE DELLA RV: NUMERI DEL TESTO")
    print("=" * 120)

    for p in (path(SIGNATURE_DAILY), path(SIGNATURE_TABLE), path(DAILY_PANEL)):
        require(p)

    daily = pd.read_csv(path(SIGNATURE_DAILY))
    table = pd.read_csv(path(SIGNATURE_TABLE))
    panel = pd.read_csv(path(DAILY_PANEL))
    daily["date"] = daily["date"].astype(str).str.strip()
    panel["quote_date"] = panel["quote_date"].astype(str).str.strip()

    integrity(set(daily["series"]) == {"SPX", "SPY"}, f"serie inattese nel file della parte 1: {set(daily['series'])}")
    integrity(not daily.duplicated(["date", "series"]).any(), "parte 1: coppie data-serie duplicate")
    series = {s: g.sort_values("date").reset_index(drop=True) for s, g in daily.groupby("series")}
    for s, g in series.items():
        integrity(len(g) == REGULAR_SESSIONS_EXPECTED, f"parte 1 {s}: {len(g)} sessioni, attese {REGULAR_SESSIONS_EXPECTED}")
        integrity(not (set(g["date"]) & EARLY_CLOSES), f"parte 1 {s}: contiene chiusure anticipate")
        integrity((g[[f"rv_{k}" for k in K_LIST]] > 0).all().all(), f"parte 1 {s}: RV non positive")
    integrity(set(series["SPX"]["date"]) == set(series["SPY"]["date"]), "parte 1: SPX e SPY su giornate diverse")

    # la tabella della parte 1 si ricalcola dal file per giornata
    for s, g in series.items():
        for k in K_LIST:
            r = g[f"rv_{k}"] / g["rv_1"]
            row = table[(table["series"] == s) & (table["k_minutes"] == k)]
            integrity(len(row) == 1, f"parte 1, tabella: riga {s} k={k} mancante o doppia")
            row = row.iloc[0]
            integrity(abs(r.mean() - row["mean_daily_ratio"]) <= TABLE_TOL
                      and abs(r.std(ddof=1) / np.sqrt(len(r)) - row["se_mean_daily_ratio"]) <= TABLE_TOL
                      and abs(r.median() - row["median_daily_ratio"]) <= TABLE_TOL,
                      f"parte 1: la tabella non si ricalcola dal file per giornata ({s}, k={k})")

    # ---------------------------------------------------------------- 1. autocorrelazione di primo ordine
    print()
    print("AUTOCORRELAZIONE DI PRIMO ORDINE DEI RENDIMENTI A UN MINUTO")
    acf = {}
    for s, g in series.items():
        x = g["acf1_1min"]
        integrity(x.notna().all(), f"parte 1 {s}: autocorrelazione mancante")
        m, se, med = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), float(x.median())
        acf[s] = (m, se)
        record("autocorrelazione", s, "media", m, f"n = {len(x)}")
        record("autocorrelazione", s, "errore standard (sd/sqrt(n))", se)
        record("autocorrelazione", s, "mediana", med)
        print(f"  {s}: media {m:+.4f}, s.e. {se:.4f}, mediana {med:+.4f}")
    check(69, "Table B.1 note: mean first-order autocorrelation, SPX (mean)", "+0.019", acf["SPX"][0], "d3")
    check(69, "Table B.1 note: mean first-order autocorrelation, SPX (s.e.)", "0.005", acf["SPX"][1], "d3")
    check(69, "Table B.1 note: mean first-order autocorrelation, SPY (mean)", "-0.019", acf["SPY"][0], "d3")
    check(69, "Table B.1 note: mean first-order autocorrelation, SPY (s.e.)", "0.005", acf["SPY"][1], "d3")

    # ---------------------------------------------------------------- 2. sessioni agitate
    print()
    print("RV(5)/RV(1) E LIVELLO DI RV(1)")
    quint_rows = []
    volatile = {}
    for s, g in series.items():
        ratio = g["rv_5"] / g["rv_1"]
        rho = float(ratio.rank().corr(g["rv_1"].rank()))
        q = pd.qcut(g["rv_1"], N_QUINTILES, labels=False)
        by_q = ratio.groupby(q).agg(["mean", "count"])
        for qi, row in by_q.iterrows():
            quint_rows.append({"serie": s, "quintile_rv1": int(qi) + 1, "n_sessioni": int(row["count"]),
                               "media_rv5_su_rv1": float(row["mean"]),
                               "rv1_min": float(g["rv_1"][q == qi].min()), "rv1_max": float(g["rv_1"][q == qi].max())})
        volatile[s] = (rho, by_q["mean"].to_list())
        record("sessioni_agitate", s, "Spearman RV5/RV1 contro RV1", rho)
        record("sessioni_agitate", s, "media RV5/RV1 per quintile di RV1 (dal piu' calmo)",
               ", ".join(f"{v:.3f}" for v in by_q["mean"]))
        print(f"  {s}: Spearman {rho:+.3f}; per quintile " + ", ".join(f"{v:.3f}" for v in by_q["mean"]))
    pd.DataFrame(quint_rows).to_csv(OUTPUT_DIR / OUT_QUINTILES, index=False)

    rho_spx, q_spx = volatile["SPX"]
    check(27, "five-minute SPX RV ... with larger differences on volatile sessions",
          "differenza maggiore nelle sessioni agitate",
          f"Spearman {rho_spx:+.3f}; quintili " + ", ".join(f"{v:.3f}" for v in q_spx), "testo",
          "criterio: Spearman > 0 e quintile piu' agitato sopra il piu' calmo; il profilo non e' monotono "
          "(il 4o quintile supera il 5o)",
          esito=(rho_spx > 0 and q_spx[-1] > q_spx[0]))

    # ---------------------------------------------------------------- 3. divario SPX-SPY
    m = series["SPX"][["date"] + [f"rv_{k}" for k in K_LIST]].merge(
        series["SPY"][["date"] + [f"rv_{k}" for k in K_LIST]], on="date", suffixes=("_spx", "_spy"))
    integrity(len(m) == REGULAR_SESSIONS_EXPECTED, "SPX e SPY: unione delle giornate incompleta")
    gap = {}
    for k in (1, 5):
        rr = m[f"rv_{k}_spx"] / m[f"rv_{k}_spy"]
        gap[k] = float(rr.mean())
        record("divario_spx_spy", "SPX/SPY", f"media dei rapporti giornalieri RV_SPX/RV_SPY, k = {k}", gap[k])
        record("divario_spx_spy", "SPX/SPY", f"rapporto delle medie RV_SPX/RV_SPY, k = {k}",
               float(m[f"rv_{k}_spx"].mean() / m[f"rv_{k}_spy"].mean()))
    print()
    print(f"DIVARIO SPX-SPY: media dei rapporti giornalieri {gap[1]:.4f} a 1 minuto, {gap[5]:.4f} a 5 minuti")
    check(68, "the SPX-SPY gap is much smaller at five minutes", "divario minore a 5 minuti",
          f"RV_SPX/RV_SPY {gap[1]:.3f} a 1 min, {gap[5]:.3f} a 5 min", "testo",
          "media dei rapporti giornalieri; criterio |rapporto - 1| minore a 5 minuti",
          esito=abs(gap[5] - 1) < abs(gap[1] - 1))

    # ---------------------------------------------------------------- 4. RV(5)/RV(1), minuti mancanti, sessioni
    r5 = series["SPX"]["rv_5"] / series["SPX"]["rv_1"]
    record("rapporto_5_1", "SPX", "media dei rapporti giornalieri RV5/RV1", float(r5.mean()))
    check(27, "five-minute SPX RV about 3% higher than one-minute (mean of daily ratios), in per cent", "3",
          100 * (r5.mean() - 1), "intero", f"rapporto medio {r5.mean():.4f}")
    check(68, "SPX mean five-minute to one-minute variance ratio", "1.032", r5.mean(), "d3")
    for s, g in series.items():
        miss = int(g["n_missing_minutes"].sum())
        record("minuti_mancanti", s, "somma di n_missing_minutes", miss, f"massimo {int(g['n_missing_minutes'].max())}")
        check(69, f"No missing minutes ({s})", "0", miss, "intero")
        check(68, f"SPX and SPY compared on the 247 regular sessions ({s})", "247", len(g), "intero")

    # ---------------------------------------------------------------- 5. tabella B.1
    for s, by_k in TABLE_B1_THESIS.items():
        for k in TABLE_B1_K:
            row = table[(table["series"] == s) & (table["k_minutes"] == k)].iloc[0]
            tm, tse, tmed = by_k[k]
            check(69, f"Table B.1 {s} k={k}: mean", tm, row["mean_daily_ratio"], "d3")
            check(69, f"Table B.1 {s} k={k}: s.e.", tse, row["se_mean_daily_ratio"], "d3")
            check(69, f"Table B.1 {s} k={k}: median", tmed, row["median_daily_ratio"], "d3")

    # ---------------------------------------------------------------- 6. tabella 3.3: numero di rendimenti
    integrity(len(panel) == ALL_SESSIONS_EXPECTED and panel["quote_date"].is_unique, "pannello 07: una riga per giornata attesa")
    early = pd.to_numeric(panel["is_early_close"], errors="coerce") == 1
    integrity(set(panel.loc[early, "quote_date"]) == EARLY_CLOSES, "pannello 07: chiusure anticipate inattese")
    cnt = {c: pd.to_numeric(panel[c], errors="coerce") for c in
           ("n_returns_full_session", "n_returns_open_window", "n_returns_fwd_short")}
    integrity(((cnt["n_returns_open_window"] + cnt["n_returns_fwd_short"]) == cnt["n_returns_full_session"]).all(),
              "pannello 07: finestra di apertura + finestra successiva diverse dalla sessione intera")

    rv_reg = single(cnt["n_returns_full_session"][~early], "RV_d, sessioni regolari")
    rv_early = single(cnt["n_returns_full_session"][early], "RV_d, chiusure anticipate")
    open_all = single(cnt["n_returns_open_window"], "SpotRV_open")
    fwd_reg = single(cnt["n_returns_fwd_short"][~early], "FwdRV_short, sessioni regolari")
    ordered = panel.sort_values("quote_date").reset_index(drop=True)
    prev_n = pd.to_numeric(ordered["n_returns_full_session"], errors="coerce").shift(1)
    prev_early = (pd.to_numeric(ordered["is_early_close"], errors="coerce") == 1).shift(1, fill_value=False)
    lag_reg = single(prev_n[(~prev_early) & prev_n.notna()], "RV_{d-1}, sessione precedente regolare")
    for voce, v in (("RV_d, sessione regolare", rv_reg), ("RV_d, chiusura anticipata", rv_early),
                    ("RV_{d-1}, sessione precedente regolare", lag_reg), ("SpotRV_open", open_all),
                    ("FwdRV_short, sessione regolare", fwd_reg)):
        record("tabella_3_3", "SPX", f"rendimenti: {voce}", v)
    record("tabella_3_3", "SPX", "rendimenti: FwdRV_short, chiusura anticipata",
           single(cnt["n_returns_fwd_short"][early], "FwdRV_short, chiusure anticipate"), "non riportato in tesi")
    check(26, "Table 3.3: RV_d returns (09:31,16:00]", "389", rv_reg, "intero")
    check(26, "Table 3.3: RV_{d-1} returns", "389", lag_reg, "intero", "sessione precedente regolare")
    check(26, "Table 3.3: SpotRV_open returns (09:31,10:00]", "29", open_all, "intero")
    check(26, "Table 3.3: FwdRV_short returns (10:00,16:00]", "360", fwd_reg, "intero")
    check(26, "Table 3.3 note: 209 for RV_d on the three early closes", "209", rv_early, "intero")
    check(26, "Table 3.3 note: three early closes", "3", int(early.sum()), "intero")

    # ---------------------------------------------------------------- uscita e verdetto
    pd.DataFrame(numbers).to_csv(OUTPUT_DIR / OUT_NUMBERS, index=False)
    th = pd.DataFrame(thesis_rows)[["pagina", "voce", "valore_tesi", "valore_calcolato", "precisione", "esito", "nota"]]
    th.to_csv(OUTPUT_DIR / OUT_THESIS, index=False)

    n_ok = int((th["esito"] == "OK").sum())
    n_diff = int((th["esito"] == "DIVERSO").sum())
    print()
    print("=" * 120)
    print(f"CONFRONTO CON LA TESI: {n_ok} OK, {n_diff} DIVERSO su {len(th)} voci")
    for _, r in th[th["esito"] == "DIVERSO"].iterrows():
        print(f"  DIVERSO p. {r['pagina']}: {r['voce']} | tesi {r['valore_tesi']} | calcolato {r['valore_calcolato']} | {r['nota']}")
    print(f"VERDETTO: {'tutti i valori coincidono' if n_diff == 0 else 'ci sono valori DIVERSI (vedi sopra)'}")
    print(f"Output in {OUTPUT_DIR}")
    print(f"Tempo: {time.time() - t0:.1f} s")


def main():
    rv_signature()
    rv_signature_text_numbers()


if __name__ == "__main__":
    main()
