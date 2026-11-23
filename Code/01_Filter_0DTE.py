"""
01_Filter_0DTE.py: dai file grezzi Cboe alle sole quotazioni 0DTE della sessione.

Per ogni giornata, in un solo passaggio DuckDB:
  1. tiene le righe 0DTE (data della quotazione = data di scadenza);
  2. sostituisce quote_datetime con quote_date e quote_time e aggiunge expiration_time
     (16:00:00, 13:00:00 nelle chiusure anticipate);
  3. toglie le quotazioni successive alla chiusura.
L'ordine delle righe e i valori restano quelli del file Cboe.

Input:  CBOE/Data/00_inputs/CBOE_raw/UnderlyingOptionsIntervals_60sec_calcs_oi_YYYY-MM-DD.zip
        (come scaricati da Cboe DataShop) oppure gli stessi file gia' scompattati (.csv)
Output: CBOE/Data/01_filter_0dte/
"""

import csv
import re
import tempfile
import time
import zipfile
from pathlib import Path

import duckdb

from common import INPUT_DIR, YEAR, close_time, output_dir

RAW_DIR = INPUT_DIR / "CBOE_raw"
OUTPUT_DIR = output_dir("01_filter_0dte")
REPORT_FILE = OUTPUT_DIR / f"01_filter_report_{YEAR}.csv"

THREADS = 8

RAW_PATTERN = re.compile(r"UnderlyingOptionsIntervals_60sec_calcs_oi_(\d{4})-(\d{2})-(\d{2})\.(csv|zip)$")

# Controllo di plausibilita': in un giorno normale le righe 0DTE sono circa 455 mila.
EXPECTED_0DTE_ROWS = (300_000, 650_000)


def sql_path(path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_raw(path) -> str:
    # tutte le colonne come testo: i valori restano quelli del file Cboe
    return f"""
        read_csv(
            {sql_path(path)},
            header = true,
            all_varchar = true,
            ignore_errors = false
        )
    """


def raw_inputs() -> list[tuple[str, Path]]:
    """Un file per giornata: il CSV se c'e', altrimenti lo zip."""
    by_day = {}
    for f in sorted(RAW_DIR.glob(f"UnderlyingOptionsIntervals_60sec_calcs_oi_{YEAR}-*")):
        m = RAW_PATTERN.search(f.name)
        if m:
            day = "-".join(m.groups()[:3])
            if m.group(4) == "csv" or day not in by_day:
                by_day[day] = f
    return sorted(by_day.items())


def unzip_csv(zip_file: Path, folder: str) -> Path:
    with zipfile.ZipFile(zip_file) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{zip_file.name}: atteso un solo CSV, trovati {len(members)}")
        return Path(z.extract(members[0], folder))


def raw_columns(con, path) -> list[str]:
    """Nomi delle colonne cosi' come li legge DuckDB."""
    return [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {read_raw(path)}").fetchall()]


def datetime_column(columns: list[str]) -> str:
    # i file Cboe usano quote_datetime oppure quoted_datetime
    for name in ("quote_datetime", "quoted_datetime"):
        if name in columns:
            return name
    raise ValueError("Non trovo la colonna quote_datetime o quoted_datetime.")


def output_columns(columns: list[str], dt_col: str) -> str:
    """
    SELECT finale: quote_datetime diventa quote_date e quote_time; dopo expiration
    arriva expiration_time; tutte le altre colonne restano nell'ordine originale.
    """
    parts = []
    for col in columns:
        col_sql = sql_identifier(col)
        if col == dt_col:
            parts.append(f"substr(trim({col_sql}), 1, 10) AS quote_date")
            parts.append(f"substr(trim({col_sql}), 12, 8) AS quote_time")
        elif col == "expiration":
            parts.append(col_sql)
            parts.append("expiration_time")
        else:
            parts.append(col_sql)
    return ",\n            ".join(parts)


def filter_day(con, raw_file, output_file, session_close: str) -> dict:
    columns = raw_columns(con, raw_file)
    dt_col = datetime_column(columns)
    if "expiration" not in columns:
        raise ValueError(f"Non trovo la colonna expiration in {raw_file.name}")

    dt_sql = sql_identifier(dt_col)
    is_0dte = f"substr(trim({dt_sql}), 1, 10) = trim({sql_identifier('expiration')})"
    quote_time = f"trim(substr(trim({dt_sql}), 12, 8))"
    in_session = f"{quote_time} <= {sql_string(session_close)}"

    total_rows, rows_0dte, rows_kept, first_time, last_time = con.execute(f"""
        SELECT
            COUNT(*),
            SUM(CASE WHEN {is_0dte} THEN 1 ELSE 0 END),
            SUM(CASE WHEN {is_0dte} AND {in_session} THEN 1 ELSE 0 END),
            MIN(CASE WHEN {is_0dte} AND {in_session} THEN {quote_time} END),
            MAX(CASE WHEN {is_0dte} AND {in_session} THEN {quote_time} END)
        FROM {read_raw(raw_file)}
    """).fetchone()
    rows_0dte = rows_0dte or 0
    rows_kept = rows_kept or 0

    if output_file.exists():
        output_file.unlink()

    con.execute(f"""
        COPY (
            WITH src AS (
                SELECT
                    *,
                    {sql_string(session_close)} AS expiration_time
                FROM {read_raw(raw_file)}
                WHERE {is_0dte}
            )
            SELECT
                {output_columns(columns, dt_col)}
            FROM src
            WHERE {in_session}
        )
        TO {sql_path(output_file)}
        (
            HEADER,
            DELIMITER ','
        )
    """)

    return {
        "total_rows": total_rows,
        "rows_0dte": rows_0dte,
        "rows_removed_after_close": rows_0dte - rows_kept,
        "rows_kept": rows_kept,
        "first_time_kept": first_time,
        "last_time_kept": last_time,
    }


def main():
    raw_files = raw_inputs()
    if not raw_files:
        raise FileNotFoundError(f"Nessun file grezzo Cboe del {YEAR} (zip o CSV) in {RAW_DIR}")

    con = duckdb.connect()
    con.execute(f"SET threads = {THREADS}")
    # mantiene l'ordine originale delle righe del file grezzo
    con.execute("SET preserve_insertion_order = true")

    print("=" * 100)
    print("FILTRO 0DTE: righe 0DTE, quote_date/quote_time, expiration_time, taglio alla chiusura")
    print(f"Input:  {RAW_DIR} ({len(raw_files)} file)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 100)

    start_all = time.time()
    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "input_file", "output_file", "expiration_time", "total_rows", "rows_0dte",
            "rows_removed_after_close", "rows_kept", "first_time_kept", "last_time_kept",
            "elapsed_seconds",
        ])

        for i, (day, raw_file) in enumerate(raw_files, start=1):
            y, m, d = day.split("-")
            session_close = close_time(day)
            output_file = OUTPUT_DIR / f"01_0DTE_{y}_{m}_{d}.csv"

            start = time.time()
            # uno zip alla volta, scompattato in una cartella temporanea e poi cancellato
            with tempfile.TemporaryDirectory(prefix="cboe_raw_") as tmp:
                csv_file = unzip_csv(raw_file, tmp) if raw_file.suffix == ".zip" else raw_file
                stats = filter_day(con, csv_file, output_file, session_close)
            elapsed = time.time() - start

            writer.writerow([
                raw_file.name, output_file.name, session_close, stats["total_rows"],
                stats["rows_0dte"], stats["rows_removed_after_close"], stats["rows_kept"],
                stats["first_time_kept"], stats["last_time_kept"], round(elapsed, 2),
            ])

            print(f"[{i}/{len(raw_files)}] {output_file.name}: {stats['rows_kept']:,} righe tenute "
                  f"su {stats['rows_0dte']:,} 0DTE ({stats['total_rows']:,} nel file), "
                  f"orari {stats['first_time_kept']}-{stats['last_time_kept']}, "
                  f"scadenza {session_close}, {elapsed:.1f} s")
            if not EXPECTED_0DTE_ROWS[0] <= stats["rows_0dte"] <= EXPECTED_0DTE_ROWS[1]:
                print("    ATTENZIONE: righe 0DTE lontane dai circa 455 mila attesi.")

    print("=" * 100)
    print(f"Report: {REPORT_FILE}")
    print(f"Tempo totale: {(time.time() - start_all) / 60:.1f} minuti")


if __name__ == "__main__":
    main()
