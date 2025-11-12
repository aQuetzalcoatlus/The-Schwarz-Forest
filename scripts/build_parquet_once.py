# scripts/build_parquet_once.py
from __future__ import annotations

import io
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

BASE_HIST = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/monthly/kl/historical"
BASE_REC = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/monthly/kl/recent"

TARGETS = {
    "Freiburg": ["FREIBURG", "FREIBURG IM BREISGAU", "HERDERN"],
    "Karlsruhe": ["KARLSRUHE", "RHEINSTETTEN"],
    "Freudenstadt": ["FREUDENSTADT"],  # highland proxy with long post-1990 record
}


def _download_text(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=45)
        if r.ok:
            # DWD uses latin1; keep errors='ignore' for rare bytes
            return r.content.decode("latin1", errors="ignore")
    except requests.RequestException:
        pass
    return None


def fetch_station_catalog() -> pd.DataFrame:
    """
    Robust reader for the monthly KL station description table.
    Handles: mixed header/data delimiters, BOM, comments, odd encodings.
    Strategy: detect header line, split header by whitespace to get names,
    then read remaining lines with semicolon; fallback to whitespace if needed.
    """
    urls = [
        f"{BASE_REC}/KL_Monatswerte_Beschreibung_Stationen.txt",
        f"{BASE_HIST}/KL_Monatswerte_Beschreibung_Stationen.txt",
    ]

    # Download
    text = None
    for url in urls:
        try:
            r = requests.get(url, timeout=45)
            if r.ok and r.content:
                text = r.content.decode("latin1", errors="ignore")
                break
        except requests.RequestException:
            pass
    if not text:
        raise RuntimeError("Could not download DWD station description file.")

    # Strip comments/empties; find header line
    lines_all = text.splitlines()
    lines = [ln for ln in lines_all if ln.strip() and not ln.strip().startswith("#")]

    # Find header line (contains 'Stations_id' and 'Stationsname' etc.)
    hdr_idx = None
    for i, ln in enumerate(lines):
        if "Stations_id" in ln and "Stationsname" in ln:
            hdr_idx = i
            break
    if hdr_idx is None:
        raise RuntimeError("Could not locate header in station description file.")

    header_raw = lines[hdr_idx].replace("\ufeff", "").strip()
    # Header is usually whitespace-separated
    header_names = [h for h in re.split(r"\s+", header_raw) if h]

    # Body (data rows) start after the header
    body = "\n".join(lines[hdr_idx + 1 :])

    # Try semicolon first (normal for data rows)
    try:
        df = pd.read_csv(
            io.StringIO(body), sep=";", engine="python", header=None, names=header_names
        )
    except Exception:
        # Fallback: whitespace for everything
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep=r"\s+", engine="python")

    # Normalize headers
    def norm(s: str) -> str:
        s = s.replace("\ufeff", "").strip()
        s = s.replace(" ", "_").replace("__", "_")
        return s.lower()

    orig_cols = list(df.columns)
    df.columns = [norm(str(c)) for c in df.columns]

    # Map variants → canonical
    col_map = {
        "stations_id": "station_id",
        "stationsid": "station_id",
        "stationsname": "station_name",
        "bundesland": "state",
        "von_datum": "from_date",
        "bis_datum": "to_date",
        "stationhoehe": "elev_m",
        "stationshoehe": "elev_m",
        "geobreite": "lat",
        "geolaenge": "lon",
        "longitude": "lon",
        "latitude": "lat",
    }
    for k, v in col_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Ensure required columns exist
    for col in [
        "station_id",
        "station_name",
        "from_date",
        "to_date",
        "lat",
        "lon",
        "elev_m",
        "state",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce numerics
    for col in ["station_id", "from_date", "to_date", "lat", "lon", "elev_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional BW filter by state if available; else leave to bbox later
    if df["state"].notna().any():
        df = df[
            df["state"].astype(str).str.contains("Baden", case=False, na=False)
        ].copy()

    print("[INFO] Station catalog columns:", orig_cols, "→", list(df.columns))
    print(f"[INFO] Catalog rows after optional BW-by-state filter: {len(df)}")

    # Helpful: write what we’ll search through (after state filter) for debugging
    (Path("data/raw").mkdir(parents=True, exist_ok=True))
    df.to_csv("data/raw/stations_catalog_bw.csv", index=False)
    return df


def _normalize_name(s: str) -> str:
    """
    Normalize station names and keywords for robust matching:
      - uppercase
      - strip accents (umlauts, etc.)
      - remove slashes, dots, dashes, spaces
    """
    s = s.upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # strip accents
    s = re.sub(r"[\s\-/\.\(\)]", "", s)
    return s


def resolve_station_ids(catalog: pd.DataFrame) -> dict[str, int]:
    """
    Pick one best station per target using robust name matching.
    Do NOT apply bbox (lat/lon often missing). Rely on state filter done earlier
    and on name normalization.
    Prefer series covering >= 1990 with the latest 'to_date'.
    """
    use = catalog.copy()

    # Build a normalized helper column for searching
    use["name_norm"] = use["station_name"].astype(str).map(_normalize_name)

    # Expand the keyword list with common variants for each city
    expanded_targets = {
        "Freiburg": [
            "FREIBURG",
            "FREIBURGBR",
            "FREIBURGBREISGAU",
            "FREIBURGIMBREISGAU",
            "HERDERN",
        ],
        "Karlsruhe": [
            "KARLSRUHE",
            "RHEINSTETTEN",
            "KARLSRUHERHEINSTETTEN",
        ],
        "Freudenstadt": [
            "FREUDENSTADT",
        ],
    }

    out: dict[str, int] = {}

    for nice, kws in expanded_targets.items():
        kw_norm = [_normalize_name(k) for k in kws]
        # Any keyword match
        mask = False
        for k in kw_norm:
            mask = mask | use["name_norm"].str.contains(k, na=False)
        hits = use[mask].copy()

        if hits.empty:
            print(f"[WARN] No catalog hits for {nice} using keywords {kws}")
            continue

        # Prefer entries whose record runs past 1990 and ends as late as possible
        if "to_date" in hits:
            hits["to_date"] = pd.to_numeric(hits["to_date"], errors="coerce")
            hits["from_date"] = pd.to_numeric(hits["from_date"], errors="coerce")
            good = hits[hits["to_date"].fillna(0) >= 19900101]
            if not good.empty:
                hits = good

        pick = hits.sort_values(["to_date", "from_date"], ascending=[False, True]).iloc[
            0
        ]
        sid = int(pick["station_id"])
        out[nice] = sid

        # Log a tiny preview so we know what matched
        print(
            f"[OK] Resolved {nice} → {sid:05d}  name='{pick['station_name']}'  "
            f"period={int(pick['from_date']) if pd.notna(pick['from_date']) else 'NA'}–"
            f"{int(pick['to_date']) if pd.notna(pick['to_date']) else 'NA'}  "
            f"elev={pick.get('elev_m', 'NA')} lat={pick.get('lat', 'NA')} lon={pick.get('lon', 'NA')}"
        )

    return out


def _try_get(url: str) -> bytes | None:
    try:
        r = requests.get(url, timeout=45)
        if r.ok and r.headers.get("Content-Type", "").lower().startswith(
            "application/zip"
        ):
            return r.content
    except requests.RequestException:
        return None
    return None


def download_zip(station_id: int) -> list[bytes]:
    sid = f"{station_id:05d}"
    # Try common historical spans + recent file name
    trials = [
        f"{BASE_HIST}/monatswerte_KL_{sid}_19900101_20241231_hist.zip",
        f"{BASE_HIST}/monatswerte_KL_{sid}_19610101_20241231_hist.zip",
        f"{BASE_HIST}/monatswerte_KL_{sid}_18810101_20241231_hist.zip",
        f"{BASE_REC}/monatswerte_KL_{sid}_akt.zip",
        # legacy names
        f"{BASE_HIST}/produkt_klima_monat_{sid}_historical.zip",
        f"{BASE_REC}/produkt_klima_monat_{sid}_akt.zip",
    ]
    blobs = []
    for u in trials:
        b = _try_get(u)
        if b:
            blobs.append(b)
        time.sleep(0.2)
    return blobs


def extract_txt_from_zip(blob: bytes) -> list[pd.DataFrame]:
    out = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".txt"):
                continue
            raw = zf.read(name).decode("latin1", errors="ignore")
            df = pd.read_csv(io.StringIO(raw), sep=";")
            df.columns = [c.strip() for c in df.columns]
            out.append(df)
    return out


def monthly_to_annual(frames: list[pd.DataFrame]) -> pd.DataFrame:
    use = []
    for df in frames:
        datecol = next((c for c in df.columns if "MESS_DATUM" in c.upper()), None)
        tempcol = (
            "TMK"
            if "TMK" in df.columns
            else next(
                (
                    c
                    for c in df.columns
                    if c.upper().endswith("TT") or "TMK" in c.upper()
                ),
                None,
            )
        )
        if not datecol or not tempcol:
            continue
        x = df[[datecol, tempcol]].copy()
        x["date"] = pd.to_datetime(x[datecol], errors="coerce")
        x["year"] = x["date"].dt.year
        x["tmean_c"] = pd.to_numeric(x[tempcol], errors="coerce")
        x.loc[x["tmean_c"] <= -900, "tmean_c"] = np.nan
        use.append(x[["year", "tmean_c"]])
    if not use:
        return pd.DataFrame()
    m = pd.concat(use, ignore_index=True).dropna(subset=["year", "tmean_c"])
    m = m[(m["year"] >= 1990) & (m["year"] <= 2024)]
    a = m.groupby("year", as_index=False).agg(
        tmean_c=("tmean_c", "mean"), n_months=("tmean_c", "size")
    )
    return a[a["n_months"] >= 10].copy()


def main():
    catalog = fetch_station_catalog()
    chosen = resolve_station_ids(catalog)
    if not chosen:
        print("[FATAL] Could not resolve any Baden-Württemberg station IDs.")
        return

    all_rows = []
    for nice, sid in chosen.items():
        blobs = download_zip(sid)
        if not blobs:
            print(
                f"[WARN] No ZIPs found for {nice} (id {sid:05d}). Trying anyway with zero frames."
            )
        frames = []
        for b in blobs:
            frames.extend(extract_txt_from_zip(b))
        ann = monthly_to_annual(frames)
        if ann.empty:
            print(f"[WARN] Parsed no valid monthly data for {nice}.")
            continue
        ann["station"] = nice
        all_rows.append(ann)
        print(
            f"[OK] {nice}: {len(ann)} annual rows ({int(ann['year'].min())}–{int(ann['year'].max())})"
        )

    if not all_rows:
        print("No data parsed for any station. Aborting without writing Parquet.")
        return

    df = pd.concat(all_rows, ignore_index=True)
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROC / "dwd_bw_annual.parquet", index=False)
    print("Wrote", PROC / "dwd_bw_annual.parquet", "rows:", len(df))
    print(
        df.groupby("station").agg(
            year_min=("year", "min"), year_max=("year", "max"), rows=("year", "size")
        )
    )


if __name__ == "__main__":
    main()
