"""Descarga resultados de carreras y datos de clasificación en bruto de Fórmula 1 desde la API Jolpica-F1.

Uso
-----
    python -m src.data.download_historical                   # por defecto 2018-2026
    python -m src.data.download_historical --start-year 2021 --end-year 2026
    python -m src.data.download_historical --include-qualifying
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from src.config import HISTORY, RAW_DIR

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100
RETRY_WAIT_SEC = 3


# ---------------------------------------------------------------------------
# Funciones auxiliares de bajo nivel
# ---------------------------------------------------------------------------

def fetch_json(url: str, retries: int = 3) -> dict:
    """Obtiene una respuesta de la API de Jolpica, reintentando ante errores transitorios."""
    for attempt in range(1, retries + 1):
        try:
            time.sleep(0.5)  # límite de tasa cortés: Jolpica permite ~2 req/s
            with urlopen(url, timeout=30) as response:
                return json.load(response)
        except (HTTPError, URLError) as exc:
            if attempt == retries:
                raise
            print(f"    [warn] attempt {attempt} failed ({exc}), retrying in {RETRY_WAIT_SEC}s…")
            time.sleep(RETRY_WAIT_SEC)
    raise RuntimeError("unreachable")


def save_json(payload: dict, path: Path) -> Path:
    """Persiste una respuesta de la API exactamente como se recibió."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Resultados de carreras
# ---------------------------------------------------------------------------

def download_race_results(season: int) -> list[Path]:
    """Descarga los resultados de carreras paginados para la temporada (*season*) y devuelve las rutas guardadas."""
    season_dir = RAW_DIR / str(season) / "races"
    saved: list[Path] = []

    total: int | None = None
    offset = 0
    while total is None or offset < total:
        page_path = season_dir / f"results_offset_{offset:04d}.json"
        if page_path.exists():
            page = json.loads(page_path.read_text(encoding="utf-8"))
        else:
            page = fetch_json(
                f"{BASE_URL}/{season}/results.json"
                f"?limit={PAGE_SIZE}&offset={offset}"
            )
            save_json(page, page_path)

        saved.append(page_path)
        meta = page["MRData"]
        total = int(meta["total"])
        returned = sum(
            len(race.get("Results", []))
            for race in meta["RaceTable"].get("Races", [])
        )
        if returned == 0:
            break
        offset += PAGE_SIZE

    return saved


# ---------------------------------------------------------------------------
# Resultados de clasificación
# ---------------------------------------------------------------------------

def download_qualifying_season(season: int) -> list[Path]:
    """Descarga todos los resultados de clasificación para la temporada (*season*), un archivo por página."""
    season_dir = RAW_DIR / str(season) / "qualifying"
    saved: list[Path] = []

    total: int | None = None
    offset = 0
    while total is None or offset < total:
        page_path = season_dir / f"qualifying_offset_{offset:04d}.json"
        if page_path.exists():
            page = json.loads(page_path.read_text(encoding="utf-8"))
        else:
            page = fetch_json(
                f"{BASE_URL}/{season}/qualifying.json"
                f"?limit={PAGE_SIZE}&offset={offset}"
            )
            save_json(page, page_path)

        saved.append(page_path)
        meta = page["MRData"]
        total = int(meta["total"])
        returned = sum(
            len(race.get("QualifyingResults", []))
            for race in meta["RaceTable"].get("Races", [])
        )
        if returned == 0:
            break
        offset += PAGE_SIZE

    return saved


# ---------------------------------------------------------------------------
# Clasificación de la carrera objetivo (ronda individual)
# ---------------------------------------------------------------------------

def download_target_qualifying(year: int, round_: int) -> Path:
    """Descarga los resultados de clasificación para una ronda específica de la temporada objetivo."""
    output = RAW_DIR / str(year) / "qualifying" / f"round_{round_:02d}_qualifying.json"
    if output.exists():
        print(f"    [skip] {output} already exists")
        return output
    payload = fetch_json(
        f"{BASE_URL}/{year}/{round_}/qualifying.json?limit=100"
    )
    races = payload["MRData"].get("RaceTable", {}).get("Races", [])
    if not races or not races[0].get("QualifyingResults"):
        raise RuntimeError(
            f"Qualifying results for {year} round {round_} are not yet available."
        )
    return save_json(payload, output)


# ---------------------------------------------------------------------------
# Calendario de la temporada
# ---------------------------------------------------------------------------

def download_schedule(season: int) -> Path:
    path = RAW_DIR / str(season) / "schedule.json"
    if path.exists():
        return path
    payload = fetch_json(f"{BASE_URL}/{season}.json")
    return save_json(payload, path)


# ---------------------------------------------------------------------------
# CLI (Interfaz de línea de comandos)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download historical F1 race and qualifying data from Jolpica."
    )
    p.add_argument("--start-year", type=int, default=HISTORY["start_year"])
    p.add_argument("--end-year", type=int, default=HISTORY["end_year"])
    p.add_argument(
        "--include-qualifying",
        action="store_true",
        help="Also download qualifying results for each season.",
    )
    p.add_argument(
        "--target-qualifying",
        action="store_true",
        help="Download qualifying for the target race defined in race_config.yaml.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")

    for season in range(args.start_year, args.end_year + 1):
        download_schedule(season)
        paths = download_race_results(season)
        msg = f"{season}: {len(paths)} race result page(s)"
        if args.include_qualifying:
            q_paths = download_qualifying_season(season)
            msg += f", {len(q_paths)} qualifying page(s)"
        print(msg)

    if args.target_qualifying:
        from src.config import TARGET
        path = download_target_qualifying(TARGET["year"], TARGET["round"])
        print(f"target qualifying: {path}")


if __name__ == "__main__":
    main()

