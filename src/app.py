"""
Sistema de agrupación de casas: uso en producción de los modelos entrenados.

Asigna un cluster a casas nuevas a partir de su latitud, longitud e ingreso medio,
con el clasificador supervisado y, como referencia, con el K-Means original.

Uso (desde la raíz del repositorio):
    python src/app.py 37.77 -122.42 8.5        # latitud longitud ingreso_medio
    python src/app.py                          # demo con varias casas de ejemplo
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KMEANS_PATH = PROJECT_ROOT / "models" / "kmeans_6_clusters.joblib"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "cluster_classifier.joblib"
FEATURES = ["Latitude", "Longitude", "MedInc"]

# Casas de ejemplo (ingreso medio en decenas de miles de dólares)
DEMO_HOUSES = pd.DataFrame(
    [
        ("San Francisco, ingreso alto", 37.77, -122.42, 9.0),
        ("Oakland, ingreso medio", 37.80, -122.27, 3.5),
        ("Los Ángeles, ingreso medio", 34.05, -118.24, 3.0),
        ("Beverly Hills, ingreso alto", 34.07, -118.40, 12.0),
        ("Fresno (interior), ingreso bajo", 36.74, -119.79, 2.0),
        ("Eureka (norte), ingreso medio", 40.80, -124.16, 3.2),
    ],
    columns=["descripcion", *FEATURES],
)


def load_models():
    """Carga ambos modelos una sola vez."""
    for path in (KMEANS_PATH, CLASSIFIER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"No se encuentra {path}. Ejecuta antes src/explore.ipynb.")
    return joblib.load(KMEANS_PATH), joblib.load(CLASSIFIER_PATH)


def parse_args(args: list[str]) -> pd.DataFrame:
    """Convierte 'lat lon ingreso' de la línea de comandos en un DataFrame de una fila."""
    if len(args) != 3:
        raise SystemExit("Uso: python src/app.py <latitud> <longitud> <ingreso_medio>")
    try:
        lat, lon, inc = (float(a) for a in args)
    except ValueError:
        raise SystemExit("Los tres valores deben ser números, p. ej.: 37.77 -122.42 8.5") from None
    return pd.DataFrame([("Casa indicada", lat, lon, inc)], columns=["descripcion", *FEATURES])


def assign_clusters(kmeans, classifier, houses: pd.DataFrame) -> pd.DataFrame:
    """Añade el cluster según el clasificador (producción) y según K-Means (referencia)."""
    result = houses.copy()
    result["cluster"] = classifier.predict(houses[FEATURES])
    result["cluster_kmeans"] = kmeans.predict(houses[FEATURES])
    return result


def main(args: list[str]) -> None:
    houses = parse_args(args) if args else DEMO_HOUSES
    kmeans, classifier = load_models()
    result = assign_clusters(kmeans, classifier, houses)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
