"""Adiciona as features de contexto epidemiológico (`local_density` e
`local_positivity`) aos datasets ml e gera os lookups de serving.

Essas features dependem do ano inteiro (contagens/confirmações por
município/semana), então não podem ser calculadas no ETL streaming chunk a
chunk nem numa linha isolada. Este passo roda DEPOIS de
`prepare_dengue_data.py`:

1. Para cada ano, calcula `local_density` e `local_positivity` a partir do
   parquet de análise (alinhado 1:1 com o parquet ml) e grava as colunas no
   parquet ml.
2. Constrói os artefatos de lookup (município, semana-do-ano) -> valor médio,
   usados pela API para uma notificação isolada.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dengue_pipeline.features import (  # noqa: E402
    build_local_density_lookup,
    build_local_positivity_lookup,
    compute_local_density,
    compute_local_positivity,
)
from dengue_pipeline.paths import (  # noqa: E402
    DENGUE_YEARS,
    LOCAL_DENSITY_LOOKUP_PATH,
    LOCAL_POSITIVITY_LOOKUP_PATH,
    analysis_dataset_path,
    ml_dataset_path,
)

_ANALYSIS_COLUMNS = [
    "residence_municipality",
    "notification_epi_week",
    "notification_year",
    "final_classification_code",
]


def _atomic_write(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)


def main() -> None:
    lookup_frames = []
    for year in DENGUE_YEARS:
        analysis_path = analysis_dataset_path(year)
        ml_path = ml_dataset_path(year)
        if not analysis_path.exists() or not ml_path.exists():
            raise FileNotFoundError(
                f"[{year}] datasets ausentes; rode prepare_dengue_data.py antes."
            )

        analysis = pd.read_parquet(analysis_path, columns=_ANALYSIS_COLUMNS)
        density = compute_local_density(analysis)
        positivity = compute_local_positivity(analysis)

        ml = pd.read_parquet(ml_path)
        if len(ml) != len(density):
            raise RuntimeError(
                f"[{year}] ml ({len(ml)}) e análise ({len(density)}) desalinhados"
            )
        ml["local_density"] = density.to_numpy()
        ml["local_positivity"] = positivity.to_numpy()
        _atomic_write(ml, ml_path)

        lookup_frames.append(analysis)
        print(
            f"[{year}] local_density + local_positivity gravadas em "
            f"{ml_path.name} | cobertura dens={pd.notna(density).mean():.3%} "
            f"posit={pd.notna(positivity).mean():.3%}",
            flush=True,
        )

    combined = pd.concat(lookup_frames, ignore_index=True)
    LOCAL_DENSITY_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    density_lookup = build_local_density_lookup(combined)
    positivity_lookup = build_local_positivity_lookup(combined)
    _atomic_write(density_lookup, LOCAL_DENSITY_LOOKUP_PATH)
    _atomic_write(positivity_lookup, LOCAL_POSITIVITY_LOOKUP_PATH)
    print(
        f"Lookups de serving escritos: densidade ({len(density_lookup):,} pares) "
        f"e positividade ({len(positivity_lookup):,} pares)"
    )


if __name__ == "__main__":
    main()
