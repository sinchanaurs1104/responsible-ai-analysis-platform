"""Loads uploaded CSV datasets into pandas DataFrames."""

from pathlib import Path

import pandas as pd

from app.core.exceptions import DatasetValidationError


def load_csv(file_path: str | Path, name: str = "dataset") -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise DatasetValidationError(
            f"{name} file could not be found on disk.",
            details={"path": str(path)},
        )

    if path.suffix != ".csv":
        raise DatasetValidationError(
            f"{name} must be a .csv file.",
            details={"extension": path.suffix},
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise DatasetValidationError(
            f"{name} could not be parsed as a CSV file.",
            details={"underlying_error": str(exc)},
        ) from exc

    if df.empty:
        raise DatasetValidationError(f"{name} is empty.")

    return df
