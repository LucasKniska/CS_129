"""Convert the existing XGBoost pickle into JSON for browser-friendly use."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from xgboost import XGBRegressor


def load_model(model_path: Path) -> XGBRegressor:
    with model_path.open("rb") as fp:
        model = pickle.load(fp)
    if not isinstance(model, XGBRegressor):
        raise TypeError(f"expected XGBRegressor, got {type(model)}")
    return model


def copy_feature_cols(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, dest.open("w", encoding="utf-8") as out:
        out.write(src.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkl",
        type=Path,
        default=Path("models/xgb_model_old_era.pkl"),
        help="path to the trained XGBRegressor pickle",
    )
    parser.add_argument(
        "--columns",
        type=Path,
        default=Path("models/feature_cols_old_era.json"),
        help="JSON containing the feature column order used during training",
    )
    parser.add_argument(
        "--out-model",
        type=Path,
        default=Path("web/data/xgb_model_old_era.json"),
        help="output destination for the model JSON",
    )
    parser.add_argument(
        "--out-columns",
        type=Path,
        default=Path("web/data/feature_cols_old_era.json"),
        help="output destination for the feature columns that the React UI should display",
    )

    args = parser.parse_args()

    print(f"Loading pickle from {args.pkl}")
    model = load_model(args.pkl)
    boost = model.get_booster()  # type: ignore[attr-defined]

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving booster as JSON to {args.out_model}")
    boost.save_model(str(args.out_model))

    print(f"Copying feature list to {args.out_columns}")
    copy_feature_cols(args.columns, args.out_columns)

    print("Conversion complete.")


if __name__ == "__main__":
    main()
