"""Recreate the notebook's structural + TF-IDF model and save it for Streamlit."""

import argparse
import html
from pathlib import Path
import re
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils import CENTER_LAT, CENTER_LON, NUMERIC, STRUCT_CAT  # noqa: E402


def parse_params(value):
    if not isinstance(value, str):
        return {}
    return dict(part.split("<=>", 1) for part in value.split("<br>") if "<=>" in part)


def clean_text(value):
    if not isinstance(value, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def strip_numbers(value):
    return re.sub(r"\d[\d\s.,]*", " ", value)


def number_or_none(value):
    return None if pd.isna(value) else float(value)


def build_artifact(csv_path):
    raw = pd.read_csv(csv_path)
    df = raw[raw["market"] == "secondary"].reset_index(drop=True)
    params = pd.DataFrame([parse_params(value) for value in df["params"]]).replace("", np.nan)
    data = df[["id", "price", "m", "price_per_m", "map_lat", "map_lon", "district_lat", "district_lon"]].copy()
    data["created_at"] = pd.to_datetime(df["created_at"])
    start_date, end_date = data["created_at"].min(), data["created_at"].max()
    data["no_rooms"] = pd.to_numeric(df["no_rooms"].replace("more", "8"))

    floor_map = {"ground_floor": 0, "floor_higher_10": 11}
    floor_map.update({f"floor_{i}": i for i in range(1, 11)})
    data["floor"] = params["floor_no"].map(floor_map)
    for column in ["building_floors_num", "build_year", "rent"]:
        data[column] = pd.to_numeric(params[column], errors="coerce")
    data.loc[~data["build_year"].between(1850, 2022), "build_year"] = np.nan
    data.loc[~data["rent"].between(1, 3000), "rent"] = np.nan

    for column in STRUCT_CAT:
        data[column] = params[column].fillna("missing")
    for column in ["media_types", "security_types", "equipment_types", "extras_types"]:
        dummies = params[column].fillna("").str.get_dummies(sep="<->").drop(columns=["0"], errors="ignore")
        dummies.columns = column.split("_")[0] + "__" + dummies.columns
        data = data.join(dummies)

    default_point = (data["map_lat"].round(5) == 52.38356) & (data["map_lon"].round(4) == 16.9408)
    data.loc[default_point, ["map_lat", "map_lon"]] = np.nan
    dy = (data["map_lat"] - CENTER_LAT) * 111
    dx = (data["map_lon"] - CENTER_LON) * 111 * np.cos(np.radians(CENTER_LAT))
    data["dist_center_km"] = np.sqrt(dx ** 2 + dy ** 2)
    data["months_since_start"] = (data["created_at"] - start_date).dt.days / 30.4
    data["m_per_room"] = data["m"] / data["no_rooms"]
    data["floor_ratio"] = data["floor"] / data["building_floors_num"]

    data["text"] = (df["title"].fillna("") + ". " + df["description"].map(clean_text)).str.strip(". ")
    data["desc_start"] = df["description"].map(clean_text).str.lower().str[:200]
    valid = data["m"].between(15, 250) & data["price_per_m"].between(2500, 20000) & (data["price"] >= 50000)
    data = data[valid].reset_index(drop=True)
    data["group"] = data.groupby(["m", "desc_start"]).ngroup()
    rows = np.sort(data.groupby("group")["created_at"].idxmax().values)
    data = data.loc[rows].reset_index(drop=True)
    groups = data["group"].unique()
    test_groups = np.random.RandomState(42).choice(groups, size=int(0.2 * len(groups)), replace=False)
    data["split"] = np.where(data["group"].isin(test_groups), "test", "train")
    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]

    flag_columns = [column for column in data.columns if "__" in column]
    struct_cols = NUMERIC + STRUCT_CAT + flag_columns
    X = data[struct_cols].copy()
    categories = {}
    for column in STRUCT_CAT:
        counts = X.loc[train.index, column].value_counts()
        X[column] = X[column].where(X[column].isin(counts[counts >= 15].index), "other")
        categories[column] = sorted(X.loc[train.index, column].unique().tolist())
        X[column] = pd.Categorical(X[column], categories=categories[column])

    y_train = train["price_per_m"].values
    text_train = train["text"].map(strip_numbers).str.lower()
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    text_matrix = vectorizer.fit_transform(text_train)
    text_model = Ridge(alpha=3)
    # The notebook trains the downstream model with out-of-fold text predictions.
    oof = cross_val_predict(text_model, text_matrix, y_train, cv=GroupKFold(n_splits=5), groups=train["group"])
    text_model.fit(text_matrix, y_train)
    X_train = X.loc[train.index].copy()
    X_train["txt_tfidf"] = oof
    model = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=31, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.7, random_state=42, verbose=-1,
    ).fit(X_train, y_train)

    presets = []
    for number, (index, row) in enumerate(test.sample(n=min(3, len(test)), random_state=42).iterrows(), 1):
        values = {column: number_or_none(row[column]) for column in NUMERIC if column not in {
            "dist_center_km", "months_since_start", "m_per_room", "floor_ratio"
        }}
        values.update({column: None if pd.isna(X.loc[index, column]) else str(X.loc[index, column]) for column in STRUCT_CAT})
        values["flags"] = [column for column in flag_columns if row[column] == 1]
        values["created_at"] = row["created_at"].date().isoformat()
        # Descriptions can contain personal data. Presets deliberately do not include them.
        values["description"] = ""
        presets.append({"label": f"Przykład {number} · {row['m']:g} m² · {row['no_rooms']:g} pok.", "values": values})

    return {
        "format_version": 1,
        "model": model,
        "vectorizer": vectorizer,
        "text_model": text_model,
        "feature_columns": list(X_train.columns),
        "flag_columns": flag_columns,
        "categories": categories,
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "presets": presets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT.parent / "olx_price_modelling/data/real_estate_ads.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/poznan_valuation.joblib")
    parser.add_argument("--force", action="store_true", help="Zastąp istniejący plik modelu.")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Nie znaleziono danych: {args.data}. Podaj ścieżkę przez --data.")
    if args.output.exists() and not args.force:
        parser.error(f"Plik już istnieje: {args.output}. Aby go zastąpić, użyj --force.")
    print(f"Przygotowanie modelu na podstawie: {args.data}")
    artifact = build_artifact(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    print(f"Zapisano model i presety: {args.output}")


if __name__ == "__main__":
    main()
