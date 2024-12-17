import pandas as pd
from glob import glob
import yaml
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from datetimejp import JDatetime
import json
import time
import warnings
from tqdm.auto import tqdm
import os
from geolib import geohash
import numpy as np
from sklearn.preprocessing import LabelEncoder
import requests

warnings.simplefilter("ignore")

INPUT_DIR = "input"
OUTPUT_DIR = "output/preprocess"
COMPEDATA_DIR = f"{INPUT_DIR}/compedata"
EXTERNALDATA_DIR = f"{INPUT_DIR}/externaldata"
CATEGORICAL_COLS = [
    "city_name",
    "district_name",
    "building_use",
    "transaction_detail",
    "geohash7",
    "geohash6",
    "geohash5",
    "geohash4",
    "geohash3",
    "geohash2",
]


def read_data():
    """
    元データ読み込み
    """
    train_files = glob(f"{COMPEDATA_DIR}/train/*.csv")
    train_df = pd.concat([pd.read_csv(file) for file in train_files]).reset_index(drop=True)
    with open(f"{INPUT_DIR}/columns_jp_en.json", "r") as f:
        columns_jp_en = yaml.load(f, Loader=yaml.SafeLoader)
    train_df = train_df.rename(columns=columns_jp_en)

    test_df = pd.read_csv(f"{COMPEDATA_DIR}/test.csv")
    test_df = test_df.rename(columns=columns_jp_en)

    train_df["test"] = 0
    test_df["test"] = 1
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    df.drop(
        columns=[
            "kind",
            "region",
            "land_shape",
            "frontage",
            "building_floor_area",
            "frontroad_direction",
            "frontroad_kind",
            "frontroad_width",
        ],
        inplace=True,
    )
    return df


def get_latlon():
    """
    国土地理院APIによる住所からの経度緯度取得
    """
    df = read_data()
    df["address"] = df["prefecture"] + df["city_name"] + df["district_name"]
    address_list = sorted(df.loc[pd.notnull(df["address"]), "address"].unique().tolist())
    url = "https://msearch.gsi.go.jp/address-search/AddressSearch"
    latlon_list = []
    for address in tqdm(address_list, desc="get_latlon"):
        try:
            params = {"q": address}
            r = requests.get(url, params=params)
            data = r.json()
            if len(data) == 1:
                lat = data[0]["geometry"]["coordinates"][1]
                lon = data[0]["geometry"]["coordinates"][0]
                latlon_list.append([address, lat, lon])
        except:
            lat = 999.999
            lon = 999.999
            latlon_list.append([address, lat, lon])
        time.sleep(1)

    latlon_df = pd.DataFrame(latlon_list, columns=["address", "latitude", "longitude"])
    latlon_df.to_csv(f"{OUTPUT_DIR}/latlon.csv", index=False)
    return latlon_df


def convert_building_year(x):
    try:
        x = JDatetime.strptime(x, "%g%e年").strftime("%Y")
    except:
        if x == "戦前":
            x = "1940"
        pass
    return float(x)


def convert_transaction_time(x):
    try:
        year = x.split("年")[0]
        year = float(year)
        term = x.split("年")[1]
        if term == "第1四半期":
            term = 4
        if term == "第2四半期":
            term = 7
        if term == "第3四半期":
            term = 10
        if term == "第4四半期":
            term = 1
            year += 1
        x = (year * 100) + term
    except:
        pass
    return float(x)


def convert_station_min(x):
    if x in ["30分?60分", "1H30?2H", "1H?1H30", "2H?"]:
        if x == "30分?60分":
            x = 45
        if x == "1H30?2H":
            x = 105
        if x == "1H?1H30":
            x = 75
        if x == "2H?":
            x = 150
    return float(x)


def convert_land_area(x):
    if x == "2000㎡以上":
        x = 300
    return float(x)


def preprocess(df):
    """
    前準備が必要なファイル
    - station.csv: get_station_df()で作成
    - latlon.csv: get_latlon()で作成
    - district_name_null_dict.json: get_district_name_null_dict()で作成
    - station_null_dict.json: get_station_null_dict()で作成
    - nearest_station_dict.json: get_nearest_station_dict()で作成
    """
    with open(f"{OUTPUT_DIR}/district_name_null_dict.json", "r") as f:
        district_name_null_dict = json.load(f)
    df.loc[pd.isnull(df["district_name"]), "district_name"] = df.loc[pd.isnull(df["district_name"]), "id"].map(lambda x: district_name_null_dict.get(str(x)))
    latlon_df = pd.read_csv(f"{OUTPUT_DIR}/latlon.csv")
    df["address"] = df["prefecture"] + df["city_name"] + df["district_name"]
    df = df.merge(latlon_df, on=["address"], how="left")
    df["latitude"] = df["latitude"].fillna(999.999)
    df["longitude"] = df["longitude"].fillna(999.999)

    with open(f"{INPUT_DIR}/encoding/prefecture.json", "r") as f:
        encoder = json.load(f)
    df["prefecture"] = df["prefecture"].map(lambda x: encoder.get(x))
    df["prefecture"] = df["prefecture"].astype(int)

    df.loc[df["station"] == "nan", "station"] = None
    with open(f"{OUTPUT_DIR}/station_null_dict.json", "r") as f:
        station_null_dict = json.load(f)
    df.loc[pd.isnull(df["station"]), "station"] = df.loc[pd.isnull(df["station"]), "id"].map(lambda x: station_null_dict.get(str(x)))
    with open(f"{OUTPUT_DIR}/nearest_station_dict.json", "r") as f:
        nearest_station_dict = json.load(f)
    df.loc[pd.isnull(df["station"]), "station"] = df.loc[pd.isnull(df["station"]), "id"].map(lambda x: nearest_station_dict.get(str(x)))
    station_df = pd.read_csv(f"{OUTPUT_DIR}/station.csv")
    df.loc[pd.notnull(df["station"]), "station"] = df.loc[pd.notnull(df["station"]), "station"].map(lambda x: str(x).split("(")[0])
    df = df.merge(station_df, on=["prefecture", "station"], how="left")

    df.loc[df["longitude"] > 999, "longitude"] = df.loc[df["longitude"] > 999, "station_lon"]
    df.loc[df["latitude"] > 999, "latitude"] = df.loc[df["latitude"] > 999, "station_lat"]

    df["building_year"] = df["building_year"].map(lambda x: convert_building_year(x))
    df["building_year_round5"] = df["building_year"] // 5

    df["transaction_time"] = df["transaction_time"].map(lambda x: convert_transaction_time(x))
    df["building_month"] = (df["transaction_time"] // 100 - df["building_year"]) * 12 + df["transaction_time"] % 100

    with open(f"{INPUT_DIR}/encoding/floor.json", "r") as f:
        encoder = json.load(f)
    df["floor"] = df["floor"].map(lambda x: encoder.get(x))

    with open(f"{INPUT_DIR}/encoding/building_structure.json", "r") as f:
        encoder = json.load(f)
    df["building_structure"] = df["building_structure"].map(lambda x: encoder.get(x))

    with open(f"{INPUT_DIR}/encoding/future_purpose.json", "r") as f:
        encoder = json.load(f)
    df["future_purpose"] = df["future_purpose"].map(lambda x: encoder.get(x))

    with open(f"{INPUT_DIR}/encoding/city_planning.json", "r") as f:
        encoder = json.load(f)
    df["city_planning"] = df["city_planning"].map(lambda x: encoder.get(x))

    with open(f"{INPUT_DIR}/encoding/reform.json", "r") as f:
        encoder = json.load(f)
    df["reform"] = df["reform"].map(lambda x: encoder.get(x))

    df["station_min"] = df["station_min"].map(lambda x: convert_station_min(x))

    df["land_area"] = df["land_area"].map(lambda x: convert_land_area(x))

    df["transaction_year"] = df["transaction_time"] // 100
    df["transaction_month"] = df["transaction_time"] % 100
    return df


def get_station_df():
    station_df = pd.read_csv(f"{EXTERNALDATA_DIR}/station20240426free.csv")
    station_df = station_df[["station_name", "pref_cd", "station_cd", "line_cd", "lon", "lat"]]
    station_df = station_df.rename(
        columns={
            "station_name": "station",
            "pref_cd": "prefecture",
            "lon": "station_lon",
            "lat": "station_lat",
        }
    )
    station_df = station_df.sort_values(["line_cd", "station_cd"]).reset_index(drop=True)
    station_df = station_df.groupby(["prefecture", "station"]).first().reset_index()
    station_df["prefecture"] = station_df["prefecture"].astype(int)
    station_df.to_csv(f"{OUTPUT_DIR}/station.csv", index=False)
    return station_df


def get_district_name_null_dict(df):
    """
    地区名欠損補完
    - 県/駅/駅からの徒歩時間10分丸めで最頻値の駅で補完
    - 元データの駅名は`()`で県表示がされているものもあるので`(`以下を除外
    """
    df["station"] = df["station"].map(lambda x: str(x).split("(")[0])
    df.loc[df["station"] == "nan", "station"] = None
    df["station_min"] = df["station_min"].map(lambda x: convert_station_min(x))
    df["station_min_round"] = df["station_min"] // 10
    station_district_mode_df = (
        df[pd.notnull(df["district_name"])]
        .groupby(["prefecture", "station", "station_min_round"])[["district_name"]]
        .value_counts()
        .reset_index()
        .groupby(["prefecture", "station", "station_min_round"])[["district_name"]]
        .first()
    )

    df_station = df[(pd.isnull(df["district_name"])) & (pd.notnull(df["station"]))][["id", "prefecture", "city_name", "station", "station_min_round"]]
    df_station = df_station.merge(
        station_district_mode_df,
        on=["prefecture", "station", "station_min_round"],
        how="left",
    )
    df_station = df_station[pd.notnull(df_station["district_name"])]
    district_name_null_dict = df_station.set_index("id")["district_name"].to_dict()
    with open(f"{OUTPUT_DIR}/district_name_null_dict.json", "w") as f:
        json.dump(district_name_null_dict, f, indent=4, ensure_ascii=False)
    return district_name_null_dict


def get_station_null_dict(df):
    """
    駅名欠損補完
    - 住所で最頻値の駅で補完
    - 元データの駅名は`()`で県表示がされているものもあるので`(`以下を除外
    """
    district_station_mode_df = (
        df.groupby(["prefecture", "city_name", "district_name"])["station"].value_counts().reset_index().groupby(["prefecture", "city_name", "district_name"])["station"].first().reset_index()
    )

    station_null_df = df[pd.isnull(df["station"])][["id", "prefecture", "city_name", "district_name"]]
    station_null_df = station_null_df.merge(
        district_station_mode_df,
        on=["prefecture", "city_name", "district_name"],
        how="left",
    )
    station_null_df = station_null_df[pd.notnull(station_null_df["station"])]
    station_null_dict = station_null_df.set_index("id")["station"].to_dict()
    with open(f"{OUTPUT_DIR}/station_null_dict.json", "w") as f:
        json.dump(station_null_dict, f, indent=4, ensure_ascii=False)
    return station_null_dict


def search_nearest_station(lat, lon, station_latlon_df):
    """
    住所緯度経度から一番近い距離の駅を探す。
    距離算出にはGeopyを使用
    """
    nearest_station = ["", ""]
    nearest_dis = 9999999
    for _, row in station_latlon_df.iterrows():
        prefecture = row["prefecture"]
        station = row["station"]
        station_lat = row["station_lat"]
        station_lon = row["station_lon"]
        dis = geodesic((lat, lon), (station_lat, station_lon)).km
        if dis < nearest_dis:
            nearest_station = [prefecture, station]
            nearest_dis = dis
    return nearest_station


def get_nearest_station_dict(df):
    """
    駅名欠損補完
    - 住所で最頻値の駅で補完
    """
    df["address"] = df["prefecture"] + df["city_name"] + df["district_name"]
    latlon_df = pd.read_csv(f"{OUTPUT_DIR}/latlon.csv")
    df = df.merge(latlon_df, on=["address"], how="left")
    station_null_df = df[pd.isnull(df["station"])][["id", "latitude", "longitude"]].reset_index(drop=True)
    station_latlon_df = pd.read_csv(f"{OUTPUT_DIR}/station.csv")
    nearest_station_dict = {}
    for _, row in tqdm(
        station_null_df.iterrows(),
        total=len(station_null_df),
        desc="get_nearest_station_dict",
    ):
        try:
            id = row["id"]
            lat = row["latitude"]
            lon = row["longitude"]
            nearest_station = search_nearest_station(lat, lon, station_latlon_df)
            nearest_station_dict[id] = nearest_station
        except:
            pass
    nearest_station_df = pd.DataFrame.from_dict(nearest_station_dict, orient="index", columns=["prefecture", "station"])
    nearest_station_df = nearest_station_df.reset_index().rename(columns={"index": "id"})
    nearest_station_df["id"] = nearest_station_df["id"].astype(int)
    nearest_station_df = nearest_station_df[["id", "station"]]
    nearest_station_dict = nearest_station_df.set_index("id")["station"].to_dict()
    with open(f"{OUTPUT_DIR}/nearest_station_dict.json", "w") as f:
        json.dump(nearest_station_dict, f, indent=4, ensure_ascii=False)
    return nearest_station_dict


def get_station_distance(lat, lon, sta_lat, sta_lon):
    dis = None
    try:
        dis = geodesic((lat, lon), (sta_lat, sta_lon)).km
    except:
        pass
    return dis


def get_geohash(latitude, longitude, precision):
    h = None
    try:
        h = geohash.encode(latitude, longitude, precision)
    except:
        pass
    return h


def create_agg_feature(df, groupby_cols, target_cols, aggs, desc=""):
    agg_cols = []
    for g in groupby_cols:
        for t in target_cols:
            for a in aggs:
                agg_d = {}
                agg_d["groupby"] = g
                agg_d["target"] = t
                agg_d["agg"] = a
                agg_cols.append(agg_d)
    df, new_cols = agg(df.copy(), agg_cols, desc=desc)
    return df, new_cols


def agg(df, agg_cols, desc=""):
    old_cols = list(df.columns)
    for c in tqdm(agg_cols, desc=desc):
        new_feature = "{}_{}_{}".format("_".join(c["groupby"]), c["agg"], c["target"])
        df[new_feature] = df.groupby(c["groupby"])[c["target"]].transform(c["agg"])
    new_cols = list(set(list(df.columns)) - set(old_cols))
    return df, new_cols


def create_features(df):
    df["station_distance"] = df[["latitude", "longitude", "station_lat", "station_lon"]].apply(
        lambda x: get_station_distance(x["latitude"], x["longitude"], x["station_lat"], x["station_lon"]),
        axis=1,
    )
    df["geohash7"] = df[["latitude", "longitude"]].apply(lambda x: get_geohash(x["latitude"], x["longitude"], 7), axis=1)
    df["geohash6"] = df[["latitude", "longitude"]].apply(lambda x: get_geohash(x["latitude"], x["longitude"], 6), axis=1)
    df["geohash5"] = df[["latitude", "longitude"]].apply(lambda x: get_geohash(x["latitude"], x["longitude"], 5), axis=1)
    df["geohash4"] = df[["latitude", "longitude"]].apply(lambda x: get_geohash(x["latitude"], x["longitude"], 4), axis=1)
    df["geohash3"] = df[["latitude", "longitude"]].apply(lambda x: get_geohash(x["latitude"], x["longitude"], 3), axis=1)
    df["geohash2"] = df[["latitude", "longitude"]].apply(lambda x: get_geohash(x["latitude"], x["longitude"], 2), axis=1)
    groupby_cols = [
        ["prefecture"],
        ["city_cd"],
        ["station"],
        ["building_year"],
        ["building_year_round5"],
        ["geohash7"],
        ["geohash6"],
        ["geohash5"],
        ["geohash4"],
        ["geohash3"],
        ["geohash2"],
    ]
    target_cols = ["id"]
    aggs = ["count"]
    df, _ = create_agg_feature(df, groupby_cols, target_cols, aggs)

    groupby_cols = [
        ["prefecture"],
        ["city_cd"],
        ["geohash7"],
        ["geohash6"],
        ["geohash5"],
        ["geohash4"],
        ["geohash3"],
        ["geohash2"],
    ]
    target_cols = ["station_cd", "line_cd"]
    aggs = ["nunique"]
    df, _ = create_agg_feature(df, groupby_cols, target_cols, aggs)

    groupby_cols = [
        ["prefecture"],
        ["city_cd"],
        ["geohash7"],
        ["geohash6"],
        ["geohash5"],
        ["geohash4"],
        ["geohash3"],
        ["geohash2"],
    ]
    target_cols = ["station_min", "building_month"]
    aggs = ["mean"]
    df, _ = create_agg_feature(df, groupby_cols, target_cols, aggs)

    df["transaction_price_total"] = 10 ** df["transaction_price_total_log"]
    df["transaction_price_per_area"] = df["transaction_price_total"] / df["land_area"]
    df["transaction_price_per_area_log"] = np.log1p(df["transaction_price_per_area"])

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


if __name__ == "__main__":
    df = read_data()
    if not os.path.exists(f"{OUTPUT_DIR}/station.csv"):
        station_df = get_station_df()
    if not os.path.exists(f"{OUTPUT_DIR}/latlon.csv"):
        latlon_df = get_latlon()
    if not os.path.exists(f"{OUTPUT_DIR}/district_name_null_dict.json"):
        district_name_null_dict = get_district_name_null_dict(df)
    if not os.path.exists(f"{OUTPUT_DIR}/station_null_dict.json"):
        station_null_dict = get_station_null_dict(df)
    if not os.path.exists(f"{OUTPUT_DIR}/nearest_station_dict.json"):
        nearest_station_dict = get_nearest_station_dict(df)
    df = preprocess(df)
    df = create_features(df)
    df.to_csv(f"{OUTPUT_DIR}/df.csv", index=False)
