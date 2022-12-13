import logging
import os
import sys
import pickle
from typing import Dict, Tuple, List

v = sys.version_info

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from config import MAIN_DIR

pd.set_option('display.max_rows', 500)


def unify_job_titles(df: pd.DataFrame) -> pd.DataFrame:

    data_scientists = [
        "Lead Data Scientist",
        "Data Science Consultant",
        "Director of Data Science",
        "Principal Data Scientist",
        "Data Science Manager",
        "Head of Data",
        "Applied Data Scientist",
        "Data Science Engineer",
        "Head of Data Science",
        "Data Specialist",
        "Staff Data Scientist",
        "Research Scientist",
    ]

    ml_engineers = [
        "Machine Learning Scientist",
        "Machine Learning Manage",
        "Machine Learning Infrastructure Engineer",
        "ML Engineer",
        "AI Scientist",
        "Computer Vision Engineer",
        "3D Computer Vision Researcher",
        "Computer Vision Software Engineer",
        "Machine Learning Developer",
        "Applied Machine Learning Scientist",
        "Head of Machine Learning",
        "NLP Engineer",
        "Lead Machine Learning Engineer",
        "Machine Learning Manager",
    ]

    data_analysts = [
        "Product Data Analyst",
        "Data Analytics Lead",
        "Analytics Engineer",
        "Principal Data Analyst",
        "Finance Data Analyst",
        "Financial Data Analyst",
        "Marketing Data Analyst",
        "BI Data Analyst",
        "Business Data Analyst",
        "Lead Data Analyst",
        "Data Analytics Manager",
        "Data Analytics Engineer",
    ]

    data_engineers = [
        "Big Data Engineer",
        "Lead Data Engineer",
        "Data Engineering Manager",
        "ETL Developer",
        "Big Data Architect",
        "Data Architect",
        "Principal Data Engineer",
        "Director of Data Engineering",
        "Cloud Data Engineer",
    ]

    for role in data_scientists:
        df.loc[df["job_title"] == role, "job_title"] = "Data Scientist"
    for role in ml_engineers:
        df.loc[df["job_title"] == role, "job_title"] = "Machine Learning Engineer"
    for role in data_analysts:
        df.loc[df["job_title"] == role, "job_title"] = "Data Analyst"
    for role in data_engineers:
        df.loc[df["job_title"] == role, "job_title"] = "Data Engineer"

    return df


def encode_features(features: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnTransformer]:

    columns_to_encode = [
        "experience_level",
        "employment_type",
        "job_title",
        "company_size",
        "remote_ratio",
    ]

    ct = ColumnTransformer(
        [
            ("encoder", OrdinalEncoder(), columns_to_encode),
        ],
        remainder="passthrough",
    )

    transformed_data = ct.fit_transform(features)

    return transformed_data, ct


def read_and_transform_data(
    file_name: os.path = os.path.join(MAIN_DIR, "data/ds_salaries.csv")
) -> Tuple[np.ndarray, pd.core.series.Series, ColumnTransformer]:
    df = pd.read_csv(file_name, index_col=0)

    cols_to_remove = ["salary", "work_year"]
    df.drop(cols_to_remove, axis=1, inplace=True)

    df["remote_ratio"] = df["remote_ratio"].map(
        {0: "No Remote", 50: "Partially Remote", 100: "Fully Remote"}
    )

    df = unify_job_titles(df)

    high_cardinality_cols = [
        col
        for col in df.columns
        if df[col].nunique() > 10 and df[col].dtype == "object"
    ]
    df.drop(high_cardinality_cols, axis=1, inplace=True)

    filtered_df = df[(np.abs(stats.zscore(df["salary_in_usd"])) < 3)]

    x = filtered_df.drop(["salary_in_usd"], axis=1)
    y = filtered_df["salary_in_usd"]

    encoded_features, ct = encode_features(x)

    categorize_salary(filtered_df)

    return encoded_features, y, ct


def prepare_model(x: np.ndarray, y: np.ndarray) -> lgbm.basic.Booster:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    lgbm_train = lgbm.Dataset(x_train, y_train)
    lgbm_test = lgbm.Dataset(x_test, y_test, reference=lgbm_train)

    params = {
        "task": "train",
        "objective": "regression",
        "boosting": "gbdt",
        "metric": "l1",
        "num_iterations": 100,
        "num_leaves": 24,
        "learning_rate": 0.09,
        "min_data_in_leaf": 20,
        "max_depth": 4,
        "force_row_wise": True,
    }

    model = lgbm.train(
        params, train_set=lgbm_train, valid_sets=lgbm_test, early_stopping_rounds=20
    )

    return model


def export_model(model: lgbm.basic.Booster) -> None:
    with open(os.path.join(MAIN_DIR, "dump/model.pkl"), "wb") as f:
        pickle.dump(model, f)


def load_model() -> lgbm.basic.Booster:
    if os.path.exists(os.path.join(MAIN_DIR, "dump/model.pkl")):
        with open("dump/model.pkl", "rb") as f:
            return pickle.load(f)


def get_model_stats(y_test, y_pred) -> None:

    stats = {}

    stats["mse"] = mean_squared_error(y_test, y_pred)
    stats["mae"] = mean_absolute_error(y_test, y_pred)
    stats["rmse"] = stats.get("mse") ** 0.5
    stats["r2"] = r2_score(y_test, y_pred)

    # print(f"MSE  = {stats['mse']:.3f}")
    # print(f"MAE  = {stats['mae']:.3f}")
    # print(f"RMSE = {stats['rmse']:.3f}")
    # print(f"R2   = {stats['r2']:.3f}")

    # x_ax = range(len(y_test))
    # plt.figure(figsize=(12, 6))
    # plt.plot(x_ax, y_test, label="original")
    # plt.plot(x_ax, y_pred, label="predicted")
    # plt.title("Test and predicted data")
    # plt.xlabel("X")
    # plt.ylabel("Salary")
    # plt.legend(loc="best", fancybox=True, shadow=True)
    # plt.grid(True)
    # plt.show()


def show_model_importance(model: lgbm.basic.Booster) -> None:
    lgbm.plot_importance(model)
    plt.show()


def setup(retrain=False) -> Tuple[lgbm.basic.Booster, ColumnTransformer]:
    x, y, ct = read_and_transform_data()

    model = load_model()

    if model is None or retrain:
        model = prepare_model(x, y)
        export_model(model)

    return model, ct


# TODO: Create classes from salary column, so that the problem will become
# multiclass clasification, not regression


def main() -> None:
    setup()


def split_salary(max_salary: int, bins_amount: int) -> List:
    step = max_salary // bins_amount

    salary_arr = [i * step for i in range(1, bins_amount + 1)]

    return salary_arr


def prepare_categories(df: pd.DataFrame, bins_amount: int, max_salary: int) -> Dict:
    salaries = df["salary_in_usd"]
    step = max_salary // bins_amount

    categories = {}
    SALARY_ARR = split_salary(max_salary, bins_amount)
    print(SALARY_ARR)

    for i in df.index:
        salary = df["salary_in_usd"][i]
        low = SALARY_ARR[0]
        high = low

        # salary_bin = salary // step
        df['salary_category'] = salary // step


    # print(f"Salary = {salary}")
    # print(f"Bin = {salary_bin}")


def categorize_salary(df: pd.DataFrame) -> pd.DataFrame:
    # df.sort_values(by=['salary_in_usd'], inplace=True)
    df['salary_category'] = ""

    min_salary = df["salary_in_usd"].min()
    max_salary = df["salary_in_usd"].max()

    BINS_AMOUNT = 10

    divisions = BINS_AMOUNT - 1

    BIN_SIZE = max_salary // divisions

    prepare_categories(df, divisions, max_salary)

    print(df.head(300))

    print(f"Min = {min_salary}")
    print(f"Max = {max_salary}")
    print(f"Bin size = {BIN_SIZE}")


if __name__ == "__main__":
    main()
