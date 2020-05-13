##########################################
### AUTHOR: ALEJANDRO VACA SERRANO #######
##########################################


import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from xgboost import XGBClassifier
from model_trainer import *
from preprocessing import *
import matplotlib.pyplot as plt
import seaborn as sns
import os
from lightgbm import plot_importance
import argparse
from models import *


direc = "stacking_models"
save_dir = "ENTREGA_NACIONAL"


if save_dir not in os.listdir():
    os.mkdir(save_dir)


def get_categorical_encoding(X_train, X_test, y_train):
    cat_encoder = CatBoostEncoder()
    X_train = cat_encoder.fit_transform(X_train, y_train)
    X_test = cat_encoder.transform(X_test)
    return X_train, X_test, y_train


def fix_train_test(X_train, X_test):
    dtrain = pd.read_csv("TOTAL_TRAIN.csv")
    X_train["lon"] = dtrain["lon"]
    X_train["lat"] = dtrain["lat"]
    cols_cat = ["ruido", "CODIGO_POSTAL", "ZONA_METROPOLITANA", "CALIDAD_AIRE"]
    cols_float = [
        col for col in X_train.columns if col not in cols_cat and col in X_test.columns
    ]
    X_train[cols_float] = X_train[cols_float].astype("float")
    cols_to_drop = [col for col in X_train.columns if col not in X_test.columns]
    X_train = X_train.drop(cols_to_drop, axis=1)
    X_test[cols_float] = X_test[cols_float].astype("float")
    if "lon" in X_test.columns and "lat" in X_test.columns:
        X_test = X_test.drop(["lon", "lat"], axis=1)
    good_colnames = []
    i = 0
    for col in X_train.columns:
        if not col.isascii():
            print(f"La columna {col} no es ascii")
            good_colnames.append(f"{i}_not_ascii")
            i += 1
        else:
            good_colnames.append(col)
    X_train.columns = good_colnames
    good_colnames = []
    i = 0
    for col in X_test.columns:
        if not col.isascii():
            print(f"La columna {col} no es ascii")
            good_colnames.append(f"{i}_not_ascii")
            i += 1
        else:
            good_colnames.append(col)
    X_test.columns = good_colnames
    if "Oeste" in X_train.columns:
        X_train = X_train.drop("Oeste", axis=1)
    if "Oeste" in X_test.columns:
        X_test = X_test.drop("Oeste", axis=1)
    return X_train, X_test


def get_matrices(args):
    if not args.use_old:
        X_train, y_train, encoder = preprocess_data(
            "TOTAL_TRAIN.csv", process_cat=False
        )
        with open("X_train_def.pkl", "wb") as f:
            pickle.dump(X_train, f)
        with open("y_train_def.pkl", "wb") as f:
            pickle.dump(y_train, f)
        with open("encoder_def.pkl", "wb") as f:
            pickle.dump(encoder, f)
        X_test = preprocess_test("TOTAL_TEST.csv", process_cat=False, X_train=X_train)
        with open("X_test_def.pkl", "wb") as f:
            pickle.dump(X_test, f)
    else:
        with open("X_train_def.pkl", "rb") as f:
            X_train = pickle.load(f)
        with open("y_train_def.pkl", "rb") as f:
            y_train = pickle.load(f)
        with open("encoder_def.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open("X_test_def.pkl", "rb") as f:
            X_test = pickle.load(f)
    return X_train, X_test, y_train, encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        required=True,
        help="The name to put to the model saved and the mlflow run",
        type=str,
    )
    parser.add_argument(
        "-uo",
        "--use_old",
        dest="use_old",
        required=False,
        default=False,
        help="whether or not to use the old matrices",
        type=bool,
    )

    args = parser.parse_args()

    print("LOADING MODEL")

    model = FINAL_MODELS[args.name]["model"]
    print(model)

    print("########## GETTING TRAIN AND TEST ###########")
    X_train, X_test, y_train, encoder = get_matrices(args)

    print("###### FIXING MATRICES ########")
    X_train, X_test = fix_train_test(X_train, X_test)
    X_train, X_test, y_train = get_categorical_encoding(X_train, X_test, y_train)
    print("########## FITTING MODEL ##############")
    model.fit(X_train, y_train)
    print("######### GETTING PREDICTIONS #########")
    preds = model.predict(X_test)
    with open(f"./{save_dir}/predicciones_{args.name}.pkl", "wb") as f:
        pickle.dump(preds, f)
    print(
        "######### PINTANDO TEST EN PREDICCIONES_FINALES_DISTRIBUCION.png ###############"
    )
    test_df = pd.read_csv("TOTAL_TEST.csv")
    preds = encoder.inverse_transform(preds)
    estimar_df = pd.DataFrame({"ID": test_df.ID, "CLASE": preds})
    print("########### GUARDANDO ESTIMAR DF ###############")
    estimar_df.to_csv(
        f"./{save_dir}/AFI_SentinelAdmins_{args.name}.txt",
        header=True,
        index=False,
        sep="|",
        encoding="utf-8",
    )
    plt.rcParams["figure.figsize"] = (15, 15)
    fig = sns.countplot(estimar_df.CLASE, orient="vertical")
    fig.figure.savefig(
        f"./{save_dir}/PREDICCIONES_FINALES_DISTRIBUCION_{args.name}.png"
    )
    print(
        f"La proporción de cada clase en test es de : {estimar_df.groupby('CLASE').count() / estimar_df.shape[0]}"
    )
