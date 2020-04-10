import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMClassifier
from model_trainer import *
from preprocessing import *
import matplotlib.pyplot as plt
import seaborn as sns
import os
from lightgbm import plot_importance

if __name__ == "__main__":

    with open("best_lightgbm_new_vars_armando_params.pkl", "rb") as f:
        params = pickle.load(f)

    model = LGBMClassifier(
        class_weight="balanced",
        objective="multiclass:softmax",
        n_jobs=-1,
        random_state=100,
        silent=True,
        **params,
    )

    print("########## GETTING TRAIN AND TEST ###########")
    # print('########## GETTING TEST ###########')
    if "X_train_def.pkl" not in os.listdir():
        X_train, y_train, encoder = preprocess_data("TOTAL_TRAIN.csv", process_cat=True)
        with open("X_train_def.pkl", "wb") as f:
            pickle.dump(X_train, f)
        with open("y_train.pkl", "wb") as f:
            pickle.dump(y_train, f)
        with open("encoder_def.pkl", "wb") as f:
            pickle.dump(encoder, f)
    else:
        with open("X_train_def.pkl", "rb") as f:
            X_train = pickle.load(f)
        with open("y_train.pkl", "rb") as f:
            y_train = pickle.load(f)
        dtrain = pd.read_csv("TOTAL_TRAIN.csv")

        encoder = LabelEncoder()
        encoder.fit(dtrain.CLASE.values)

    X_test = preprocess_test("TOTAL_TEST.csv", process_cat=True)
    X_train = X_train.astype("float")
    X_test = X_test.astype("float")
    y_train = y_train.astype("int")

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
    print("########## FITTING MODEL ##############")
    model.fit(X_train, y_train)

    print("######### GETTING PREDICTIONS #########")
    preds = model.predict(X_test)

    print(
        "######### PINTANDO TEST EN PREDICCIONES_FINALES_DISTRIBUCION.png ###############"
    )
    preds = encoder.inverse_transform(preds)
    plt.rcParams["figure.figsize"] = (15, 15)
    fig = sns.countplot(preds, orient="vertical")
    fig.figure.savefig("PREDICCIONES_FINALES_DISTRIBUCION2.png")

    fig = plot_importance(model, ignore_zero=False, figsize=(30, 40))
    fig.figure.savefig("FEATURE_IMPORTANCE_FINAL.png")

    """
    test_df = pd.read_csv('TOTAL_TEST.csv')

    estimar_df = pd.DataFrame({'ID': test_df.ID, 'CLASE': preds})

    print('########### GUARDANDO ESTIMAR DF ###############')
    estimar_df.to_csv('AFI_SentinelAdmins.txt', header=True, index=False, sep='|', encoding='utf-8')
    """
