import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from missingpy import MissForest
import pickle
from tqdm import tqdm
import geopandas as gpd
import multiprocessing as mp
from functools import partial
from sklearn.decomposition import PCA 
import os


def partial_preprocess_train(
    f, scale=True, scaler="std", process_cat=True, y_name="CLASE", sample_trials=None
):
    """
    takes a file name and returns the processed dataset
    
    Parameters
    -------------
    f
        the filename
    scale
        whether to scale the numerical variables
    scaler
        which scaler to use for numerical variables
    process_cat
        whether to do one-hot encoding for categorical, as some models like catboost don't want them one-hot.
    y_name
        name of the variable where the objective is.
    sample_trials
        number of samples to take, if None, full data is returned.
    
    Returns
    -------------
    X
        The matrix with features
    y
        The vector with objective variable
    """
    df = pd.read_csv(f)
    if sample_trials is not None:
        df = df.sample(sample_trials)
    encoder.fit(df.CLASE.values)
    y = encoder.transform(df.CLASE.values)
    X = df.drop(["CLASE", "ID", "lat", "lon", "cluster"], axis=1)
    print(f"Valores unicos de CADASTRAL--- {X.CADASTRALQUALITYID.unique()}")
    cols_conflictivas = [
        "edad_media",
        "p_poblacion_menor_de_18",
        "p_poblacion_mayor_65",
        "media_personas_por_hogar",
        "p_hogares_unipersonales",
        "poblacion",
    ]
    for col in tqdm(cols_conflictivas):
        for i in range(X.shape[0]):
            try:
                float(X[col].iloc[i])
            except:
                X[col].iloc[i] = np.nan
    X[cols_conflictivas] = X[cols_conflictivas].astype("float")
    X.CADASTRALQUALITYID = X.CADASTRALQUALITYID.astype("str")
    X.CODIGO_POSTAL = X.CODIGO_POSTAL.astype("str")
    X.ruido = X.ruido.astype("str")

    ########## COLOR VARIABLES ###################

    ########### HERE WE DEAL WITH GEOM VARS AND CREATE NEW GEOM VARS ############
    cols_geoms = [col for col in X.columns if "GEOM" in col]
    for col in tqdm(cols_geoms):
        otras = [c for c in cols_geoms if c != col]
        for otracol in otras:
            if (
                f"{col}_x_{otracol}" not in X.columns
                and f"{otracol}_x_{col}" not in X.columns
            ):
                X[f"{col}_x_{otracol}"] = X[col] * X[otracol]
        X[f"{col}_x_area"] = X[col] * X["AREA"]
        X[f"{col}_2"] = X[col] ** 2
    X = process_cadqual(X)
    print(f"En momento 2 el shape es de {X.shape}")
    print(f"Las columnas que tienen dtype object son {X.columns[X.dtypes == object]}")
    for col in X.columns[X.dtypes == object]:
        if sum(X[col].isna()):
            X.loc[X[col].isna(), col] = f"{col}_Ausente"
    if process_cat:
        X = pd.get_dummies(X, columns=X.columns[X.dtypes == object])
    return X


def partial_preprocess_train2(
    f, scale=True, scaler="std", process_cat=True, y_name="CLASE", sample_trials=None
):
    """
    takes a file name and returns the processed dataset
    
    Parameters
    -------------
    f
        the filename
    scale
        whether to scale the numerical variables
    scaler
        which scaler to use for numerical variables
    process_cat
        whether to do one-hot encoding for categorical, as some models like catboost don't want them one-hot.
    y_name
        name of the variable where the objective is.
    sample_trials
        number of samples to take, if None, full data is returned.
    
    Returns
    -------------
    X
        The matrix with features
    y
        The vector with objective variable
    """
    df = pd.read_csv(f)
    if sample_trials is not None:
        df = df.sample(sample_trials)
    # encoder.fit(df.CLASE.values)
    # y = encoder.transform(df.CLASE.values)
    X = df.drop(["ID", "lat", "lon", "cluster"], axis=1)
    print(f"Valores unicos de CADASTRAL--- {X.CADASTRALQUALITYID.unique()}")
    cols_conflictivas = [
        "edad_media",
        "p_poblacion_menor_de_18",
        "p_poblacion_mayor_65",
        "media_personas_por_hogar",
        "p_hogares_unipersonales",
        "poblacion",
    ]
    for col in tqdm(cols_conflictivas):
        for i in range(X.shape[0]):
            try:
                float(X[col].iloc[i])
            except:
                X[col].iloc[i] = np.nan
    X[cols_conflictivas] = X[cols_conflictivas].astype("float")
    X.CADASTRALQUALITYID = X.CADASTRALQUALITYID.astype("str")
    X.CODIGO_POSTAL = X.CODIGO_POSTAL.astype("str")
    X.ruido = X.ruido.astype("str")
    ########### HERE WE DEAL WITH GEOM VARS AND CREATE NEW GEOM VARS ############
    cols_geoms = [col for col in X.columns if "GEOM" in col]
    for col in tqdm(cols_geoms):
        otras = [c for c in cols_geoms if c != col]
        for otracol in otras:
            if (
                f"{col}_x_{otracol}" not in X.columns
                and f"{otracol}_x_{col}" not in X.columns
            ):
                X[f"{col}_x_{otracol}"] = X[col] * X[otracol]
        X[f"{col}_x_area"] = X[col] * X["AREA"]
        X[f"{col}_2"] = X[col] ** 2
    X = process_cadqual(X)
    print(f"En momento 2 el shape es de {X.shape}")
    print(f"Las columnas que tienen dtype object son {X.columns[X.dtypes == object]}")
    for col in X.columns[X.dtypes == object]:
        if sum(X[col].isna()):
            X.loc[X[col].isna(), col] = f"{col}_Ausente"
    if process_cat:
        X = pd.get_dummies(X, columns=X.columns[X.dtypes == object])
    return X
