import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from missingpy import MissForest
import pickle
from tqdm import tqdm
import geopandas as gpd

imputer = MissForest(n_jobs=-1)
vars_postal_code = pd.read_csv("vars_censo_codigo_postal_def.csv")
cod_postales = gpd.read_file("codigos_postales_madrid/codigos_postales_madrid.shp")
categorical = [53, 54]

minmax = MinMaxScaler()
stdscaler = StandardScaler()
encoder = LabelEncoder()

############## TODO ###############
def fill_cods_nas(df):
    print('###### RELLENANDO VARS CODIGO POSTAL ###########')
    cols_rellenar = df.columns[df.isna().sum()>20]
    especiales = ['p_solteros','p_casados', 'p_viudos', 'p_separados', 'p_divorciados']
    #cols_rellenar1 = [col for col in cols_rellenar if col not in especiales]
    inds = (pd.isnull(df[cols_rellenar]).any(1)) & (df.CODIGO_POSTAL == 28907)
    df.loc[inds, cols_rellenar] = vars_postal_code.loc[vars_postal_code.CODIGO_POSTAL == 28903, cols_rellenar].values
    df.loc[df.CODIGO_POSTAL == 28524, especiales] = vars_postal_code.loc[vars_postal_code.CODIGO_POSTAL == 28522, especiales].values
    return df


def nearest_postal_code(pcode):
    pass


###################################


def process_categorical(df, cat_columns):
    for col in tqdm(cat_columns):
        print(f"procesada columna {col}")
        set_ = [l for l in set(df[col])]
        dic_ = {l: i for i, l in enumerate(set_)}
        for i in range(df.shape[0]):
            df[col].iloc[i] = dic_[df[col].iloc[i]]
        df[col] = df[col].astype("category")
    return df


"""
merged_df2["population_density"] = (
        merged_df2["poblacion"] / merged_df2["area_cod_postal"]
    )
"""


def get_points_density(df, around=5, pobs=None):
    """
    Function to get the points density some kilometers around.

    Parameters
    -------------
    df: pandas.DataFrame
        df containing the points and the poblacion per postal code variables.
    around: int or float
        the number of km to define the space around the point for density estimation
    pobs: array or None
        if None, the population is taken from the 
    Returns
    -------------
    density for each of the points in the 
    
    """
    grad_to_lat = 1 / 111
    grad_to_lon = 1 / 85
    densities = np.empty((df.shape[0],), dtype="float64")
    for i in tqdm(range(df.shape[0]), desc="GETTING POINTS DENSITY"):
        lon, lat = df["lon"].iloc[i], df["lat"].iloc[i]
        min_lon = lon - around * grad_to_lon
        max_lon = lon + around * grad_to_lon
        min_lat = lat - around * grad_to_lat
        max_lat = lat + around * grad_to_lat
        mybool = (
            (df["lat"] >= min_lat)
            & (df["lat"] <= max_lat)
            & (df["lon"] >= min_lon)
            & (df["lon"] <= max_lon)
        )
        points_around = df.loc[mybool, :].shape[0]
        population_postal_code = df.poblacion.iloc[i]
        if pobs is None:
            densities[i] = points_around / df.shape[0]  # / population_postal_code
        else:
            densities[i] = points_around / df.shape[0]  # / pobs[i]
        # TODO : PROBAR SOLO CON LOS PUNTOS AL REDEDOR O ESCALÁNDOLO DE OTRA MANERA.
    return densities


def process_cadqual(df):
    dic_ = {
        "A": 12,
        "B": 11,
        "C": 10,
        "1": 9,
        "2": 8,
        "3": 7,
        "4": 6,
        "5": 4,
        "6": 3,
        "7": 2,
        "8": 1,
        "9": 0,
        "1.0": 9,
        "2.0": 8,
        "3.0": 7,
        "4.0": 6,
        "5.0": 4,
        "6.0": 3,
        "7.0": 2,
        "8.0": 1,
        "9.0": 0,
        "nan": np.nan,
    }
    for i in tqdm(range(df.shape[0])):
        df["CADASTRALQUALITYID"].iloc[i] = dic_[df["CADASTRALQUALITYID"].iloc[i]]
    df.CADASTRALQUALITYID = df.CADASTRALQUALITYID.astype("float")
    return df


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def three_dim_space(lat, lon):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def get_mean_color(df):
    colores = ["R", "G", "B", "NIR"]
    for color in tqdm(colores):
        cols_color = [col for col in df.columns if f"Q_{color}_" in col]
        df[f"media_{color}"] = df[cols_color].mean(axis=1)
    return df


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


def preprocess_data(
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
    # lon_, lat_ = df['lon'], df['lat']
    X = df.drop(["CLASE", "ID", "cluster", "lon", "lat"], axis=1)
    print(f"Valores unicos de CADASTRAL--- {X.CADASTRALQUALITYID.unique()}")
    cols_conflictivas = [
        "edad_media",
        "p_poblacion_menor_de_18",
        "p_poblacion_mayor_65",
        "media_personas_por_hogar",
        "p_hogares_unipersonales",
        "poblacion_cp",
        "poblacion_municipio"
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
    X.CALIDAD_AIRE = X.CALIDAD_AIRE.astype("str")
    
    ########### color variables ##################
    print("##### GETTING COLOR VARIABLES ##########")
    X = get_mean_color(X)

    ########### suma geoms #######################
    X["suma_geoms"] = X[[col for col in X.columns if "GEOM_" in col]].sum(axis=1)

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
    
    if process_cat:
        X = pd.get_dummies(X, columns=X.columns[X.dtypes == object])
    # else:
    #    cat_features = X.columns[X.dtypes == object]
    #    not_select = X.columns[X.dtypes == object]

    print(f"En momento 3 el shape es de {X.shape}")
    print("Imputando valores con Random Forest")
    cols = X.columns
    X[X == np.inf] = np.nan
    X = fill_cods_nas(X)
    X['renta_media_por_hogar'] = [
        X.loc[i, 'renta_media_por_hogar'].replace(',', '') for i in range(X.shape[0])
    ]
    X['renta_media_por_hogar'] = X['renta_media_por_hogar'].astype('float64')
    print(f"Las columnas que tienen dtype object son {X.columns[X.dtypes == object]}")
    for col in X.columns[X.dtypes == object]:
        if sum(X[col].isna()) != 0:
            X.loc[X[col].isna(), col] = f"{col}_Ausente"
    imputer.fit(X.loc[:, X.columns[X.dtypes != object]])
    X.loc[:, X.columns[X.dtypes != object]] = imputer.transform(
        X.loc[:, X.columns[X.dtypes != object]]
    )
    with open("imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    print(f"En momento 4 el shape es de {X.shape}")
    X = pd.DataFrame(X, columns=cols)
    # X["point_density"] = get_points_density(df=df, around=5, pobs=X.poblacion.values)
    X["population_density"] = X["poblacion"] / X["area_cod_postal"]
    X.MAXBUILDINGFLOOR.clip(0.0, 25.0, inplace=True)
    X.CADASTRALQUALITYID.clip(0.0, 12.0, inplace=True)
    ########## HERE I TREAT LAT AND LON ########################
    geo_x, geo_y, geo_z = three_dim_space(X.X, X.Y)
    X["GEO_X"] = geo_x
    X["GEO_Y"] = geo_y
    X["GEO_Z"] = geo_z
    points = [(x, y) for x, y in zip(X.X, X.Y)]
    origin = (X.X.mean(), X.Y.mean())
    rotated90 = rotate(points, origin, degrees=90)
    rotated180 = rotate(points, origin, degrees=180)
    x_rot_90 = [r[0] for r in rotated90]
    y_rot_90 = [r[1] for r in rotated90]
    x_rot_180 = [r[0] for r in rotated180]
    y_rot_180 = [r[1] for r in rotated180]
    X["x_rot_90"] = x_rot_90
    X["y_rot_90"] = y_rot_90
    X["x_rot_180"] = x_rot_180
    X["y_rot_180"] = y_rot_180
    X["lat_2"] = X.X ** 2
    X["lon_2"] = X.Y ** 2
    X["latxlon"] = X.X * X.Y
    print(f"En momento 5 el shape es de {X.shape}")
    select_columns = (
        X.dtypes != object
    )  # [i for i, col in enumerate(X.columns) if col not in not_select]
    """
    if not process_cat:
        categoricas = []
        for i in range(X.shape[1]):
            if X.dtypes[i] == object:
                categoricas.append(i)
        X = process_categorical(X, X.columns[X.dtypes == object])
    """
    colnames = X.columns
    X = np.array(X)
    if scale:
        if scaler == "std":
            X[:, select_columns] = stdscaler.fit_transform(X[:, select_columns])
            with open("SCALER.pkl", "wb") as f:
                pickle.dump(stdscaler, f)
        elif scaler == "minmax":
            X[:, select_columns] = minmax.fit_transform(X[:, select_columns])
        print(f"En momento 6 el shape es de {X.shape}")
    if not process_cat:
        return pd.DataFrame(X, columns=colnames), y, encoder
    else:
        return pd.DataFrame(X, columns=colnames), y, encoder


def preprocess_test(f, scale=True, scaler="std", process_cat=True, sample_trials=None):
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
    sample_trials
        number of samples to take, if None, full data is returned.
    
    Returns
    -------------
    X
        The matrix with features
    """
    df = pd.read_csv(f)
    if sample_trials is not None:
        df = df.sample(sample_trials)

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
    # if process_cat:
    X = pd.get_dummies(X, columns=X.columns[X.dtypes == object])
    print(f"En momento 3 el shape es de {X.shape}")
    print("Imputando valores con Random Forest")
    cols = list(X.columns)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    try:
        print("PRIMERA")
        with open("imputer.pkl", "rb") as f:
            imputer = pickle.load(f)
        X = imputer.transform(X)
    except Exception as e:
        print(e)
        try:
            print("SEGUNDA")
            Xtr = partial_preprocess_train("TOTAL_TRAIN.csv", process_cat=True)
            cols_ausentes_test = [col for col in Xtr.columns if col not in X.columns]
            if len(cols_ausentes_test) != 0:
                for col in cols_ausentes_test:
                    print(f"la columna {col} está ausente en test!")
                    X[col] = [0] * X.shape[0]
            imputer = MissForest()
            imputer.fit(Xtr)
            X = imputer.transform(X)
        except Exception as e:
            print(e)
            X = partial_preprocess_train2("TOTAL_TEST.csv", process_cat=True)
            imputer = MissForest()
            X = imputer.fit_transform(X)

    print(f"En momento 4 el shape es de {X.shape}")
    cols = [c for c in cols] + [c for c in cols_ausentes_test]
    X = pd.DataFrame(X, columns=cols)
    X.MAXBUILDINGFLOOR.clip(0.0, 25.0, inplace=True)
    X.CADASTRALQUALITYID.clip(0.0, 12.0, inplace=True)
    ########## HERE I TREAT LAT AND LON ########################
    geo_x, geo_y, geo_z = three_dim_space(X.X, X.Y)
    X["GEO_X"] = geo_x
    X["GEO_Y"] = geo_y
    X["GEO_Z"] = geo_z
    points = [(x, y) for x, y in zip(X.X, X.Y)]
    origin = (X.X.mean(), X.Y.mean())
    rotated90 = rotate(points, origin, degrees=90)
    rotated180 = rotate(points, origin, degrees=180)
    x_rot_90 = [r[0] for r in rotated90]
    y_rot_90 = [r[1] for r in rotated90]
    x_rot_180 = [r[0] for r in rotated180]
    y_rot_180 = [r[1] for r in rotated180]
    X["x_rot_90"] = x_rot_90
    X["y_rot_90"] = y_rot_90
    X["x_rot_180"] = x_rot_180
    X["y_rot_180"] = y_rot_180
    X["lat_2"] = X.X ** 2
    X["lon_2"] = X.Y ** 2
    X["latxlon"] = X.X * X.Y
    print(f"En momento 5 el shape es de {X.shape}")
    select_columns = [i for i, col in enumerate(X.columns) if len(X[col].unique()) > 2]
    if not process_cat:
        categoricas = []
        for i in range(X.shape[1]):
            if X.dtypes[i] == object:
                categoricas.append(i)
        X = process_categorical(X, X.columns[X.dtypes == object])
    colnames = X.columns
    X = np.array(X)
    if scale:
        if scaler == "std":
            with open("SCALER.pkl", "rb") as f:
                stdscaler = pickle.load(f)
            try:
                X[:, select_columns] = stdscaler.transform(X[:, select_columns])
            except:
                Xtr = partial_preprocess_train("TOTAL_TRAIN.csv", process_cat=True)
                cols_ausentes_test = [
                    col for col in Xtr.columns if col not in X.columns
                ]
                if len(cols_ausentes_test) != 0:
                    for col in cols_ausentes_test:
                        print(f"la columna {col} está ausente en test!")
                        X[col] = [0] * X.shape[0]
                select_columns = [col for col in X.columns if len(X[col].unique()) > 2]
                X[:, select_columns] = stdscaler.transform(X[:, select_columns])
        elif scaler == "minmax":
            X[:, select_columns] = minmax.fit_transform(X[:, select_columns])
        print(f"En momento 6 el shape es de {X.shape}")
    if not process_cat:
        return pd.DataFrame(X, columns=colnames), categoricas
    else:
        return pd.DataFrame(X, columns=colnames)
