##########################################
### AUTHOR: ALEJANDRO VACA SERRANO #######
##########################################


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# from missingpy import MissForest
import pickle
from tqdm import tqdm
import geopandas as gpd
import multiprocessing as mp
from functools import partial
from sklearn.decomposition import PCA
import os
from sklearn.impute import SimpleImputer

pd.options.display.max_rows = 2000
SAVE_DIR = "objects_fitted"
if SAVE_DIR not in os.listdir():
    os.mkdir(SAVE_DIR)
# imputer = MissForest(n_jobs=-1)
vars_postal_code = pd.read_csv("vars_censo_codigo_postal_def.csv")
cod_postales = gpd.read_file("codigos_postales_madrid/codigos_postales_madrid.shp")
categorical = [53, 54]
dataset_train = pd.read_csv("dataset_train.csv")
minmax = MinMaxScaler()
stdscaler = StandardScaler()
encoder = LabelEncoder()


def get_neighbors_means(points, var, X, around=7):
    print(f"{var} NEIGHBORS")
    grad_to_lat = 1 / 111
    grad_to_lon = 1 / 85
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    min_lons = [lon - around * grad_to_lon for lon in lons]
    max_lons = [lon + around * grad_to_lon for lon in lons]
    min_lats = [lat - around * grad_to_lat for lat in lats]
    max_lats = [lat + around * grad_to_lat for lat in lats]
    means_ = np.empty((len(points),), dtype="float64")
    i = 0
    for min_lon, max_lon, min_lat, max_lat in zip(
        min_lons, max_lons, min_lats, max_lats
    ):
        mybool = (
            (X["lat"] >= min_lat)
            & (X["lat"] <= max_lat)
            & (X["lon"] >= min_lon)
            & (X["lon"] <= max_lon)
        )
        try:
            means_[i] = float(X.loc[mybool, var].median())
        except:
            means_[i] = 0.0
        i += 1
    return means_


def fill_cods_nas(df):
    print("###### RELLENANDO VARS CODIGO POSTAL ###########")
    cols_rellenar = df.columns[df.isna().sum() > 20]
    cols_rellenar = [col for col in cols_rellenar if col in vars_postal_code.columns]
    especiales = ["p_solteros", "p_casados", "p_viudos", "p_separados", "p_divorciados"]
    # cols_rellenar1 = [col for col in cols_rellenar if col not in especiales]
    inds = (pd.isnull(df[cols_rellenar]).any(1)) & (df.CODIGO_POSTAL == 28907)
    df.loc[inds, cols_rellenar] = vars_postal_code.loc[
        vars_postal_code.CODIGO_POSTAL == 28903, cols_rellenar
    ].values
    df.loc[df.CODIGO_POSTAL == 28524, especiales] = vars_postal_code.loc[
        vars_postal_code.CODIGO_POSTAL == 28522, especiales
    ].values
    return df


def process_categorical(df, cat_columns):
    for col in tqdm(cat_columns):
        print(f"procesada columna {col}")
        set_ = [l for l in set(df[col])]
        dic_ = {l: i for i, l in enumerate(set_)}
        for i in range(df.shape[0]):
            df[col].iloc[i] = dic_[df[col].iloc[i]]
        df[col] = df[col].astype("category")
    return df


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

def geospatial_vars(X):
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
    return X


def get_mean_color(df):
    colores = ["R", "G", "B", "NIR"]
    numeros = [4, 3, 2, 8]
    for color, numero in tqdm(zip(colores, numeros)):
        cols_color = [col for col in df.columns if f"Q_{color}_" in col]
        df[f"media_{color}"] = df[cols_color].mean(axis=1)
        df[f"std_{color}"] = (
            df[f"Q_{color}_{numero}_0_9"] - df[f"Q_{color}_{numero}_0_1"]
        )  
    return df


def get_pca_colors(X, cols, test=False):
    if not test:
        pca = PCA(n_components=3, random_state=42)
        components = pca.fit_transform(X[cols].values)
        save_obj(pca, "pca_colors")
    else:
        pca = load_obj("pca_colors")
        components = pca.transform(X[cols].values)
    return components


def get_pca_geoms(X):
    cols = [col for col in X.columns if "GEOM_" in col]
    pca = PCA(n_components=3, random_state=42)
    components = pca.fit_transform(X[cols].values)
    with open("pca_geoms.pkl", "wb") as f:
        pickle.dump(pca, f)
    return components


def load_obj(name):
    with open(f"./{SAVE_DIR}/{name}.pkl", "rb") as f:
        o = pickle.load(f)
    return o


def save_obj(obj, name):
    with open(f"./{SAVE_DIR}/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def get_vegetation_indices_deciles(df):
    sc = MinMaxScaler()
    qs = [f"_0_{i}" for i in range(10)] + ["_1_0"]
    qs = [q for q in qs if q != "_0_5"]
    cols_imputar = []
    for q in qs:
        R = sc.fit_transform(df[f"Q_R_4{q}"].values.reshape(-1, 1))
        save_obj(sc, f"scaler_Q_R_4{q}")
        G = sc.fit_transform(df[f"Q_G_3{q}"].values.reshape(-1, 1))
        save_obj(sc, f"scaler_Q_G_3{q}")
        B = sc.fit_transform(df[f"Q_B_2{q}"].values.reshape(-1, 1))
        save_obj(sc, f"scaler_Q_B_2{q}")
        N = sc.fit_transform(df[f"Q_NIR_8{q}"].values.reshape(-1, 1))
        save_obj(sc, f"scaler_Q_NIR_8{q}")
        df[f"GRVI{q}"] = (G - R) / (G + R)
        df[f"MGRVI{q}"] = (G ** 2 - R ** 2) / (G ** 2 + R ** 2)
        df[f"RGBVI{q}"] = (G ** 2 - (R * B)) / (G ** 2 + (R * B))
        ExG = 2 * G - R - B
        df[f"ExG{q}"] = ExG
        df[f"ExGR{q}"] = ExG - 1.4 * R - G
        df[f"NDVI{q}"] = (N - R) / (N + R)
        df[f"SAVI{q}"] = ((N - R) * (1 + 0.5)) / (N + R + 0.5)
        df[f"MSAVI{q}"] = ((2 * N + 1) - np.sqrt(((2 * N + 1) ** 2) - 8 * (N - R))) / 2
        df[f"GNDVI{q}"] = (N - G) / (N + G)
        df[f"OSAVI{q}"] = ((N - R) * (1 + 0.16)) / (N + R + 0.16)
        df[f"GCI{q}"] = (N / (G + 0.001)) - 1
        df[f"NDWI{q}"] = (G - N) / (G + N)
        df[f"BAI{q}"] = 1 / ((0.01 - R) ** 2 + (0.06 - N) ** 2)
        df[f"NPCRI{q}"] = (R - B) / (R + B)
        df[f"AVI{q}"] = (N * (1 - R) * (N - R)) ** (1 / 3)
        df[f"SI{q}"] = ((1 - R) * (1 - G) * (1 - B)) ** (1 / 3)
        df[f"RVI{q}"] = N / R
        df[f"SR{q}"] = R / N
        df[f"TVI{q}"] = 0.5 * (120 * (N - G)) - 200 * (R - G)
        df[f"WII{q}"] = 0.91 * R + 0.43 * N
        df[f"DVW{q}"] = df.NDVI.values - df.NDWI.values
        df[f"IFW{q}"] = N - G
        df[f"IPVI{q}"] = N / (N + R)
        df.replace(np.inf, 0, inplace=True)
        df[
            [
                f"GRVI{q}",
                f"MGRVI{q}",
                f"RGBVI{q}",
                f"ExG{q}",
                f"ExGR{q}",
                f"NDVI{q}",
                f"SAVI{q}",
                f"MSAVI{q}",
                f"GNDVI{q}",
                f"OSAVI{q}",
                f"GCI{q}",
                f"NDWI{q}",
                f"BAI{q}",
                f"NPCRI{q}",
                f"AVI{q}",
                f"SI{q}",
                f"RVI{q}",
                f"SR{q}",
                f"TVI{q}",
                f"WII{q}",
                f"DVW{q}",
                f"IFW{q}",
                f"IPVI{q}",
            ]
        ] = SimpleImputer(
            missing_values=np.nan, strategy="mean", verbose=0
        ).fit_transform(
            df[
                [
                    f"GRVI{q}",
                    f"MGRVI{q}",
                    f"RGBVI{q}",
                    f"ExG{q}",
                    f"ExGR{q}",
                    f"NDVI{q}",
                    f"SAVI{q}",
                    f"MSAVI{q}",
                    f"GNDVI{q}",
                    f"OSAVI{q}",
                    f"GCI{q}",
                    f"NDWI{q}",
                    f"BAI{q}",
                    f"NPCRI{q}",
                    f"AVI{q}",
                    f"SI{q}",
                    f"RVI{q}",
                    f"SR{q}",
                    f"TVI{q}",
                    f"WII{q}",
                    f"DVW{q}",
                    f"IFW{q}",
                    f"IPVI{q}",
                ]
            ].values
        )
    return df


def get_media_veg_ind(df, cols_veget):
    for col in cols_veget:
        cols = [c for c in df.columns if col in c]
        df[f"media_{col}"] = df[cols].mean(axis=1)
        df = df.drop(cols, axis=1)
    return df


def get_vegetation_indices(df, test=False):
    sc = MinMaxScaler()
    if not test:
        R = sc.fit_transform(df["media_R"].values.reshape(-1, 1))  # Q_R_4_0_5
        save_obj(sc, "scaler_media_R")
        G = sc.fit_transform(df["media_G"].values.reshape(-1, 1))  # Q_G_3_0_5
        save_obj(sc, "scaler_media_G")
        B = sc.fit_transform(df["media_B"].values.reshape(-1, 1))  # Q_B_2_0_5
        save_obj(sc, "scaler_media_B")
        N = sc.fit_transform(df["media_NIR"].values.reshape(-1, 1))  # Q_NIR_8_0_5
        save_obj(sc, "scaler_media_NIR")
    else:
        sc_R = load_obj("scaler_media_R")
        R = sc_R.transform(df["media_R"].values.reshape(-1, 1))
        sc_G = load_obj("scaler_media_G")
        G = sc_G.transform(df["media_G"].values.reshape(-1, 1))
        sc_B = load_obj("scaler_media_B")
        B = sc_B.transform(df["media_B"].values.reshape(-1, 1))
        sc_N = load_obj("scaler_media_NIR")
        N = sc_N.fit_transform(df["media_NIR"].values.reshape(-1, 1))
    df["GRVI"] = (G - R) / (G + R)
    # df["MGRVI"] = (G ** 2 - R ** 2) / (G ** 2 + R ** 2)
    df["RGBVI"] = (G ** 2 - (R * B)) / (G ** 2 + (R * B))
    ExG = 2 * G - R - B
    df["ExG"] = ExG
    df["ExGR"] = ExG - 1.4 * R - G
    df["NDVI"] = (N - R) / (N + R)
    df["SAVI"] = ((N - R) * (1 + 0.5)) / (N + R + 0.5)
    df["MSAVI"] = ((2 * N + 1) - np.sqrt(((2 * N + 1) ** 2) - 8 * (N - R))) / 2
    df["GNDVI"] = (N - G) / (N + G)
    df["OSAVI"] = ((N - R) * (1 + 0.16)) / (N + R + 0.16)
    df["GCI"] = (N / (G + 0.001)) - 1
    df["NDWI"] = (G - N) / (G + N)
    df["BAI"] = 1 / ((0.01 - R) ** 2 + (0.06 - N) ** 2)
    df["NPCRI"] = (R - B) / (R + B)
    df["AVI"] = (N * (1 - R) * (N - R)) ** (1 / 3)
    df["SI"] = ((1 - R) * (1 - G) * (1 - B)) ** (1 / 3)
    df["RVI"] = N / R
    df["SR"] = R / N
    df["TVI"] = 0.5 * (120 * (N - G)) - 200 * (R - G)
    df["WII"] = 0.91 * R + 0.43 * N
    df["DVW"] = df.NDVI.values - df.NDWI.values
    df["IFW"] = N - G
    df["IPVI"] = N / (N + R)
    # df[['NPCRI', 'AVI']] = df[['NPCRI', 'AVI']].fillna(value=0, inplace=True)
    df[["NPCRI", "AVI"]] = SimpleImputer(
        missing_values=np.nan, strategy="mean", verbose=0
    ).fit_transform(df[["NPCRI", "AVI"]])
    return df


matriz_ = np.matrix("0.299 0.587 0.114; -0.147 -0.289 0.436; 0.615 -0.515 -0.100")


def get_yuv(df, matriz_=matriz_, test=False, X_train=None):
    # qs = [f"_0_{i}" for i in range(10)] + ["_1_0"]
    # colors = ["Q_R_4", "Q_G_3", "Q_B_2"]
    colors = ["media_R", "media_G", "media_B"]
    if not test:
        sc = MinMaxScaler()
        mat_col = sc.fit_transform(df[colors].values).T
        save_obj(sc, "scaler_yuv")
    else:
        sc = MinMaxScaler()
        mat_col = sc.fit_transform(df[colors].values.T)
    print("multiplicando")
    yuv = np.dot(matriz_, mat_col)
    df["Y_YUV"] = yuv[0, :].T
    df["U_YUV"] = yuv[1, :].T
    df["V_YUV"] = yuv[2, :].T
    """
    for q in qs:
        cols_select = [f"{col}{q}" for col in colors]
        mat_col = sc.fit_transform(df[cols_select].values.T)
        save_obj(sc, f"scaler_yuv_{q}")
        yuv = np.dot(matriz_, mat_col)
        print(yuv.shape)
        df[f"Y_YUV_{q[1:]}"] = yuv[0, :].T
        df[f"U_YUV_{q[1:]}"] = yuv[1, :].T
        df[f"V_YUV_{q[1:]}"] = yuv[2, :].T
    """
    return df


def solve_cols_conflictivas(X):
    cols_conflictivas = [
        "edad_media",
        "p_poblacion_menor_de_18",
        "p_poblacion_mayor_65",
        "media_personas_por_hogar",
        "p_hogares_unipersonales",
        "poblacion_cp",
        "poblacion_municipio",
    ]
    for col in tqdm(cols_conflictivas):
        for i in range(X.shape[0]):
            try:
                float(X[col].iloc[i])
            except:
                X[col].iloc[i] = np.nan
    X[cols_conflictivas] = X[cols_conflictivas].astype("float")
    return X


def create_geovars(X):
    cols_geoms = ["GEOM_R1", "GEOM_R2", "GEOM_R3", "GEOM_R4"]
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
    return X


def fix_renta_media_por_hogar(X):
    X["renta_media_por_hogar"] = [
        X.loc[i, "renta_media_por_hogar"].replace(",", "")
        if type(X.loc[i, "renta_media_por_hogar"]) != float
        else X.loc[i, "renta_media_por_hogar"]
        for i in range(X.shape[0])
    ]
    X["renta_media_por_hogar"] = X["renta_media_por_hogar"].astype("float64")
    return X


def fix_nas_categorical(X):
    for col in X.columns[X.dtypes == object]:
        if sum(X[col].isna()) != 0:
            X.loc[X[col].isna(), col] = f"{col}_Ausente"
    return X


def transform_types_str(X):
    X.CADASTRALQUALITYID = X.CADASTRALQUALITYID.astype("str")
    X.CODIGO_POSTAL = X.CODIGO_POSTAL.astype("str")
    X.ruido = X.ruido.astype("str")
    X.CALIDAD_AIRE = X.CALIDAD_AIRE.astype("str")
    return X


def preprocess_data(
    f,
    scale=True,
    scaler="std",
    process_cat=True,
    y_name="CLASE",
    sample_trials=None,
    impute_data=True,
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
    X = df.drop(["CLASE", "ID", "cluster", "Oeste"], axis=1)
    if 'Unnamed: 0' in X.columns:
        X = X.drop('Unnamed: 0', axis=1)
    print(f"Valores unicos de CADASTRAL--- {X.CADASTRALQUALITYID.unique()}")
    X = solve_cols_conflictivas(X)
    X = transform_types_str(X)
    ########### color variables ##################
    print("##### GETTING COLOR VARIABLES ##########")
    X = get_mean_color(X)
    cols_color = [col for col in X.columns if "Q_" in col]
    comp = get_pca_colors(X, cols_color)
    X["PCA1"] = comp[:, 0]
    X["PCA2"] = comp[:, 1]
    X["PCA3"] = comp[:, 2]
    # X = get_yuv(X)
    X = create_geovars(X)
    ########### NEIGHBORS VARIABLES ##################
    X = process_cadqual(X)
    vars_neigh = [
        "CONTRUCTIONYEAR",
        "distance_to_transporte",
        "GEOM_R1",
        "GEOM_R4",
        "GEOM_R3",
        "GEOM_R2"
    ]

    points = [(lon, lat) for lon, lat in zip(X["lon"], X["lat"])]
    points_sp = np.array_split(points, mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())
    for var in tqdm(vars_neigh, desc="VARS NEIGH"):
        resp = pool.map(partial(get_neighbors_means, var=var, X=X), points_sp)
        if len(resp) != len(points):
            resp = np.concatenate(resp)
        X[f"NEIGHBORS_{var}"] = X[var].values - resp
        X[f"NEIGHBORS_{var}"] = X[f"NEIGHBORS_{var}"].astype("float64")
        X[f"NEIGHBORS_{var}"].fillna(value=0, inplace=True)
    pool.close()
    X = X.drop(["lon", "lat"], axis=1)
    print(f"En momento 2 el shape es de {X.shape}")
    if process_cat:
        X = pd.get_dummies(X, columns=X.columns[X.dtypes == object])
    print(f"En momento 3 el shape es de {X.shape}")
    cols = X.columns
    X = fill_cods_nas(X)
    X = fix_renta_media_por_hogar(X)
    print(f"Las columnas que tienen dtype object son {X.columns[X.dtypes == object]}")
    X = fix_nas_categorical(X)
    print(X.isna().sum())
    X[X == np.inf] = np.nan
    X[X == -np.inf] = np.nan
    print(f"dtypes ==> {X.dtypes}")
    if impute_data:
        X = X.fillna(X.mean())
        X.MAXBUILDINGFLOOR.clip(0.0, 25.0, inplace=True)
        X.CADASTRALQUALITYID.clip(0.0, 12.0, inplace=True)
    else:
        imp = [
            col
            for col in X.columns
            if col not in ["MAXBUILDINGFLOOR", "CADASTRALQUALITYID"]
        ]
        X[imp] = X[imp].fillna(X.mean())
    print(f"En momento 4 el shape es de {X.shape}")
    ########## HERE I TREAT LAT AND LON ########################
    X = geospatial_vars(X)
    print(f"En momento 5 el shape es de {X.shape}")
    select_columns = X.dtypes != object
    select_columns = [
        col
        for col in select_columns
        if col not in ["MAXBUILDINGFLOOR", "CADASTRALQUALITYID"]
    ]
    colnames = X.columns
    print(f"NAs: {X.isna().sum()}")
    X = np.array(X)
    if scale:
        if scaler == "std":
            X[:, select_columns] = stdscaler.fit_transform(X[:, select_columns])
            X = pd.DataFrame(X, columns=colnames)
            X["population_density"] = X["poblacion_cp"] / X["area_cod_postal"]
            save_obj(stdscaler, "global_scaler")
        elif scaler == "minmax":
            X[:, select_columns] = minmax.fit_transform(X[:, select_columns])
        print(f"En momento 6 el shape es de {X.shape}")
    if not process_cat:
        return X, y, encoder
    else:
        return pd.DataFrame(X, columns=colnames), y, encoder


def preprocess_test(
    f, scale=True, scaler="std", process_cat=True, sample_trials=None, X_train=None, impute_data=True
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
    sample_trials
        number of samples to take, if None, full data is returned.
    impute_data: bool
        Whether or not to impute the missing values.
    
    Returns
    -------------
    X
        The matrix with features
    """
    X_train["lon"] = dataset_train["lon"]
    X_train["lat"] = dataset_train["lat"]
    df = pd.read_csv(f)
    if sample_trials is not None:
        df = df.sample(sample_trials)
    X = df.drop(["ID", "cluster", "Oeste"], axis=1)
    if 'Unnamed: 0' in X:
        X.drop('Unnamed: 0', axis=1, inplace=True)
    print(f"Valores unicos de CADASTRAL--- {X.CADASTRALQUALITYID.unique()}")
    X = solve_cols_conflictivas(X)
    X = transform_types_str(X)
    print("##### GETTING COLOR VARIABLES ##########")
    X = get_mean_color(X)
    cols_color = [col for col in X.columns if "Q_" in col]
    comp = get_pca_colors(X, cols_color, test=True)
    X["PCA1"] = comp[:, 0]
    X["PCA2"] = comp[:, 1]
    X["PCA3"] = comp[:, 2]
    X = create_geovars(X)
    ########### NEIGHBORS VARIABLES ##################
    X = process_cadqual(X)
   
    vars_neigh = [
        "CONTRUCTIONYEAR",
        "distance_to_transporte",
        "GEOM_R1",
        "GEOM_R4",
        "GEOM_R3",
        "GEOM_R2"
    ]

    points = [(lon, lat) for lon, lat in zip(X["lon"], X["lat"])]
    points_sp = np.array_split(points, mp.cpu_count())
    pool = mp.Pool(processes=mp.cpu_count())
    for var in tqdm(vars_neigh, desc="VARS NEIGH"):
        resp = pool.map(partial(get_neighbors_means, var=var, X=X_train), points_sp)
        if len(resp) != len(points):
            resp = np.concatenate(resp)
        X[f"NEIGHBORS_{var}"] = X[var].values - resp
        X[f"NEIGHBORS_{var}"] = X[f"NEIGHBORS_{var}"].astype("float64")
        X[f"NEIGHBORS_{var}"].fillna(value=0, inplace=True)
    pool.close()
    X.drop(["lon", "lat"], axis=1, inplace=True)
    X_train.drop(["lon", "lat"], axis=1, inplace=True)

    print(f"En momento 2 el shape es de {X.shape}")
    if process_cat:
        X = pd.get_dummies(X, columns=X.columns[X.dtypes == object])
    print(f"En momento 3 el shape es de {X.shape}")

    cols = X.columns

    X = fill_cods_nas(X)
    X = fix_renta_media_por_hogar(X)
    print(f"Las columnas que tienen dtype object son {X.columns[X.dtypes == object]}")
    X = fix_nas_categorical(X)
    print(X.isna().sum())
    X[X == np.inf] = np.nan
    X[X == -np.inf] = np.nan
    print("Imputando valores con Random Forest")

    # imputer.fit(X.loc[:, X.columns[X.dtypes != object]])
    # X.loc[:, X.columns[X.dtypes != object]] = imputer.transform(
    #    X.loc[:, X.columns[X.dtypes != object]]
    # )
    # with open("imputer.pkl", "wb") as f:
    #    pickle.dump(imputer, f)
    print(f"dtypes ==> {X.dtypes}")
    if impute_data:
        X = X.fillna(X_train.mean())
        X.MAXBUILDINGFLOOR.clip(0.0, 25.0, inplace=True)
        X.CADASTRALQUALITYID.clip(0.0, 12.0, inplace=True)
    else:
        imp = [
            col
            for col in X.columns
            if col not in ["MAXBUILDINGFLOOR", "CADASTRALQUALITYID", "population_density"]
        ]
        X[imp] = X[imp].fillna(X_train[imp].mean())
    print(f"En momento 4 el shape es de {X.shape}")

    X.MAXBUILDINGFLOOR.clip(0.0, 25.0, inplace=True)
    X.CADASTRALQUALITYID.clip(0.0, 12.0, inplace=True)

    ########## HERE I TREAT LAT AND LON ########################
    X = geospatial_vars(X)
    print(f"En momento 5 el shape es de {X.shape}")
    select_columns = X.dtypes != object
    select_columns = [
        col
        for col in select_columns
        if col not in ["MAXBUILDINGFLOOR", "CADASTRALQUALITYID"]
    ]
    colnames = X.columns
    X = np.array(X)
    if scale:
        if scaler == "std":
            stdscaler = load_obj("global_scaler")
            X[:, select_columns] = stdscaler.transform(X[:, select_columns])
            X = pd.DataFrame(X, columns=colnames)
            X["population_density"] = X["poblacion_cp"] / X["area_cod_postal"]
        elif scaler == "minmax":
            X[:, select_columns] = minmax.fit_transform(X[:, select_columns])
        print(f"En momento 6 el shape es de {X.shape}")
    if not process_cat:
        return X
    else:
        return X


