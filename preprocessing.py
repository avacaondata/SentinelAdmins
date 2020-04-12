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
from sklearn.impute import SimpleImputer

SAVE_DIR = "objects_fitted"
if SAVE_DIR not in os.listdir():
    os.mkdir(SAVE_DIR)
imputer = MissForest(n_jobs=-1)
vars_postal_code = pd.read_csv("vars_censo_codigo_postal_def.csv")
cod_postales = gpd.read_file("codigos_postales_madrid/codigos_postales_madrid.shp")
categorical = [53, 54]

minmax = MinMaxScaler()
stdscaler = StandardScaler()
encoder = LabelEncoder()


def get_neighbors_means(points, var, X, around=2):
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
        # desc=f"GETTING NEIGHBORS FOR {var}",:
        mybool = (
            (X["lat"] >= min_lat)
            & (X["lat"] <= max_lat)
            & (X["lon"] >= min_lon)
            & (X["lon"] <= max_lon)
        )
        try:
            means_[i] = float(X.loc[mybool, var].mean())
        except:
            means_[i] = 0.0
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


def get_mean_color(df):
    colores = ["R", "G", "B", "NIR"]
    for color in tqdm(colores):
        cols_color = [col for col in df.columns if f"Q_{color}_" in col]
        df[f"media_{color}"] = df[cols_color].mean(axis=1)
    return df


def get_pca_colors(X, cols):
    pca = PCA(n_components=3, random_state=42)
    components = pca.fit_transform(X[cols].values)
    with open(f"./{SAVE_DIR}/pca_colors.pkl", "wb") as f:
        pickle.dump(pca, f)
    return components


def get_pca_geoms(X):
    cols = [col for col in X.columns if "GEOM_" in col]
    pca = PCA(n_components=3, random_state=42)
    components = pca.fit_transform(X[cols].values)
    with open("pca_geoms.pkl", "wb") as f:
        pickle.dump(pca, f)
    return components


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
        # df[df == np.inf] = 0
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


def get_vegetation_indices(df):
    sc = MinMaxScaler()
    R = sc.fit_transform(df["Q_R_4_0_5"].values.reshape(-1, 1))
    save_obj(sc, "scaler_Q_R_4_0_5")
    G = sc.fit_transform(df["Q_G_3_0_5"].values.reshape(-1, 1))
    save_obj(sc, "scaler_Q_G_3_0_5")
    B = sc.fit_transform(df["Q_B_2_0_5"].values.reshape(-1, 1))
    save_obj(sc, "scaler_Q_B_2_0_5")
    N = sc.fit_transform(df["Q_NIR_8_0_5"].values.reshape(-1, 1))
    save_obj(sc, "scaler_Q_NIR_8_0_5")
    df["GRVI"] = (G - R) / (G + R)
    df["MGRVI"] = (G ** 2 - R ** 2) / (G ** 2 + R ** 2)
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
    #df[['NPCRI', 'AVI']] = df[['NPCRI', 'AVI']].fillna(value=0, inplace=True)
    df[['NPCRI', 'AVI']] = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=0).fit_transform(df[['NPCRI', 'AVI']])
    return df


matriz_ = np.matrix("0.299 0.587 0.114; -0.147 -0.289 0.436; 0.615 -0.515 -0.100")


def get_yuv(df, matriz_=matriz_):
    sc = MinMaxScaler()
    qs = [f"_0_{i}" for i in range(10)] + ["_1_0"]
    colors = ["Q_R_4", "Q_G_3", "Q_B_2"]
    for q in qs:
        cols_select = [f"{col}{q}" for col in colors]
        mat_col = sc.fit_transform(df[cols_select].values.T)
        save_obj(sc, f"scaler_yuv_{q}")
        yuv = np.dot(matriz_, mat_col)
        print(yuv.shape)
        df[f"Y_YUV_{q[1:]}"] = yuv[0, :].T
        df[f"U_YUV_{q[1:]}"] = yuv[1, :].T
        df[f"V_YUV_{q[1:]}"] = yuv[2, :].T
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
    cols_geoms = ['GEOM_R1', 'GEOM_R2', 'GEOM_R3', 'GEOM_R4']
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


def transform_types_str(X):
    X.CADASTRALQUALITYID = X.CADASTRALQUALITYID.astype("str")
    X.CODIGO_POSTAL = X.CODIGO_POSTAL.astype("str")
    X.ruido = X.ruido.astype("str")
    X.CALIDAD_AIRE = X.CALIDAD_AIRE.astype("str")
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
    X = df.drop(["CLASE", "ID", "cluster"], axis=1)
    print(f"Valores unicos de CADASTRAL--- {X.CADASTRALQUALITYID.unique()}")
    X = solve_cols_conflictivas(X)
    X = transform_types_str(X)
    ########### color variables ##################
    print("##### GETTING COLOR VARIABLES ##########")
    X = get_mean_color(X)
    X = get_vegetation_indices(X)
    X = get_vegetation_indices_deciles(X)
    cols_color = [col for col in X.columns if "Q_" in col]
    comp = get_pca_colors(X, cols_color)
    X["PCA1"] = comp[:, 0]
    X["PCA2"] = comp[:, 1]
    X["PCA3"] = comp[:, 2]
    X = get_yuv(X)
    ########### NEIGHBORS VARIABLES ##################
    vars_veget = [
        "GRVI",
        "MGRVI",
        "RGBVI",
        "ExG",
        "ExGR",
        "NDVI",
        "SAVI",
        "MSAVI",
        "GNDVI",
        "OSAVI",
        "GCI",
        "BAI",
        "NDWI",
    ]
    vars_color = ["media_R", "media_G", "media_B", "media_NIR"]
    vars_pca = [f"PCA{i}" for i in range(1, 4)]
    vars_yuv = ["Y_YUV_0_5", "U_YUV_0_5", "V_YUV_0_5"]
    vars_neigh = [
        col
        for col in X.columns
        if "MAXBUILDINGFLOOR" in col
        or "GEOM_" in col
        or col in vars_veget
        or col in vars_color
        or col == "AREA"
        or col in vars_pca
        or col in vars_yuv
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
    X.drop(["lon", "lat"], axis=1, inplace=True)
    # "lon", "lat"
    ########### suma geoms #######################
    #X["suma_geoms"] = X[[col for col in X.columns if "GEOM_" in col]].sum(axis=1)
    ########### HERE WE DEAL WITH GEOM VARS AND CREATE NEW GEOM VARS ############
    # X = get_pca_geoms(X)
    # comp_geoms = get_pca_geoms(X)
    # X['PCA1_GEOM'] = comp_geoms[:, 0]
    X = create_geovars(X)
    X = process_cadqual(X)
    print(f"En momento 2 el shape es de {X.shape}")
    ##########
    if process_cat:
        X = pd.get_dummies(X, columns=X.columns[X.dtypes == object])
    print(f"En momento 3 el shape es de {X.shape}")

    cols = X.columns

    X = fill_cods_nas(X)
    X = fix_renta_media_por_hogar(X)
    print(f"Las columnas que tienen dtype object son {X.columns[X.dtypes == object]}")
    X = fix_nas_categorical(X)
    X[X == np.inf] = np.nan
    X[X == -np.inf] = np.nan
    with open("X_analizar.pkl", "wb") as f:
        pickle.dump(X, f)
    print("Imputando valores con Random Forest")

    # imputer.fit(X.loc[:, X.columns[X.dtypes != object]])
    # X.loc[:, X.columns[X.dtypes != object]] = imputer.transform(
    #    X.loc[:, X.columns[X.dtypes != object]]
    # )
    # with open("imputer.pkl", "wb") as f:
    #    pickle.dump(imputer, f)
    print(f"dtypes ==> {X.dtypes}")
    X = X.fillna(X.mean())
    print(f"En momento 4 el shape es de {X.shape}")
    # X = pd.DataFrame(X, columns=cols)
    # X["point_density"] = get_points_density(df=df, around=5, pobs=X.poblacion.values)

    X.MAXBUILDINGFLOOR.clip(0.0, 25.0, inplace=True)
    X.CADASTRALQUALITYID.clip(0.0, 12.0, inplace=True)

    ########## HERE I TREAT LAT AND LON ########################
    X = geospatial_vars(X)
    print(f"En momento 5 el shape es de {X.shape}")
    select_columns = X.dtypes != object
    colnames = X.columns
    X = np.array(X)
    if scale:
        if scaler == "std":
            X[:, select_columns] = stdscaler.fit_transform(X[:, select_columns])
            X = pd.DataFrame(X, columns=colnames)
            X["population_density"] = X["poblacion_cp"] / X["area_cod_postal"]
            X[select_columns] = X[select_columns].astype('float')
            with open("SCALER.pkl", "wb") as f:
                pickle.dump(stdscaler, f)
        elif scaler == "minmax":
            X[:, select_columns] = minmax.fit_transform(X[:, select_columns])
        print(f"En momento 6 el shape es de {X.shape}")
    if not process_cat:
        return X, y, encoder
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
