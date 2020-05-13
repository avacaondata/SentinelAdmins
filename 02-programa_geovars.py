##########################################
### AUTHOR: ALEJANDRO VACA SERRANO #######
##########################################


import geopandas as gpd
from shapely.geometry import Point
from pyproj import Proj
import pandas as pd
import os
from pyproj import Proj
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pickle
from functools import reduce
import zipfile
import numpy as np
from tqdm import tqdm
import requests
import time
from functools import partial
import rdflib
from shapely.ops import transform
import pyproj
import multiprocessing as mp

project = partial(
    pyproj.transform,
    pyproj.Proj(init="epsg:25830"),  # source coordinate system
    pyproj.Proj(init="epsg:4326"),
)

pd.options.display.max_rows = 10000
scaler_lon = StandardScaler()
scaler_lat = StandardScaler()
inProj = Proj(init="epsg:25830")
outProj = Proj(init="epsg:4326")
distance_thres = 0.0016
COD = "geo_"
train_df = pd.read_csv(
    "dataset_train.csv"
)  # pd.read_csv("dataset_train.csv")  # dataset_train.csv
MODE = "test"
if MODE == "test":
    test_df = pd.read_csv("dataset_test.csv")
variables_armando = pd.read_csv("vars_censo_codigo_postal_def.csv")
if "CODIGO_MUNICIPIO" in variables_armando.columns:
    variables_armando = variables_armando.drop("CODIGO_MUNICIPIO", axis=1)

cod_postales = gpd.read_file("codigos_postales_madrid/codigos_postales_madrid.shp")
actividad_empresarial = gpd.read_file(
    "./nomecalles2/Colectivo empresarial por tamaño y actividad/dir2019a.shp"
)
zonas_metropolitanas = gpd.read_file("./Zonas Metropolitanas/200001093.shp")
zonas_educativas = gpd.read_file("./Zonas Educativas/200001703.shp")
alt = gpd.read_file("./altimetria/SIGI_MA_ALTIMETRIA_100Line.shp")
aire = gpd.read_file("./calidad_aire/SIGI_MA_ZONAS_CALIDAD_AIREPolygon.shp")
m30 = gpd.read_file("./m_30/POLYLINE.shp")
m40 = gpd.read_file("./m_40/POLYLINE.shp")
castellana = gpd.read_file("./paseo_castellana/POLYLINE.shp")

trafico = gpd.read_file("./shp_imd/IMD_2013.shp")

trafico = trafico[trafico["HP T"] != "-"]
trafico = trafico[trafico["Valor HP T"] != "-"]
trafico[["HP T", "Valor HP T"]] = trafico[["HP T", "Valor HP T"]].astype("float")
zonas_metropolitanas.geometry = [
    transform(project, zonas_metropolitanas.geometry.iloc[i])
    for i in range(zonas_metropolitanas.shape[0])
]
alt.geometry = [transform(project, alt.geometry.iloc[i]) for i in range(alt.shape[0])]
aire.geometry = [
    transform(project, aire.geometry.iloc[i]) for i in range(aire.shape[0])
]


def get_bus_locs():
    '''
    Function to get the locations of bus stops, both from local buses in Madrid city
    and inter-city buses in the region of Madrid.

    Returns
    ------------
    An array containing the location of all bus stops in Madrid.
    '''
    g = rdflib.Graph()
    g.load("EMT_2019_12.rdf")
    locs = []
    for s, p, o in g:
        locs.append(o)
    locs = [str(loc).split() for loc in locs if type(loc) == rdflib.term.Literal]
    locs2 = []
    for loc in tqdm(locs, desc="COGIENDO BUSES"):
        if len(loc) == 2:
            try:
                locs2.append((float(loc[1]), float(loc[0])))
            except:
                continue
    g = rdflib.Graph()
    g.load("Interurbanos_2019_12.rdf")
    locs = []
    for s, p, o in g:
        locs.append(o)
    locs = [str(loc).split() for loc in locs if type(loc) == rdflib.term.Literal]
    locs3 = []
    for loc in tqdm(locs, desc="INTERURBANOS"):
        if len(loc) == 2:
            try:
                locs3.append((float(loc[1]), float(loc[0])))
            except:
                continue
    return np.concatenate([locs2, locs3])


def get_traffic(pts=None):
    '''
    Get the traffic for the closest place reported to each point in pts.
    The traffic variables are: the top traffic time in the day, the
    traffic at that time and the distance from the point to the traffic-recorded
    place that is closest. This is because only certain places in Madrid record
    traffic, therefore the traffic assigned to a point that is too far away 
    from the closest traffic-recorded place is less likely to be informative.

    Parameters
    --------------
    pts: iterable of points (lon, lat)
        List or array containing points in the form (lon, lat)
    
    Returns
    --------------
    todo: 2-d array
        Array containing the variables traffic_hour, traffic, dist_traffic (axis 1)
        for each point in pts (axis 0).
    '''

    todo = np.empty((len(pts), 3), dtype="float64")
    for i in tqdm(range(len(pts)), desc="GETTING TRAFFIC"):
        try:
            pt = pts[i]
        except:
            pt = pts[i, :]
        p = Point(pt[0], pt[1])
        which = np.argmin(
            [p.distance(trafico.geometry.iloc[j]) for j in range(trafico.shape[0])]
        )
        traffic_hour = trafico["HP T"].iloc[which]
        traffic = trafico["Valor HP T"].iloc[which]
        dist_traffic = np.min(
            [p.distance(trafico.geometry.iloc[j]) for j in range(trafico.shape[0])]
        )
        todo[i, 0] = traffic_hour
        todo[i, 1] = traffic
        todo[i, 2] = dist_traffic
    return todo


def get_points_density(df, around=5):
    """
    Function to get the points density some kilometers around.

    Parameters
    -------------
    df: pandas.DataFrame
        df containing the points and the poblacion per postal code variables.
    around: int or float
        the number of km to define the space around the point for density estimation
    
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
        points_around = df.iloc[mybool, :].shape[0]
        population_postal_code = df.poblacion.iloc[i]
        densities[i] = points_around / population_postal_code
        # TODO : PROBAR SOLO CON LOS PUNTOS AL REDEDOR O ESCALÁNDOLO DE OTRA MANERA.
    return densities



def get_distance_to_place(points, place):
    """
    Get distance of points to a place, represented by a LineString, Polygon or Multipolygon.

    Parameters
    -------------
    points: list of tuples
        Points in the space in format (lon, lat)
    place: geopandas.DataFrame
        A geopandas DataFrame containing exactly 1 row. This row must contain the geometry of the
        place you want to calculate the distances to. 
    
    Returns
    -------------
    Distances.
    """
    distances = np.empty((len(points),), dtype="float64")
    for i, point in tqdm(enumerate(points), desc="GETTING POINTS DISTANCES"):
        p = Point(point[0], point[1])
        distances[i] = p.distance(place["geometry"].iloc[0])
    return distances


def get_altitude(points):
    """
    Gets altitudes for each point in the dataset.

    Parameters
    ------------
    points: list of tuples
        Points of the dataset (lon, lat)
    
    Returns
    ------------
    An array of altitudes for points.
    """
    altitudes = np.zeros((len(points),), dtype="float64")
    for i, point in tqdm(enumerate(points), desc="GETTING ALTITUDE"):
        p = Point(point[0], point[1])
        altitudes[i] = alt.NM_COTA.iloc[
            np.argmin([p.distance(alt.geometry.iloc[j]) for j in range(alt.shape[0])])
        ]
    return altitudes


def air_quality(points):
    """
    Get air quality.

    Parameters
    -------------
    points: list of tuples
        points in the form (lon, lat)
    
    Returns
    -------------
    qualities: array 

    """
    qualities = np.empty((len(points),), dtype=np.dtype("U100"))
    qualities[:] = "No_Registrada"
    for i, point in tqdm(enumerate(points), desc="GETTING AIR QUALITY"):
        p = Point(point[0], point[1])
        for j in range(aire.shape[0]):
            if aire.geometry[j].contains(p):
                qualities[i] = str(aire.CD_ZONA.iloc[j])
    return qualities



def get_close_interest_points(
    points, dicvar=None, var=None, around=2, points_compare=None
):
    """
    This function takes an iterable of points (lon, lat), and for each of those
    points, it checks how many "var" are in 2 km around (square-form). "var" here
    refers to the name of the "interesting point" from Nomecalles, therefore
    this function will return how many churches are 2km around of each of the points,
    how many hospitals, companies, etc.

    Parameters
    ---------------
    points: list
        Geographical points in the form (lon, lat)
    dicvar: dict
        The dictionary with the lon, lat of the variable points.
    var: str
        The name of the variable.
    around: float
        The range to search for interest points.
    points_compare: list of tuples
        In case one does not want to compare the points against a variable in the dictionary,
        one can alternative pass some points to compare against.
    
    Returns
    ----------------
    points_around: iterable of floats
        The number of "var" points around each of the points.
    """
    grad_to_lat = 1 / 111
    grad_to_lon = 1 / 85
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    min_lons = [lon - around * grad_to_lon for lon in lons]
    max_lons = [lon + around * grad_to_lon for lon in lons]
    min_lats = [lat - around * grad_to_lat for lat in lats]
    max_lats = [lat + around * grad_to_lat for lat in lats]
    if points_compare is None:
        df = pd.DataFrame({"lon": dicvar["lon"], "lat": dicvar["lat"]})
    elif points_compare is not None:
        lons = [p[0] for p in points]
        lats = [p[1] for p in points]
        df = pd.DataFrame({"lon": lons, "lat": lats})
    points_around = np.empty((len(points),), dtype="float64")
    i = 0
    for min_lon, max_lon, min_lat, max_lat in tqdm(
        zip(min_lons, max_lons, min_lats, max_lats),
        desc=f"### GETTING POINTS FOR {var}",
    ):
        mybool = (
            (df["lat"] >= min_lat)
            & (df["lat"] <= max_lat)
            & (df["lon"] >= min_lon)
            & (df["lon"] <= max_lon)
        )
        try:
            points_around[i] = df.loc[mybool, :].shape[0]
        except:
            points_around[i] = 0
        i += 1
    return points_around


def get_zona_metropolitana_o_educativa(points, mode="metropolitana"):
    """
    Function to retrieve the metropolitan area to which a point in the map belongs. 

    Parameters
    ---------------
    points
        a list of points in lon, lat format
    
    Returns
    ----------------
    The metropolitan area for each of those points.
    """
    if mode == "metropolitana":
        z = zonas_metropolitanas.copy()
    elif mode == "educativa":
        z = zonas_educativas.copy()
    zonas = np.empty((len(points),), dtype=np.dtype("U100"))
    zonas[:] = "no_registrada" if mode == "metropolitana" else "No Educativa"
    for i, p in tqdm(enumerate(points), desc=f"GETTING {mode}"):
        p = Point(p[0], p[1])
        for j in range(z.shape[0]):
            if z.geometry.iloc[j].contains(p):
                zonas[i] = z.DESCCODIGO.iloc[j]
    return zonas


def get_postal_codes(pts):
    """
    Function to retrieve the postal code for points, where points are in the form (lon, lat).
    Using the cod_postales df, which has the polygons for each postal code, it checks which
    of those polygons each point falls into.

    Parameters
    ------------
    points
        a list or array-like with the points (lon, lat)
    
    Returns
    ------------
    The postal code for each of those points. It excludes the zeroes as they mean the postal code
    for that point could not be retrieved.
    """
    codigos = np.zeros((len(pts),))
    for i, p in tqdm(enumerate(pts), desc="GETTING POSTAL CODES"):
        p = Point(p[0], p[1])
        for j in range(cod_postales.shape[0]):
            if cod_postales.geometry.iloc[j].contains(p):
                codigos[i] = cod_postales.geocodigo.iloc[j]
    return codigos[codigos != 0]



def get_dfs(d):
    """
    Get nomecalles geopandas dfs and put them in a list so that it's easier to work with them.

    Parameters
    ------------
    d
        Directory where the .shp files of nomecalles are.
    
    Returns
    ------------
    dfs
        List of geopandas dataframes from nomecalles.
    nombres
        Names of the variable contained in each of those dataframes.
    """
    dfs, nombres = [], []
    for folder in tqdm(os.listdir(d), desc="GETTING DFS"):
        try:
            nombre = [
                f
                for f in os.listdir(f"{d}/{folder}/".replace(".zip", ""))
                if ".shp" in f
            ][0]
            dfs.append(
                gpd.read_file(
                    f"{d}/{folder}/{nombre}".replace(".zip", ""), encoding="latin1"
                )
            )
            nombres.append(nombre)
        except Exception as e:
            print(e)
    return dfs, nombres


def closest_node(node, nodes):
    """
    Computes the closest point and the distance to that point between a node and a bunch of nodes.

    Parameters
    ------------
    node
        The node for which we want to find the closest point.
    nodes
        The nodes to compare against. 
    
    Returns
    ------------
    The closest point to "node" among "nodes".
    """
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.argmin(dist_2), np.min(dist_2)


def get_lon_lat(df, nombre, ruido=False):
    """
    This function receives a geopandas df, and a desired name for the variable
    extracted from that dataframe. It inspects the geometric objects inside the
    geopandas df and returns the latitude and longitude of each observation.
    
    Parameters
    ---------------
    df
        geopandas df.
    nombre
        the desired name for the variable.
    ruido
        This boolean controls whether we are calling it for the nomecalles variables 
        or for the "ruido" one.
    
    Return
    ----------------
    Dic
        {nombre: {'lat':..., 'lon':...}}
    """
    from pyproj import transform

    lat, lon = [], []
    for index, row in tqdm(df.iterrows()):
        lati, loni = [], []
        try:
            for pt in list(row["geometry"].exterior.coords):
                lati.append(pt[1])
                loni.append(pt[0])
        except Exception as e:
            try:
                row.geometry = row.geometry.map(lambda x: x.convex_hull)
                for pt in list(row["geometry"].exterior.coords):
                    lati.append(pt[1])
                    loni.append(pt[0])
            except Exception as e:
                try:
                    lati.append(df.iloc[index].geometry.centroid.y)
                    loni.append(df.iloc[index].geometry.centroid.x)
                except Exception as e:
                    if not ruido:
                        continue
                    else:
                        print(e)
                        print(df.iloc[index].geometry.centroid)
        lat.append(sum(lati) / len(lati))
        lon.append(sum(loni) / len(loni))
    latnew, lonnew = [], []
    for la, lo in zip(lat, lon):
        o, a = transform(inProj, outProj, lo, la)
        if o != float("inf") and a != float("inf"):
            latnew.append(a)
            lonnew.append(o)
    return {nombre: {"lon": lonnew, "lat": latnew}}


def get_zpae():
    """
    Returns the corrected geopandas df for ruido.
    """
    zpae = gpd.read_file("./ZPAE/ZPAE/TODAS_ZPAE_ETRS89.shp")
    mydic = get_lon_lat(zpae, "zpae", ruido=True)
    mydic["zpae"]["ruido"] = zpae.ZonaSupera
    return pd.DataFrame(mydic["zpae"])


def get_clusters(nombre):
    """
    Takes a name and, looking for the lat and lon inside the dictionary of that name,
    it applies a cluster over them and therefore we obtain a cluster assignation per
    observation. This is no longer used, as finally the nomecalles variables are merged
    by postal code, not by cluster.

    Parameters
    ------------
    nombre
        The name of the variable from nomecalles.
    
    Returns
    ------------
    clusters
        The cluster corresponding to each member or unit of that variable. 
    """
    lon, lat = mydic[nombre]["lon"], mydic[nombre]["lat"]
    scaled_lon = scaler_lon.transform(np.array(lon).reshape(-1, 1))
    scaled_lat = scaler_lat.transform(np.array(lat).reshape(-1, 1))
    clusters = kmeans.predict(
        pd.DataFrame({"x": [l for l in scaled_lat], "y": [l for l in scaled_lon]})
    )
    return clusters


def get_suma_var(nombre):
    """
    Counts how many obs for the variable <name> are in each postal code.

    Parameters
    ------------
    nombre
        The name of the variable from nomecalles
    
    Returns
    ------------
    contador
        A Counter dict with the form {28007:1, 28003:5, 28100:2...}, that is,
        with the postal codes as keys and the members of the nomecalles variable
        found inside that postal code as values.
    """
    contador = dict(Counter([c for c in mydic[nombre]["CODIGO_POSTAL"]]))
    contador = {int(k): int(v) for k, v in contador.items()}
    return contador


def get_individual_df(nombre):
    """
    Creates a df with the subdictionary of that variable (name).

    Parameters
    -----------
    nombre
        The name of the nomecalles variable
    
    Returns
    -----------
    A pandas.DataFrame with 2 columns: CODIGO_POSTAL and contadores_{nombre}, 
    """
    clusters = []
    contadores = []
    for k, v in mydic[nombre]["contador"].items():
        clusters.append(k)
        contadores.append(v)
    return pd.DataFrame({"CODIGO_POSTAL": clusters, f"contadores_{nombre}": contadores})


def get_madrid_codes(df):
    """
    Finally this does not filter Madrid, as it's not necessary for the merge.
    It only gets rid of NAs.
    """
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    start_time = time.time()
    print("###### abriendo dfs ##########")
    conflictivos = []
    dfs, nombres = get_dfs("./nomecalles2")
    kmeans = KMeans(n_clusters=100, max_iter=1000, random_state=42)

    if MODE != "test":
        points = [(lon, lat) for lon, lat in zip(train_df.lon, train_df.lat)]
        points_sp = np.array_split(points, mp.cpu_count())
    else:
        points = [(lon, lat) for lon, lat in zip(test_df.lon, test_df.lat)]
        points_sp = np.array_split(points, mp.cpu_count())
    assert len(points[0]) == 2

    '''
    In this section we take the number of bus stations less than 2km close,
    and the traffic variables.
    '''
    pool = mp.Pool(mp.cpu_count())
    print("GETTING BUSES")
    pts_bus = get_bus_locs()
    print(pts_bus)
    resp = pool.map(
        partial(get_close_interest_points, var="BUSES", points_compare=pts_bus),
        points_sp,
    )
    if len(resp) != len(points):
        resp = np.concatenate(resp)
    if MODE != "test":
        train_df["paradas_bus"] = resp
    else:
        test_df["paradas_bus"] = resp

    resp = pool.map(get_traffic, points_sp)
    if len(resp) != len(points):
        resp = np.concatenate(resp)
    if MODE != "test":
        train_df["Hora Punta"] = resp[:, 0]
        train_df["Trafico Hora Punta"] = resp[:, 1]
        train_df["Distance_Trafico"] = resp[:, 2]
    else:
        test_df["Hora Punta"] = resp[:, 0]
        test_df["Trafico Hora Punta"] = resp[:, 1]
        test_df["Distance_Trafico"] = resp[:, 2]

    mydic = {}
    print("########### COGIENDO LAT LON ##########")
    """
    In this section we get the lon and lat for the cinemas, subway stations etc. (nomecalles)
    """
    for df, nombre in zip(dfs, nombres):
        try:
            mydic.update(get_lon_lat(df, nombre))
        except Exception as e:
            print(e)
            conflictivos.append(nombre)
            continue

    """
    In this section we take the total area covered by each postal code.
    """
    areas = np.zeros((cod_postales.shape[0],), dtype="float64")
    for i in tqdm(range(cod_postales.shape[0]), desc="COGIENDO AREA CODIGO POSTAL"):
        areas[i] = cod_postales.geometry.iloc[i].area
    areas_postal_codes = pd.DataFrame(
        {"CODIGO_POSTAL": cod_postales.geocodigo.values, "area_cod_postal": areas}
    )
    """
    Then, we apply kmeans with lon and lat from the train/test dataframe as input variables, with 
    100 cluster, that's dividing the geospace in 100 groups. These clusters will be later used 
    for computing the distance of each point to the centroid of the cluster it belongs to, therefore
    computing distance_to_centroid variable.
    """
    print("AJUSTANDO KMEANS")
    train_df.drop("Unnamed: 0", axis=1, inplace=True)
    if MODE != "test":
        scaler_lat.fit(train_df.lat.values.reshape(-1, 1))
        scaler_lon.fit(train_df.lon.values.reshape(-1, 1))
        lat_scaled = scaler_lat.transform(train_df.lat.values.reshape(-1, 1)).reshape(
            -1, 1
        )
        lon_scaled = scaler_lon.transform(train_df.lon.values.reshape(-1, 1)).reshape(
            -1, 1
        )
        kmeans.fit(
            pd.DataFrame({"x": [l for l in lat_scaled], "y": [l for l in lon_scaled]})
        )
        with open("kmeans.pkl", "wb") as f:
            pickle.dump(kmeans, f)
        with open("scaler_lat.pkl", "wb") as f:
            pickle.dump(scaler_lat, f)
        with open("scaler_lon.pkl", "wb") as f:
            pickle.dump(scaler_lon, f)

    else:
        with open("kmeans.pkl", "rb") as f:
            kmeans = pickle.load(f)
        with open("scaler_lat.pkl", "rb") as f:
            scaler_lat = pickle.load(f)
        with open("scaler_lon.pkl", "rb") as f:
            scaler_lon = pickle.load(f)
        lat_scaled = scaler_lat.transform(test_df.lat.values.reshape(-1, 1)).reshape(
            -1, 1
        )
        lon_scaled = scaler_lon.transform(test_df.lon.values.reshape(-1, 1)).reshape(
            -1, 1
        )
    print("####### COGIENDO CODIGOS POSTALES #########")
    """
    We use close_interest_points to get all the nomecalles variables located in mydic.
    """
    for nombre in tqdm(nombres):
        resp = pool.map(
            partial(get_close_interest_points, dicvar=mydic[nombre], var=nombre),
            points_sp,
        )
        if len(resp) != len(points):
            resp = np.concatenate(resp)
        if MODE != "test":
            train_df[nombre] = resp
        else:
            test_df[nombre] = resp

    '''
    Transport variables (distance to transporte).
    '''
    points_bocas = [
        (lon, lat)
        for lon, lat in zip(mydic["bocas.shp"]["lon"], mydic["bocas.shp"]["lat"])
    ]
    points_renfe = [
        (lon, lat)
        for lon, lat in zip(mydic["estcerca.shp"]["lon"], mydic["estcerca.shp"]["lat"])
    ]
    points_transporte = points_bocas + points_renfe
    closest_nodes_trans = [
        closest_node(point, points_transporte)
        for point in tqdm(points, desc="closest transporte")
    ]
    distances_trans = [t[1] for t in tqdm(closest_nodes_trans)]
    if MODE != "test":
        train_df["distance_to_transporte"] = distances_trans
    else:
        test_df["distance_to_transporte"] = distances_trans

    closest_nodes_bus = [
        closest_node(point, pts_bus) for point in tqdm(points, desc="closest bus")
    ]
    distances_bus = [t[1] for t in tqdm(closest_nodes_bus)]
    if MODE != "test":
        train_df["distance_to_bus"] = distances_bus
    else:
        test_df["distance_to_bus"] = distances_bus

    """
    We transform the subdicts inside mydic into individual dataframes and put them into a list.
    """

    print("########## AÑADIENDO VARIABLE DE RUIDO ##############")

    """
    In this part we get the ZPAE variables, which are the variables representing how much noise
    was reported (no date available) in different parts in Madrid. For that, we get the lon lat points
    in the df, and then we compare them against the lon lat points of the places where noise was reported.
    Then we use closest_node to get the closest "noise-reported" point, and we also get the distance to them.
    A filter is performed, as most points in the dataframe are far away from any noise-reported point (we select
    this threshold with the histogram); to those points distanced more than threshold, we set the variable 
    ruido to be ruido_No_Registrado; and for the rest, the closest reported noise description (is a categorical
    variable). The distance to the closest noise is also a new feature for the dataset.
    """
    zpae = get_zpae()

    comparing_points = [
        [l for l in zpae[["lon", "lat"]].iloc[ii]] for ii in tqdm(range(zpae.shape[0]))
    ]
    closest_nodes = [
        closest_node(point, comparing_points)
        for point in tqdm(points, desc="closest nodes")
    ]
    distances = [t[1] for t in tqdm(closest_nodes)]
    which_points = [t[0] for t in tqdm(closest_nodes)]
    if MODE != "test":
        ruidos = np.empty((train_df.shape[0],), dtype=np.dtype("U100"))
        ruidos[:] = "ruido_No_Registrado"
        for i in tqdm(range(train_df.shape[0]), desc="DISTANCIAS AL RUIDO"):
            if distances[i] <= distance_thres:
                try:
                    ruidos[i] = zpae["ruido"].iloc[which_points[i]]
                except:
                    ruidos[i] = "ruido_Non_Ascii"
        train_df["ruido"] = ruidos
        train_df["distancias_al_ruido"] = distances
    else:
        ruidos = np.empty((test_df.shape[0],), dtype=np.dtype("U100"))
        ruidos[:] = "ruido_No_Registrado"
        for i in tqdm(range(test_df.shape[0]), desc="DISTANCIAS AL RUIDO"):
            if distances[i] <= distance_thres:
                try:
                    ruidos[i] = zpae["ruido"].iloc[which_points[i]]
                except:
                    ruidos[i] = "ruido_Non_Ascii"
        test_df["ruido"] = ruidos
        test_df["distancias_al_ruido"] = distances
    print("####### MERGEANDO #########")

    """
    We get the postal code for each point, and the cluster they belong to.
    Then we compute the distance_to_centroid variable.
    """

    resp = pool.map(get_postal_codes, points_sp)
    if len(resp) != len(points):
        resp = np.concatenate(resp)

    if MODE != "test":
        train_df["CODIGO_POSTAL"] = resp
    else:
        test_df["CODIGO_POSTAL"] = resp
    
    clusters_orig = kmeans.predict(
        pd.DataFrame({"x": [l for l in lat_scaled], "y": [l for l in lon_scaled]})
    )
    if MODE != "test":
        train_df["cluster"] = clusters_orig
    else:
        test_df["cluster"] = clusters_orig
    distances = kmeans.transform(
        pd.DataFrame({"x": [l for l in lat_scaled], "y": [l for l in lon_scaled]})
    )
    if MODE != "test":
        distances_to_centroids = np.empty((train_df.shape[0],), dtype="float64")
        for i in tqdm(range(train_df.shape[0])):
            distances_to_centroids[i] = distances[i, train_df["cluster"].iloc[i]]
            # )  # se podría hacer un np.min(..., axis=1) y debería salir igual, pero no se nota tanto el coste (unos segundos), y así nos aseguramos.
        train_df["distance_to_centroid"] = distances_to_centroids
        merged_df = train_df.copy()
    else:
        distances_to_centroids = np.empty((test_df.shape[0],), dtype="float64")
        for i in tqdm(range(test_df.shape[0])):
            distances_to_centroids[i] = distances[i, test_df["cluster"].iloc[i]]
        test_df["distance_to_centroid"] = distances_to_centroids
        merged_df = test_df.copy()
    
    ############# INCLUYENDO AREA POR CODIGO POSTAL ########################
    areas_postal_codes.CODIGO_POSTAL = areas_postal_codes.CODIGO_POSTAL.astype("float")
    merged_df.CODIGO_POSTAL = merged_df.CODIGO_POSTAL.astype("float")
    merged_df = pd.merge(merged_df, areas_postal_codes, on="CODIGO_POSTAL", how="left")
    cols_with_nas = ["MAXBUILDINGFLOOR", "CADASTRALQUALITYID"]
    print(f"En el momento 4 el shape es de {merged_df.shape}")
    cols_imputar = []
    for col in merged_df:
        if col not in cols_with_nas:
            cols_imputar.append(col)
    merged_df[cols_imputar].fillna(value=0, inplace=True)
    print(f"En el momento 5 el shape es de {merged_df.shape}")

    """
    Aquí sacamos el area metropolitana para cada punto. 
    """

    print(
        "########### COGIENDO ZONA METROPOLITANA Y EDUCATIVA, ALTITUD Y CALIDAD DEL AIRE, TRAFICO ##########"
    )
    resp = pool.map(get_zona_metropolitana_o_educativa, points_sp)
    if len(resp) != len(points):
        resp = np.concatenate(resp)
    merged_df["ZONA_METROPOLITANA"] = resp

    resp = pool.map(get_altitude, points_sp)
    if len(resp) != len(points):
        resp = np.concatenate(resp)
    merged_df["ALTITUD"] = resp

    resp = pool.map(air_quality, points_sp)
    if len(resp) != len(points):
        resp = np.concatenate(resp)
    merged_df["CALIDAD_AIRE"] = resp

    resp = pool.map(partial(get_distance_to_place, place=m30), points_sp)
    if len(resp) != len(points):
        resp = np.concatenate(resp)
    merged_df["distance_m30"] = resp

    resp = pool.map(partial(get_distance_to_place, place=m40), points_sp)
    if len(resp) != len(points):
        resp = np.concatenate(resp)
    merged_df["distance_m40"] = resp

    resp = pool.map(partial(get_distance_to_place, place=castellana), points_sp)
    if len(resp) != len(points):
        resp = np.concatenate(resp)
    merged_df["distance_castellana"] = resp
    pool.close()
    """
    En este apartado sacamos información más granular de la actividad empresarial en ese código postal.
    """
    print("####### SACANDO VARIABLES ACTIVIDAD EMPRESARIAL ##########")
    act_empresarial_points = get_lon_lat(actividad_empresarial, nombre="comercios")
    actividad_empresarial_zara = actividad_empresarial.copy()
    actividad_empresarial_zara["lon"] = [
        lon for lon in act_empresarial_points["comercios"]["lon"]
    ]
    actividad_empresarial_zara["lat"] = [
        lat for lat in act_empresarial_points["comercios"]["lat"]
    ]
    actividad_empresarial_zara1 = actividad_empresarial_zara.loc[
        ["ZARA ESPAÑA" in etiqueta for etiqueta in actividad_empresarial_zara.ETIQUETA],
        ["ETIQUETA", "lon", "lat"],
    ]
    zara_home = actividad_empresarial_zara.loc[
        [
            "ZARA HOME ESPAÑA" in etiqueta
            for etiqueta in actividad_empresarial_zara.ETIQUETA
        ],
        ["ETIQUETA", "lon", "lat"],
    ]
    zara_pts = [
        (lon, lat)
        for lon, lat in zip(
            actividad_empresarial_zara1.lon, actividad_empresarial_zara1.lat
        )
    ]
    zara_home_pts = [(lon, lat) for lon, lat in zip(zara_home.lon, zara_home.lat)]
    act_empresarial_points = [
        (lon, lat)
        for lon, lat in zip(
            act_empresarial_points["comercios"]["lon"],
            act_empresarial_points["comercios"]["lat"],
        )
    ]

    closest_zara = [closest_node(point, zara_pts) for point in tqdm(points)]
    merged_df["dist_closest_zara"] = [t[1] for t in tqdm(closest_zara)]
    closest_zhome = [closest_node(point, zara_home_pts) for point in tqdm(points)]
    merged_df["dist_closest_zara_HOME"] = [t[1] for t in tqdm(closest_zhome)]

    
    print("###### SACANDO VARIABLES ARMANDO #####")
    """
    As variables_armando was obtained with Excel and the Internet, this dataframe already comes with the 
    CODIGO_POSTAL variable to merge these new variables (from "el censo") with the train/test df. The NAs
    in this part are not imputed, as they will be imputed using Random Forest in preprocessing.py
    """
    merged_df.CODIGO_POSTAL = merged_df.CODIGO_POSTAL.astype("float")
    variables_armando.CODIGO_POSTAL = variables_armando.CODIGO_POSTAL.astype("float")
    variables_armando.drop_duplicates(subset=["CODIGO_POSTAL"], inplace=True)
    # variables_armando.drop('poblacion_municipio', axis=1, inplace=True)
    cods_postales_faltan = [
        code
        for code in merged_df.CODIGO_POSTAL.unique()
        if code not in variables_armando.CODIGO_POSTAL.unique()
    ]
    print(
        f"########## FALTAN LOS SIGUIENTES CODIGOS POSTALES: {cods_postales_faltan} ##################"
    )
    merged_df.loc[merged_df.CODIGO_POSTAL == 28907.0, "CODIGO_POSTAL"] = 28903
    merged_df2 = pd.merge(merged_df, variables_armando, on="CODIGO_POSTAL", how="left")
    print(f"En el momento 6 el shape es de {merged_df2.shape}")
    
    if MODE != "test":
        merged_df2.to_csv(
            "TOTAL_TRAIN.csv", header=True, index=False
        )  # TOTAL_TRAIN.csv
    else:
        merged_df2.to_csv("TOTAL_TEST.csv", header=True, index=False)
    print(f"NAs in final DF is {merged_df2.isna().sum()}")
    print("********** Finalizado ***********")
    print(
        f"****************** \n Los archivos conflictivos han sido \n {set(conflictivos)} ************"
    )
    print(
        f"******######## TOTAL TIME: {time.time() - start_time} ############*********"
    )
