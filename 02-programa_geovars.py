import geopandas as gpd
from shapely.geometry import Point
from pyproj import Proj
#import pgeocode
import pandas as pd
import os
from pyproj import Proj, transform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pickle
from functools import reduce
import zipfile
import numpy as np
from tqdm import tqdm
import requests 

pd.options.display.max_rows = 10000
scaler_lon = StandardScaler()
scaler_lat = StandardScaler()
inProj = Proj(init='epsg:25830')
outProj = Proj(init='epsg:4326')
distance_thres = 0.0016
COD = 'geo_'
train_df = pd.read_csv('dataset_train.csv')
variables_armando = pd.read_csv('vars_censo_codigo_postal.csv')
cod_postales = gpd.read_file('codigos_postales_madrid/codigos_postales_madrid.shp')


def get_postal_codes(points):
    codigos = np.zeros((len(points),))
    for i, p in tqdm(enumerate(points)):
        p = Point(p[0], p[1])
        for j in range(cod_postales.shape[0]):
            if cod_postales.geometry.iloc[j].contains(p):
                codigos[i] = cod_postales.geocodigo.iloc[j]
    #assert np.sum(codigos==0) == 0
    return codigos[codigos!=0]


def get_dfs(d):
    '''
    Get nomecalles geopandas dfs and put them in a list so that it's easier to work with them.
    '''
    dfs, nombres =[], []
    for folder in tqdm(os.listdir(d)):
        try:
            nombre = [f for f in os.listdir(f"{d}/{folder}/".replace('.zip', '')) if '.shp' in f][0]
            dfs.append(gpd.read_file(f"{d}/{folder}/{nombre}".replace('.zip', ''),
                                    encoding='latin1'))
            nombres.append(nombre)
        except Exception as e:
            print(e)
    return dfs, nombres


def closest_node(node, nodes):
    '''
    Computes the closest point and the distance to that point between a node and a bunch of nodes.
    '''
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2), np.min(dist_2)



def get_lon_lat(df, ruido=False):
    '''
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
    '''
    lat, lon = [], []
    for index, row in tqdm(df.iterrows()):
        lati, loni = [], []
        try:
            for pt in list(row['geometry'].exterior.coords):
                lati.append(pt[1])
                loni.append(pt[0])
        except Exception as e:
            try:
                row.geometry = row.geometry.map(lambda x: x.convex_hull)
                for pt in list(row['geometry'].exterior.coords):
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
        lat.append(sum(lati)/len(lati))
        lon.append(sum(loni)/len(loni))
    latnew, lonnew = [], []
    for la, lo in zip(lat, lon):
        o, a = transform(inProj,outProj,lo, la)
        if o != float("inf") and a != float("inf"):
            latnew.append(a)
            lonnew.append(o)
    return {'lon':lonnew, 'lat':latnew}


def get_zpae():
    '''
    Returns the corrected geopandas df for ruido.
    '''
    zpae = gpd.read_file('./ZPAE/ZPAE/TODAS_ZPAE_ETRS89.shp')
    mydic = get_lon_lat(zpae, ruido=True)
    mydic['ruido'] = zpae.ZonaSupera
    return pd.DataFrame(mydic)


def get_clusters(nombre):
    '''
    Takes a name and, looking for the lat and lon inside the dictionary of that name,
    it applies a cluster over them and therefore we obtain a cluster assignation per
    observation.
    '''
    lon, lat = mydic[nombre]['lon'], mydic[nombre]['lat']
    scaled_lon = scaler_lon.transform(np.array(lon).reshape(-1, 1))
    scaled_lat = scaler_lat.transform(np.array(lat).reshape(-1, 1))
    clusters = kmeans.predict(pd.DataFrame(
                                {'x': [l for l in scaled_lat],
                                'y':[l for l in scaled_lon]}))
    return clusters


def get_suma_var(nombre):
    '''
    Counts how many obs for the variable <name> are in each cluster.
    '''
    print(mydic[nombre]['CODIGO_POSTAL'])
    contador = dict(Counter([c for c in mydic[nombre]['CODIGO_POSTAL']])) #['clusters']
    contador = {int(k):int(v) for k, v in contador.items()}
    #print(contador)
    return contador


def get_individual_df(nombre):
    '''
    Creates a df with the subdictionary of that variable (name).
    '''
    clusters = []
    contadores = []
    for k, v in mydic[nombre]['contador'].items():
        clusters.append(k)
        contadores.append(v)
    return pd.DataFrame({'CODIGO_POSTAL': clusters, f'contadores_{nombre}':contadores})


def get_madrid_codes(df):
    df.dropna(inplace=True)
    #df_madrid = df[(df.CODIGO_POSTAL>27000) & (df.CODIGO_POSTAL<29000)]
    return df #df_madrid


print("###### abriendo dfs ##########")
conflictivos = []
dfs, nombres = get_dfs('./nomecalles2')
kmeans = KMeans(n_clusters=100, max_iter = 1000, random_state=42)
mydic = {}
print('########### COGIENDO LAT LON ##########')
for df, nombre in zip(dfs, nombres):
    try:
        mydic[nombre] = get_lon_lat(df)
    except Exception as e:
        print(e)
        conflictivos.append(nombre)
        continue
train_df.drop('Unnamed: 0', axis=1, inplace=True)
scaler_lat.fit(train_df.lat.values.reshape(-1, 1))
scaler_lon.fit(train_df.lon.values.reshape(-1, 1))
lat_scaled = scaler_lat.transform(train_df.lat.values.reshape(-1, 1)).reshape(-1, 1)
lon_scaled = scaler_lon.transform(train_df.lon.values.reshape(-1, 1)).reshape(-1, 1)
kmeans.fit(pd.DataFrame({'x':[l for l in lat_scaled], 'y':[l for l in lon_scaled]}))
with open('kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
with open('scaler_lat.pkl', 'wb') as f:
    pickle.dump(scaler_lat, f)
with open('scaler_lon.pkl', 'wb') as f:
    pickle.dump(scaler_lon, f)
print('####### COGIENDO CODIGOS POSTALES #########')
for nombre in tqdm(nombres):
    try:
        #mydic[nombre]['clusters'] = get_clusters(nombre)
        pts = [(lon, lat) for lon, lat in zip(mydic[nombre]['lon'], mydic[nombre]['lat'])]
        mydic[nombre]['CODIGO_POSTAL'] = get_postal_codes(pts)
    except Exception as e:
        print(e)
        mydic.pop(nombre, None)
        conflictivos.append(nombre)
print("######### SUMA VAR ############")
for nombre in tqdm(nombres):
    try:
        mydic[nombre]['contador'] = get_suma_var(nombre)
    except Exception as e:
        print(e)
        print(mydic[nombre])
        conflictivos.append(nombre)
print('########### INDIVIDUAL DFS ###########')
processed_dfs = []
for nombre in tqdm(nombres):
    try:
        processed_dfs.append(get_individual_df(nombre))
    except Exception as e:
        #print(e)
        print(mydic[nombre])
        conflictivos.append(nombre)
print('########## AÑADIENDO VARIABLE DE RUIDO ##############')
zpae = get_zpae()
points = [[l for l in train_df[['lon','lat']].iloc[ii]] for ii in tqdm(range(train_df.shape[0]))]
comparing_points = [[l for l in zpae[['lon','lat']].iloc[ii]] for ii in tqdm(range(zpae.shape[0]))]
closest_nodes = [closest_node(point, comparing_points) for point in tqdm(points)]
distances = [t[1] for t in tqdm(closest_nodes)]
which_points = [t[0] for t in tqdm(closest_nodes)]
ruidos = []
dists = []
for i in tqdm(range(train_df.shape[0])):
    if distances[i] >= distance_thres:
        ruidos.append(zpae['ruido'].iloc[which_points[i]])
    else:
        ruidos.append('ruido_No_Registrado')
    dists.append(distances[i])
train_df['ruido'] = ruidos
train_df['distancias_al_ruido'] = dists
print('####### MERGEANDO #########')
train_points = [(lon, lat) for lon, lat in zip(train_df.lon, train_df.lat)]
train_df['CODIGO_POSTAL'] = get_postal_codes(train_points)

df_final = reduce(lambda left,right: pd.merge(left,right,on='CODIGO_POSTAL', how='outer', sort=True),
                                              processed_dfs)
df_final.fillna(0, inplace=True)
clusters_orig = kmeans.predict(pd.DataFrame({'x':[l for l in lat_scaled], 'y':[l for l in lon_scaled]}))
train_df['cluster'] = clusters_orig
distances = kmeans.transform(pd.DataFrame({'x':[l for l in lat_scaled], 'y':[l for l in lon_scaled]}))
distances_to_centroids = []
for i in tqdm(range(train_df.shape[0])):
    distances_to_centroids.append(distances[i, train_df['cluster'].iloc[i]]) #se podría hacer un np.min(..., axis=1) y debería salir igual, pero no se nota tanto el coste (unos segundos), y así nos aseguramos.
train_df['distance_to_centroid'] = distances_to_centroids

print(f'##################### Codigos Postales Unicos: \n train df {train_df.CODIGO_POSTAL.unique()}\
       and \n {df_final.CODIGO_POSTAL.unique()}; lens: {len(train_df.CODIGO_POSTAL.unique())}\
       and {len(df_final.CODIGO_POSTAL.unique())}')

merged_df = pd.merge(train_df, df_final, on='CODIGO_POSTAL', how='inner')
cols_with_nas = ['MAXBUILDINGFLOOR', 'CADASTRALQUALITYID']
print(f"En el momento 4 el shape es de {merged_df.shape}")
cols_imputar = []
for col in merged_df:
    if col not in cols_with_nas:
        cols_imputar.append(col)
merged_df[cols_imputar].fillna(value=0, inplace=True)
print(f'En el momento 5 el shape es de {merged_df.shape}')

print('###### SACANDO VARIABLES ARMANDO #####')
#comparing_points = [[l for l in variables_armando[['lon','lat']].iloc[ii]] for ii in range(variables_armando.shape[0])]
#closest_nodes = [closest_node(point, comparing_points) for point in points]
#distances = [t[1] for t in closest_nodes]
#which_points = [t[0] for t in closest_nodes]
'''
variables_armando['lon'] = scaler_lon.transform(variables_armando['lon'])
variables_armando['lat'] = scaler_lon.transform(variables_armando['lat'])
variables_armando['cluster'] = kmeans.predict(variables_armando[['lat', 'lon']])
variables_armando.drop(['lon', 'lat', '
'''
#for k in 
#df_arm = pd.DataFrame({k:[variables_armando.loc[which_points[i], k]] for k in variables_armando.columns for i in tqdm(range(len(which_points)))})
'''
for i in tqdm(range(len(which_points))):
    df_arm = pd.concat([df_arm, variables_armando.iloc[which_points[i], :]], ignore_index=True)
'''
#cols_merged = merged_df.columns
#cols_arm = df_arm.columns
#variables_armando.drop(['medianaEdad', 'CODIGO_POSTAL_NUMBER'], axis=1, inplace=True)
#variables_armando = get_madrid_codes(variables_armando)
merged_df.CODIGO_POSTAL = merged_df.CODIGO_POSTAL.astype('float')
variables_armando.CODIGO_POSTAL = variables_armando.CODIGO_POSTAL.astype('float')
merged_df = pd.merge(merged_df, variables_armando, on='CODIGO_POSTAL', how='left')
print(f'En el momento 6 el shape es de {merged_df.shape}')
merged_df.to_csv('TOTAL_TRAIN.csv', header=True, index=False)
print(f'NAs in final DF is {merged_df.isna().sum()}')
print('********** Finalizado ***********')
print(f'****************** \n Los archivos conflictivos han sido \n {set(conflictivos)} ************')
