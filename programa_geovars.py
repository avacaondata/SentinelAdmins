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

scaler_lon = StandardScaler()
scaler_lat = StandardScaler()
inProj = Proj(init='epsg:25830')
outProj = Proj(init='epsg:4326')

COD = 'geo_'
train_df = pd.read_csv('dataset_train.csv')


def get_dfs(d):
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


def get_lon_lat(df, nombre):
    lat, lon = [], []
    for index, row in df.iterrows():
        try:
            for pt in list(row['geometry'].exterior.coords):
                lat.append(pt[1])
                lon.append(pt[0])
        except Exception as e:
            try:
                row.geometry = row.geometry.map(lambda x: x.convex_hull)
                for pt in list(row['geometry'].exterior.coords):
                    lat.append(pt[1])
                    lon.append(pt[0])
            except Exception as e:
                try:
                    point = df.iloc[index].geometry.centroid
                    lat.append(point.y)
                    lon.append(point.x)
                except Exception as e:
                    print(e)
    latnew, lonnew = [], []
    for la, lo in zip(lat, lon):
        o, a = transform(inProj,outProj,lo, la)
        if o != float("inf") and a != float("inf"):
            latnew.append(a)
            lonnew.append(o)
    return {nombre: {'lon':lonnew, 'lat':latnew}}


def get_clusters(nombre):
    lon, lat = mydic[nombre]['lon'], mydic[nombre]['lat']
    scaled_lon = scaler_lon.transform(np.array(lon).reshape(-1, 1))
    scaled_lat = scaler_lat.transform(np.array(lat).reshape(-1, 1))
    clusters = kmeans.predict(pd.DataFrame(
                                {'x': [l for l in scaled_lat],
                                'y':[l for l in scaled_lon]}))
    return clusters


def get_suma_var(nombre):
    print(mydic[nombre]['clusters'])
    contador = dict(Counter([c for c in mydic[nombre]['clusters']]))
    contador = {int(k):int(v) for k, v in contador.items()}
    #print(contador)
    return contador


def get_individual_df(nombre):
    clusters = []
    contadores = []
    for k, v in mydic[nombre]['contador'].items():
        clusters.append(k)
        contadores.append(v)
    return pd.DataFrame({'cluster': clusters, f'contadores_{nombre}':contadores})


if __name__ == '__main__':
    print("###### abriendo dfs ##########")
    dfs, nombres = get_dfs('./nomecalles2')
    kmeans = KMeans(n_clusters=30, max_iter = 1000, random_state=42)
    mydic = {}
    print('########### COGIENDO LAT LON ##########')
    for df, nombre in zip(dfs, nombres):
        try:
            mydic.update(get_lon_lat(df, nombre))
        except Exception as e:
            print(e)
            continue
    train_df.drop('Unnamed: 0', axis=1, inplace=True)
    scaler_lat.fit(train_df.lat.values.reshape(-1, 1))
    scaler_lon.fit(train_df.lon.values.reshape(-1, 1))
    lat_scaled = scaler_lat.transform(train_df.lat.values.reshape(-1, 1)).reshape(-1, 1)
    lon_scaled = scaler_lon.transform(train_df.lon.values.reshape(-1, 1)).reshape(-1, 1)
    #print(lat_scaled)
    #print(lat_scaled.shape)
    kmeans.fit(pd.DataFrame({'x':[l for l in lat_scaled], 'y':[l for l in lon_scaled]}))
    with open('kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    print('####### COGIENDO CLUSTERS #########')
    for nombre in nombres:
        mydic[nombre]['clusters'] = get_clusters(nombre)
    print("######### SUMA VAR ############")
    for nombre in nombres:
        mydic[nombre]['contador'] = get_suma_var(nombre)
    print('########### INDIVIDUAL DFS ###########')
    processed_dfs = [get_individual_df(nombre) for nombre in nombres]
    print(processed_dfs[0].head())
    print(processed_dfs[10].head())
    print('####### MERGEANDO #########')
    df_final = reduce(lambda left,right: pd.merge(left,right,on='cluster', how='outer', sort=True),
                                                  processed_dfs)
    df_final.fillna(0, inplace=True)
    clusters_orig = kmeans.predict(pd.DataFrame({'x':[l for l in lat_scaled], 'y':[l for l in lon_scaled]}))
    train_df['cluster'] = clusters_orig
    merged_df = pd.merge(train_df, df_final, on='cluster', how='outer')
    merged_df.to_csv('TOTAL_TRAIN.csv', header=True, index=False)
    print('********** Finalizado ***********')
