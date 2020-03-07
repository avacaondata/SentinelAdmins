# attemp to use data from nomecalles

library(RgoogleMaps)	
library(ggmap)	
library(rgdal)	
library(leaflet)	
library(lubridate)	
library(dplyr)	
library(raster)	

df = read.csv('dataset_train.csv')

map_barrios <- readOGR(dsn = "/Users/alejandrovaca/Documents/SentinelAdmins/Educación_Campus universitarios", layer="campus.shp",encoding = "latin1",stringsAsFactors=F)	
library(raster)
sp = shapefile("/Users/alejandrovaca/Documents/SentinelAdmins/Educación_Campus universitarios/campus")



