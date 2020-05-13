## interpolacion

library(leaflet)
library(tidyverse)
library(ggplot2)
library(plotly)

df = read.csv('Modelar_UH2020.txt', sep='|')

puntos_identificados <- data.frame(
  Lugar = c("Retiro","Legazpi", "Plaza de Toros","Arguelles", 
            "Cuatro Caminos", "Zapateria Robledo","Tetuan"),
  X = c(2209489320, 2209747825, 2219461192, 2196313578, 2201844156, 2196736748, 2203891600),
  Y = c(165537783, 165397664, 165616041, 165611443, 165677624, 165795600, 165734820),
  Lat = c(40.413588, 40.386163, 40.432474, 40.430805, 40.447087, 40.475766, 40.460913),
  Lon = c(-3.683393, -3.680577, -3.663236, -3.716191, -3.703295, -3.71556, -3.698528)
)

puntos_identificados$Sur <- ifelse(165724537<puntos_identificados$Y, 1,0)
puntos_identificados$Oeste <- ifelse(2208137136<puntos_identificados$X, 1, 0)
df$Sur <- ifelse(165724537<df$Y, 1,0)
df$Oeste <- ifelse(2208137136<df$X, 1, 0)

model_lat <- lm(Lat ~ Y + Sur + Oeste, data = puntos_identificados)
model_lon <- lm(Lon ~ X + Sur + Oeste, data = puntos_identificados)
summary(model_lon)
summary(model_lat)

df$lat <- predict.lm(model_lat, df)
df$lon <- predict.lm(model_lon, df)

# Para ver el ajuste sobre un mapa
leaflet(data = df) %>%
  addTiles(urlTemplate = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png') %>%
  addCircles(~lon, ~lat, radius = 0.2, fillOpacity = 0.03)

leaflet(data = puntos_identificados) %>%
  addTiles(urlTemplate = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png') %>%
  addCircles(~Lon, ~Lat, radius = 0.2, fillOpacity = 0.03, label = ~as.character(Lugar))

df$Sur = NULL
df$Oeste = NULL

# Calculando la Distancia a Sol

sol = c(lat = 40.418460, lon = -3.706529)

df$dist_eucl_sol = sqrt((df$lat - sol['lat'])**2 + (df$lon - sol['lon'])**2)
df$dist_taxi_sol = (df$lat - sol['lat']) + (df$lon - sol['lon'])

write.csv(df, 'dataset_train.csv')

# Interpolando para Estimar

df_test = read.csv('Estimar_UH2020.txt', sep='|')
df_test$Sur <- ifelse(165724537<df_test$Y, 1,0)
df_test$Oeste <- ifelse(2208137136<df_test$X, 1, 0)
df_test$lat = predict.lm(model_lat, df_test)
df_test$lon = predict.lm(model_lon, df_test)
df_test$Sur = NULL
df_test$Oeste = NULL
df_test$dist_eucl_sol = sqrt((df_test$lat - sol['lat'])**2 + (df_test$lon - sol['lon'])**2)
df_test$dist_taxi_sol = (df_test$lat - sol['lat']) + (df_test$lon - sol['lon'])
write.csv(df_test, 'dataset_test.csv')

