## interpolacion

library(leaflet)
library(tidyverse)

df = read.csv('Modelar_UH2020.txt', sep='|')

puntos_identificados <- data.frame(
  Lugar = c("Retiro","Legazpi", "Plaza de Toros","Arguelles", 
            "Cuatro Caminos", "Zapateria Robledo","Tetuan", "Puerta Alcalá", "Puente M-30", "Centro", 
            "Puente Segovia", "Principe Pío", "Templo de Debod", "Tetas de Vallecas", "Nuevos Ministerios"),
  X = c(2209489320, 2209747825, 2219461192, 2196313578, 2201844156, 2196736748, 2203891600, 2209352526, 
        2218209219, 2205043708, 2194249147, 2193287504, 2196048999, 2220668045, 2205811289),
  Y = c(165537783, 165397664, 165616041, 165611443, 165677624, 165795600, 165734820, 165570333, 165558640, 
        165564480, 165537870, 165577124, 165591244, 165472795, 165644994),
  Lat = c(40.413588, 40.386163, 40.432474, 40.430805, 40.447087, 40.475766, 40.460913, 40.420477, 
          40.419595, 40.418460, 40.413867, 40.421123, 40.424126, 40.398253, 40.442252),
  Lon = c(-3.683393, -3.680577, -3.663236, -3.716191, -3.703295, -3.71556, -3.698528, -3.689503, 
          -3.664541, -3.706529, -3.721971, -3.720090, -3.717863, -3.657689, -3.692191)
)

model_lat <- lm(Lat ~ poly(Y, 1), data = puntos_identificados) # El que mejor resultado saca es el poly grado 1
summary(model_lat)

model_lon <- lm(Lon ~ poly(X, 1), data = puntos_identificados)
summary(model_lon)

df$lat <- predict.lm(model_lat, df)
df$lon <- predict.lm(model_lon, df)

# Mapa para visualizar

leaflet(data = df) %>%
  addTiles(urlTemplate = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png') %>%
  addCircles(~lon, ~lat, radius = 0.3, fillOpacity = 0.03)


# ### Probando una regresion de mayor grado
# model_lat <- lm(Lat ~ poly(Y, 2), data = puntos_identificados)
# summary(model_lat)
# 
# model_lon <- lm(Lon ~ poly(X, 2), data = puntos_identificados)
# summary(model_lon)
# 
# df$lat <- predict.lm(model_lat, df)
# df$lon <- predict.lm(model_lon, df)
# 
# leaflet(data = df) %>%
#   addTiles(urlTemplate = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png') %>%
#   addCircles(~lon, ~lat, radius = 0.3, fillOpacity = 0.05)
# 
# # no sirve :p
# # La mejor es la lineal
# 
# # Viendo si se mejora la precisión contando las filas
# puntos_identificados$Fila1 <- ifelse(puntos_identificados$Y>165898546, 1,0)
# puntos_identificados$Fila2 <- ifelse(165724537<puntos_identificados$Y & puntos_identificados$Y<=165898546, 1,0)
# puntos_identificados$Fila3 <- ifelse(165556724<puntos_identificados$Y & puntos_identificados$Y<=165724537, 1,0)
# puntos_identificados$Fila4 <- ifelse(puntos_identificados$Y<=165556724, 1,0)
# 
# 
# df$Fila1 <- ifelse(df$Y>165898546, 1,0)
# df$Fila2 <- ifelse(165724537<df$Y & df$Y<=165898546, 1,0)
# df$Fila3 <- ifelse(165556724<df$Y & df$Y<=165724537, 1,0)
# df$Fila4 <- ifelse(df$Y<=165556724, 1,0)
# 
# model_lat <- lm(Lat ~ Y + Fila1 + Fila2 + Fila3 + Fila4, data = puntos_identificados)
# summary(model_lat)
# df$lat <- predict.lm(model_lat, df)
# leaflet(data = df) %>%
#   addTiles(urlTemplate = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png') %>%
#   addCircles(~lon, ~lat, radius = 0.3, fillOpacity = 0.03)
# 
# 


# # Esto mejora un poquitín el mapa, pero hay que ver si vale la pena
puntos_identificados$Sur <- ifelse(165724537<puntos_identificados$Y, 1,0)

df$Sur <- ifelse(165724537<df$Y, 1,0)

model_lat <- lm(Lat ~ Y + Sur, data = puntos_identificados)
model_lon <- lm(Lon ~ X + Sur, data = puntos_identificados)
summary(model_lon)
summary(model_lat)
df$lat <- predict.lm(model_lat, df)
df$lon <- predict.lm(model_lon, df)
leaflet(data = df) %>%
  addTiles(urlTemplate = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png') %>%
  addCircles(~lon, ~lat, radius = 0.2, fillOpacity = 0.03)


df$Sur = NULL

write.csv(df, 'dataset_train.csv')

df_test = read.csv('Estimar_UH2020.txt', sep='|')
df_test$Sur <- ifelse(165724537<df_test$Y, 1,0)
df_test$lat = predict.lm(model_lat, df_test)
df_test$lon = predict.lm(model_lon, df_test)
df_test$Sur = NULL
write.csv(df_test, 'dataset_test.csv')

