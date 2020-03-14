setwd("D:/Armando/Documents/Cajamar2020/models_armando")
library(data.table)
library(tidyverse)

preprocess <- function(d, balance = FALSE){
  
  d <- d[complete.cases(d)]
  
  # Convierto CadastralQuality a numérico de menor a mayor calidad.
  d$CADASTRALQUALITYID <- recode(d$CADASTRALQUALITYID,
                                 `9`=1,
                                 `8`=2,
                                 `7`=3,
                                 `6`=4,
                                 `5`=5,
                                 `4`=6,
                                 `3`=7,
                                 `2`=8,
                                 `1`=9,
                                 `C`=10,
                                 `B`=11,
                                 `A`=12,
                                 .missing = 6,
                                 .default = 6)
  
  # Imputo MaxBuildingFloor con la media (cambiar a algo mejor)
  d$MAXBUILDINGFLOOR <- replace_na(d$MAXBUILDINGFLOOR, mean(d$MAXBUILDINGFLOOR, na.rm = T))
  
  # Cambio el nivel de ruido a numérica
  d$ruido <- recode(d$ruido, Alta = 3, Moderada = 2, Baja = 1, No_Reg_Cerca = 2)
  
  # Elimino X e Y por ser colineales con longitud y latitud
  d$X <- NULL
  d$Y <- NULL
  
  ##### DownSampling #####
  if(balance == TRUE){
    d_residential <- d[d$CLASE=="RESIDENTIAL",]
    set.seed(42)
    sample_residential <- sample(1:nrow(d_residential), size = ceiling(nrow(d)*0.2))
    d_balanced <- union_all(d[d$CLASE!="RESIDENTIAL",],d_residential[sample_residential])
    
    return(d_balanced)
  }
  else{
    return(d)
  }
}

# Dataset de train balanceado
fread('TOTAL_TRAIN.csv') %>%
  preprocess(balance = TRUE) %>%
  fwrite("BALANCED_TRAIN.csv")

# Dataset de test preprocesado
fread('TOTAL_TEST.csv') %>%
  preprocess(balance = FALSE) %>%
  fwrite("FINAL_TEST.csv")

