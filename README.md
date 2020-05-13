# SentinelAdmins

Código y programas utilizados para ganar el Cajamar UniversityHack 2020, reto Minsait Land Classification. En AFI_SentinelAdmins.pdf, pueden encontrar un resumen del proceso, así como una explicación breve de los programas, que puede ayudar de guía para entender el código. 


## Uso Típico: Sacar predicciones de test sin re-entrenar modelos

### Ejecución en Linux

Para usar el código: crear entorno y activarlo.
```bash
conda env create -f environment.yaml
conda activate sentinel_env
```
Para desactivar el entorno:
```bash
conda deactivate sentinel_env
```

Desde el terminal, ejecutar:

1. Para obtener la interpolación de la latitud y longitud, además de variables como distancia a Sol, euclídea y en taxi (Manhattan).

```bash
Rscript 01-InterpolacionLatLon.R
```

2. Para obtener las variables geográficas:

```bash
python 02-programa_geovars.py
```

Dependiendo de la máquina, este proceso tardará un tiempo. Recomendamos, en caso de necesitar dejar este programa corriendo en background, ejecutarlo con el siguiente formato:

```bash
nohup python -u 02-programa_geovars.py > log_programa_geovars.txt &
```

Estos dos programas nos facilitan el proceso que se puede ver en el siguiente diagrama:

![Feature Generation](feature_generation.png)

3. Para sacar las predicciones sobre el fichero de test:

```bash
python -u 03-sacar_test_stacking.py
```

En este tercer paso va incluido, "under the hood", el proceso de feature engineering, que se puede apreciar
![en la siguiente imagen](feature_engineering.png)


## Jugar con el código: entrenar otros modelos, hacer experimentos o pruebas, etc.

En caso de que se quieran entrenar otros modelos o realizar pruebas, disponemos de muchos scripts para hacerlo. Imaginemos por ejemplo que queremos ver qué tal lo hace una de las arquitecturas de Stacking que tenemos en models.py, por ejemplo StackingAlex1. En tal caso, el comando de entrenamiento sería el siguiente.  

```bash
python -u train_stacking.py --model StackingAlex1 --name Experiment_01 --final_model LGBM --cv 3 --encoder CatBoost
```

Para entrenar modelos simples, y no de stacking, uno puede usar model_trainer_refactor.py, de la siguiente forma:

```bash
python -u model_trainer_refactor.py --model LGBM --name LGBM_01 --use_old False --iter 100
```

Con estos programas se puede emular el sistema de modelización final que está resumido ![en esta figura](modelizacion.png)
