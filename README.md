# SentinelAdmins

Código y programas utilizados para llegar a la final de Cajamar UniversityHack 2020, reto Minsait Land Classification. En AFI_SentinelAdmins.pdf, pueden encontrar un resumen del proceso, así como una explicación breve de los programas, que puede ayudar de guía para entender el código. 

Para usar el código:
```bash
conda env create -f environment.yaml
```
Desde el terminal, ejecutar:

1. Para obtener la interpolación de la latitud y longitud, además de variables como distancia a Sol, euclídea y en taxi (Manhattan).

```bash
Rscript InterpolacionLatLon01.R
```

2. Para obtener las variables geográficas:

```bash
python programa_geovars.py
```

Dependiendo de la máquina tardará un tiempo u otro, pues está programado para utilizar todos los cores disponibles, por lo que de disponer de pocas cpus el tiempo de cálculo será mayor. En caso de necesitar dejar este programa corriendo en background, ejecutar:

```bash
nohup python -u programa_geovars.py > log_programa_geovars.txt &
```

3. Para sacar las predicciones sobre el fichero de test:

```bash
nohup python -u programa_geovars.py > log_programa_geovars.txt &
```
