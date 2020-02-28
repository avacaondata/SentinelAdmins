import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

##### variables que se distribuyen "raro": 
# 3 , 13, 14 (rarísimo), 15 (casi normal pero algo asimétrico),
# 23 (casi normal pero algo asimétrico), 24, 25,
# 26 (asimétrica a tope), 27, 28, 30 (muy centrada, aunque así hay más)
# 31 (parecida a 30), 32 (igual q 31, 30), 33 (asimétrica con algunos outliers)
# 34 (parecida a 33), 35 (bastante asimétrica con outliers claros), 
# 36 (solo tiene valores en una franja concreta a excepción de unos pocos outliers)
# 38 (mucha asimetría), 39 (igual), 40 (igual), 41 (igual), 42 (igual), 43 (igual)
# 44 (igual), 45 (igual), 46 (rarísima, échale un ojo tú mismo),  
# 47 (casi todos los valores cerca del 0, algunos valores a tomar x culo)
# 49 (parecida a 47, pero el rango es menor), 50 (igual q 49)
# 51 (igual que 49 y 50 pero algo mayor el rango), 
# 52 (claramente entre los años 50 y 2000 se construyen la mayoría de las casas)
# 53 (me parece claramente categórica, dime tu como lo ves )
# 54 (claramente categórica; las clases B, A y C poco presentes)
# OBJ: 55 (una asimetría brutal la verdad..., pillar x ejemplo la clase agriculture va a ser muy complicado).

categorical = [53, 54]

minmax = MinMaxScaler()
stdscaler = StandardScaler()

def preprocess_data(f, scale=True, scaler = 'std', process_cat = False, y_name='CLASE'):
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
    
    Returns
    -------------
    X
        The matrix with features
    y
        The vector with objective variable
    """
    df = pd.read_csv(f, sep='|')
    y = df['CLASE'].values
    X = df.drop(['CLASE', 'id'], axis = 1)
    if process_cat:
        X = pd.get_dummies(X, columns = df.columns[categorical])
    X = np.array(X)
    select_columns = [i for i in range(X.shape[1]) if i != categorical]
    # en caso de que lo veas bien, mete aquí transformaciones del tipo X[:, var] = np.log1p(X[:, var]),
    # antes de escalar
    if scale:
        if scaler == 'std':
            X[:, select_columns] = stdscaler.fit_transform(X[:, select_columns])
        elif scaler == 'minmax':
            X[:, select_columns] = minmax.fit_transform(X[:, select_columns])
    
    return X, y
