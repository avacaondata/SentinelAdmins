# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 23:39:46 2020

@author: ArmandoPC
"""

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from skopt import BayesSearchCV
from sklearn.metrics import f1_score, make_scorer, fbeta_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle
from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from preprocessing import *
import mlflow
import mlflow.sklearn
import mlflow.tracking

# NUEVAS
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import class_weight

NAME = 'mlp'
N_ITER = 2
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)
MODE = 'BAYES'
my_f1 = make_scorer(f1_score, average='macro')

""" =================== FUNCIONES =================== """

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, normalize=True):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    fmt = '.2f' if normalize else 'd'
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def main():
    if 'X_train.pkl' not in os.listdir():
        print('procesando los datos')
        X, y, encoder = preprocess_data('TOTAL_TRAIN.csv', process_cat=True)
        X = X.astype('float')
        print(X.shape)
        
        with open('X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        print(f"##################### The shape of X is {X.shape} #######################")
        y=y.astype('int')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=15, stratify=y)
        print(X_train.shape)
        with open(f'X_train.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open(f'X_test.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open(f'y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open(f'y_test.pkl', 'wb') as f:
            pickle.dump(y_test, f)
            
    else:
        with open('X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open('X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open('y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open('y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)
        with open(f'label_encoder_{NAME}.pkl', 'rb') as f:
            encoder = pickle.load(f)
        with open(f'{NAME}_params.pkl', 'rb') as f:
            params = pickle.load(f)
    labs_names = [c for c in encoder.classes_]

    counter = Counter(y_train)
    maximo = 0
    for k, v in dict(counter).items():
        if v > maximo:
            maximo = v
            llave = k
        else:
            continue

    dic_smote = {k:int(v*10*(2/3)) for k, v in dict(counter).items()
                                    if k != llave}
        
    over = SMOTE(sampling_strategy=dic_smote)

    under = RandomUnderSampler(sampling_strategy={k:int(v*0.95*(2/3)) for k, v in dict(counter).items() if k == llave})

    class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(y_train),
                                                    y_train)
    class_weights = dict(enumerate(class_weights))                                                 

    def create_mlp_model(n_features, layers, layer_units,
                        dropout_rate, activation_func, optimizer_strat,
                        use_batch_normalization, kernel_init):
        model = Sequential()
        for i in range(layers):
            if(i==1):
                model.add(Dense(layer_units, input_shape = (n_features,), activation=activation_func, kernel_initializer=kernel_init))
            else:
                model.add(Dense(layer_units, activation = activation_func, kernel_initializer=kernel_init))
            model.add(Dropout(dropout_rate))
            if(use_batch_normalization == True):
                model.add(BatchNormalization())
        # Final layer
        model.add(Dense(7, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizer_strat,
                    metrics = ['accuracy'])
        return model

    grid_params = {
        ### create_mlp_model params ###
        'n_features':[X_train.shape[1]],
        'layers': (2,6),
        'layer_units': (64, 128, 'log-uniform'),
        'dropout_rate': (0.0, 0.5),
        'activation_func': ['relu', 'tanh', LeakyReLU()],
        'optimizer_strat':['adam'],
        'use_batch_normalization':[True, False],
        'kernel_init':['glorot_uniform', 'glorot_normal'],
        'epochs':(5, 8),
        'batch_size':(32, 512, 'log-uniform'),
    }
    fit_params = {
        ### fit params ###
        'validation_split':0.2,
        'class_weight': class_weights,
        #'workers':10,
    }

    model = KerasClassifier(create_mlp_model, verbose = 2)
    steps = [('under', under), ('model', model)]
    pipeline = Pipeline(steps)
    
    mlflow.start_run(run_name=NAME)

    best_model = BayesSearchCV(
                        model,
                        grid_params,
                        n_iter = N_ITER,
                        n_points=1,
                        cv=cv,
                        scoring=my_f1,
                        random_state=100,
                        optimizer_kwargs= {'n_initial_points': 20},
                        n_jobs=1,
                        fit_params=fit_params,
                        verbose=1)

    def on_step(optim_result):
        score = best_model.best_score_
        results = best_model.cv_results_
        try:
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'results_{NAME}.csv', header=True, index=False)
            print(f'############ Llevamos {results_df.shape[0]} pruebas #################')
            print(f'los resultados del cv de momento son \n {results_df}')
        except:
            print('Unable to convert cv results to pandas dataframe')
        mlflow.log_metric('best_score', score)
        with open(f'./best_{NAME}_params.pkl', 'wb') as f:
            pickle.dump(best_model.best_params_, f)
        # with open(f'./totalbs_{NAME}_model.pkl', 'wb') as f:
        #     pickle.dump(best_model, f)
        print("best score: %s" % score)
        if score >= 0.98:
            print('Interrupting!')
            return True

    good_colnames = []
    i = 0
    for col in X_train.columns:
        if not col.isascii():
            print(f'La columna {col} no es ascii')
            good_colnames.append(f'{i}_not_ascii')
            i+=1
        else:
            good_colnames.append(col)
    X_train.columns = good_colnames
    print('ajustando modelo')
    if MODE != 'INDIVIDUAL':
        best_model.fit(X_train.values, y_train, callback=[on_step])
        # with open(f'./best_{NAME}_model.pkl', 'wb') as f:
        #     pickle.dump(best_model, f)
        model_json = best_model.best_estimator_.to_json()
        with open("best_model_Keras.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        best_model.best_estimator_.save_weights("model.h5")
        print("Saved model to disk")
        preds = best_model.predict(X_test)
    else:
        params = {k.replace('model__', ''):v for k, v in params.items()}
        best_model = KerasClassifier(create_mlp_model, verbose = 2)
        
        print('without pipeline:')
        best_model.fit(X_train, y_train)
        with open(f'./best_{NAME}_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        preds = best_model.predict(X_test)
        print(f'F1 SCORE IS {f1_score(y_test, preds, average="macro")}')

        print('with pipeline:')
        steps = [('over', over), ('under', under)] # ('o', over), 
        pipeline = Pipeline(steps)
        X_train, y_train = pipeline.fit_resample(X_train, y_train)
        best_model.fit(X_train, y_train)
        with open(f'./best_{NAME}_model_with_pipeline.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        preds = best_model.predict(X_test)
        print(f'F1 SCORE WITH PIPELINE IS {f1_score(y_test, preds, average="macro")}')
    print('loggeando movidas')
    mlflow.log_metrics(metrics={'f1': f1_score(y_test, preds, average='macro'),
                            'precision': precision_score(y_test, preds, average='macro'),
                            'recall': recall_score(y_test, preds, average='macro'),
                            'accuracy': accuracy_score(y_test, preds),
                            'f05': fbeta_score(y_test, preds, beta=0.5, average='macro'),
                            'f2': fbeta_score(y_test, preds, beta=2, average='macro')})
    if MODE != 'INDIVIDUAL':
        best_params = best_model.best_params_
        for param in best_params.keys():
            mlflow.log_param(param, best_params[param])
    cm = confusion_matrix(y_test, preds)
    grafico_conf_matrix = print_confusion_matrix(cm, class_names = labs_names)
    grafico_conf_matrix.savefig(NAME)
    grafico_norm = print_confusion_matrix(cm, class_names = labs_names, normalize=False)
    grafico_norm.savefig(f'{NAME}_no_norm')
    #mlflow.log_artifact(f'./{NAME}.png')
    #mlflow.log_artifact(f'./{NAME}_no_norm.png')
    mlflow.end_run()

if __name__ == '__main__':
    main()

##### Pruebas

""" 
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(128, activation=LeakyReLU(), kernel_initializer='truncated_normal'))
model.add(Dropout(0.15))
model.add(Dense(7, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

history_mlp = model.fit(X_train, y_train,
                          epochs=10,
                          batch_size=256,
                          validation_split = 0.20,
                          verbose=2)

preds = [np.argmax(list(l)) for l in model.predict(X_test)]

f1_score(y_test, preds, average='macro') 
"""
