from sklearn.ensemble import RandomForestClassifier
from preprocessing import *
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import mlflow
import mlflow.sklearn
import mlflow.tracking
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.utils import class_weight
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from imblearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

NAME = 'lightgbm_vars_armando_1803'
N_ITER = 100
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)


class UnderSampling(TransformerMixin):
    """
    This way we can perform parametric search of this also!!! 
    """
    def __init__(self, llave, perc=None):
        super(UnderSampling).__init__()
        self.perc = perc
        self.llave = llave
        
    def fit(self, X, y=None):
        counter = Counter(y)
        self.under = RandomUnderSampler(sampling_strategy={k:int(v*self.perc) for k, v in dict(counter).items()
                               if k == self.llave})
        self.under.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        return self.under.transform(X, y)
    

class OverSampling(TransformerMixin):
    """
    This way we perform parametric seach of this also!!!
    """
    def __init__(self, llave, n_0=None, n_1=None, n_2=None, n_3=None, n_4=None, n_5=None):
        super(OverSampling).__init__()
        self.ns = [n_0, n_1, n_2, n_3, n_4, n_5]
        self.llave=llave
    
    def fit(self, X, y=None):
        counter=Counter(y)
        self.dic_smote = {k:int(v*n) for k, v, n in zip(dict(counter).items(), self.ns)
                          if k != llave}
        self.over = SMOTE(sampling_strategy=self.dic_smote)
        self.over.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        return self.over.transform(X, y)


'''
params = {'n_estimators': (100, 1500),
            'max_depth': (6, 12),
            'class_weight': [None, 'balanced'],
            'n_jobs': [-1],
            'max_features': ['auto', 'log2'],
            }
'''


params = {'model__reg_alpha': (1e-3, 5.0, 'log-uniform'),
          'model__reg_lambda': (1e-2, 50.0, 'log-uniform'),
          'model__n_estimators': (600, 3000),
          'model__learning_rate': (5e-4, 1.0, 'log-uniform'),
          'model__num_leaves': (25, 60),
          'model__boosting_type': ['gbdt', 'goss'],
          'model__colsample_bytree': (0.2, 1.0, 'uniform'),
          'model__subsample': (0.5, 1.0, 'uniform'),
          'model__min_child_samples': (1, 25),
          'model__min_child_weight': (1e-6, 0.01, 'log-uniform'),}




'''
'under__perc': (0.3, 1.0, 'uniform'),
'over__n0': (1, 15),
'over__n1': (1, 15),
'over__n2': (1, 15),
'over__n3': (1, 15),
'over__n4': (1, 15),
'over__n5': (1, 15)}
'''

'''
params = {
        'model__learning_rate': (0.0001, 1.0, 'log-uniform'),
        'model__min_child_weight': (0, 10),
        'model__max_depth': (2, 12),
        #'max_delta_step': (1, 20),
        'model__subsample': (0.01, 1.0, 'uniform'),
        'model__colsample_bytree': (0.01, 1.0, 'uniform'),
        'model__colsample_bylevel': (0.01, 1.0, 'uniform'),
        'model__reg_lambda': (1e-2, 50., 'log-uniform'),
        'model__reg_alpha': (1e-3, 10.0, 'log-uniform'),
        'model__gamma': (1e-9, 1.0, 'log-uniform'),
        'model__n_estimators': (100, 1600),
        'model__scale_pos_weight': (1e-6, 100., 'log-uniform')
    }
'''

'''
params = {
        'depth':(6,15),
        'iterations': (500, 1600),
        #'learning_rate': (1e-7, 1e-1),
        'reg_lambda': (1e-5, 10.0),
        #'l2_leaf_reg':(0.1, 100.),
        'bagging_temperature':(1e-8, 1., 'log-uniform'),
        'border_count':(1,255),
        #'rsm':(0.10, 0.8, 'uniform'),
        'random_strength':(1e-3, 3.0, 'log-uniform'),
    }
'''



#model = RandomForestClassifier()
#model = XGBClassifier(eval_metric = 'auc', n_jobs=-1, objective='multi:softmax', num_class=7)
#model = CatBoostClassifier(silent=True, objective='multi:softmax')

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


def get_classes_order_catboost(X_train, y_train):
    cat = CatBoostClassifier(iterations=10, depth=2, learning_rate=0.05, loss_function='MultiClass')
    cat.fit(X_train, y_train)
    return cat.classes_

def main():
    mlflow.start_run(run_name=NAME)
    print('procesando los datos')
    X, y, encoder = preprocess_data('new_total_train.csv', process_cat=True)
    #tag2idx = {k: v for k, v in sorted(tag2idx.items(), key=lambda item: item[1])}
    #labs_names = [k for k in tag2idx.keys()]
    with open(f"label_encoder_{NAME}.pkl", "wb") as f:
        pickle.dump(encoder, f)
    print(f"##################### The shape of X is {X.shape} #######################")
    y=y.astype('int')
    print('######### Eliminando columnas que antes no estaban ###########')
    #X.drop(['distancias_al_ruido' , 'contadores_pension1.shp', 'contadores_host3.shp', 'contadores_hoteles4.shp', 'contadores_host1.shp', 'contadores_campus.shp'], axis=1, inplace=True)
    if 'X_train.pkl' not in os.listdir():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=15, stratify=y)

    else:
        with open('X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open('X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open('y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open('y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)
    
    X = X.astype('float')
    print(X.shape)
    print(X_train.shape)
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
    
    #print(dic_smote)
    #dic_smote[tag2idx['OFFICE']] = int(dic_smote[tag2idx['OFFICE']]*1.7)
    
    over = SMOTE(sampling_strategy=dic_smote)
    
    under = RandomUnderSampler(sampling_strategy={k:int(v*0.95*(2/3)) for k, v in dict(counter).items() if k == llave})
    
    #under = UnderSampling(llave=llave)
    #over = OverSampling(llave=llave)
    
    with open('X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

    #setlabs = [l for l in set(y_train)]
    #tag2idx = {i: l for l, i in enumerate(setlabs)}
    '''
    print(f"tag2idx is {tag2idx}")
    with open(f"tag2idx_{NAME}.pkl", "wb") as f:
        pickle.dump(tag2idx, f)
    '''
    '''
    pipe_imb = Pipeline([('o', over), ('u', under)])
    X_train_resam = pd.get_dummies(X_train, columns = X_train.columns[categoricas])
    X_resam, y_resam = pipe_imb.fit_resample(X_train_resam, y_train)
    '''
    '''
    cw = list(class_weight.compute_class_weight('balanced',
                                             get_classes_order_catboost(X_train, y_train),
                                             y_train))
    '''
    #print(f"Las features categoricas son {categoricas}, con dtypes {X_train.dtypes[categoricas]}")
    #model = CatBoostClassifier(silent=True, loss_function='MultiClass', cat_features=categoricas, class_weights=cw, boosting_type='Plain', max_ctr_complexity=2,  thread_count=-1) #, task_type="GPU", devices='0:1')
    
    model = LGBMClassifier(class_weight='balanced', objective='multiclass:softmax', n_jobs=-1, random_state=100)
    #model = XGBClassifier(class_weight='balanced', objective='multiclass:softmax', n_jobs=-1, random_state=100)
    steps = [('over', over), ('under', under), ('model', model)] # ('o', over), 
    pipeline = Pipeline(steps)
    
    '''
    with open('best_lightgbm_geovars_10_03_params.pkl', 'rb') as f:
        params_probar_primero = pickle.load(f)
    
    nuevo_dic = {}
    
    for k in params_probar_primero.keys():
        k_ = k.replace('model__', '')
        nuevo_dic[k_] = params_probar_primero[k]
    
    pipeline_prueba = Pipeline([('o', over), 
                       ('u', under), 
                       ('model', LGBMClassifier(class_weight='balanced', objective='multiclass:softmax', n_jobs=-1, **nuevo_dic))])
    
    pipeline_prueba.fit(X_train, y_train)
    preds_pipeline_prueba = pipeline_prueba.predict(X_test)
    print(f"Resultado pipeline prueba: {f1_score(y_test, preds_pipeline_prueba, average='macro')}")
    #print(f"Score is {pipeline_prueba.score(y_test, X_test)}")
    '''
    best_model = BayesSearchCV(
                pipeline,
                params,
                n_iter = N_ITER,
                n_points=1,
                cv=cv,
                scoring='f1_macro',
                random_state=100,
                optimizer_kwargs= {'n_initial_points': 20})

    def on_step(optim_result):
        score = best_model.best_score_
        results = best_model.cv_results_
        #preds = best_model.predict(X_test)
        try:
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'results_{NAME}.csv', header=True, index=False)
            print(f'############ Llevamos {results_df.shape[0]} pruebas #################')
            print(f'los resultados del cv de momento son {results_df}')
        except:
            print('Unable to convert cv results to pandas dataframe')
        mlflow.log_metric('best_score', score)
        with open(f'./best_{NAME}_params.pkl', 'wb') as f:
            pickle.dump(best_model.best_params_, f)
        with open(f'./totalbs_{NAME}_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        print("best score: %s" % score)
        if score >= 0.98:
            print('Interrupting!')
            return True

    #print(f'Los nombres de los features son {X.columns}')
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
    best_model.fit(X_train, y_train, callback=[on_step])
    with open(f'./best_{NAME}_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print('loggeando movidas')
    preds = best_model.predict(X_test)
    preds = encoder.inverse_transform(preds)
    y_test = encoder.inverse_transform(y_test)
    #print(f"El score ha sido de {best_model.score(X_test, y_test)}")
    #mlflow.log_artifact(f'./best_{NAME}_model.pkl')
    mlflow.log_metrics(metrics={'f1': f1_score(y_test, preds, average='macro'),
                           'precision': precision_score(y_test, preds, average='macro'),
                           'recall': recall_score(y_test, preds, average='macro'),
                           'accuracy': accuracy_score(y_test, preds)})
    best_params = best_model.best_params_
    for param in best_params.keys():
         mlflow.log_param(param, best_params[param])
    preds = best_model.predict(X_test)
    preds_proba = best_model.predict_proba(X_test)
    cm = confusion_matrix(y_test, preds)
    grafico_conf_matrix = print_confusion_matrix(cm, class_names = labs_names)
    grafico_conf_matrix.savefig(NAME)
    grafico_norm = print_confusion_matrix(cm, class_names = labs_names, normalize=False)
    grafico_norm.savefig(f'{NAME}_no_norm')
    mlflow.log_artifact(f'./{NAME}.png')
    mlflow.log_artifact(f'./{NAME}_no_norm.png')
    #mlflow.sklearn.log_model(best_model.best_estimator_, 'random_forest_model')
    mlflow.end_run()


if __name__ == '__main__':
    main()
