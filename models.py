from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.base import TransformerMixin


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
                          if k != self.llave}
        self.over = SMOTE(sampling_strategy=self.dic_smote)
        self.over.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        return self.over.transform(X, y)


models_dic =  {'lgbm': 
                {'model':  LGBMClassifier(class_weight='balanced', objective='multiclass:softmax', 
                                          n_jobs=-1, random_state=100, silent=True),
                 'parameters': {'reg_alpha': (1e-3, 5.0, 'log-uniform'),
                                    'reg_lambda': (1e-2, 50.0, 'log-uniform'),
                                    'n_estimators': (600, 3000),
                                    'learning_rate': (5e-4, 1.0, 'log-uniform'),
                                    'num_leaves': (25, 60),
                                    'boosting_type': ['gbdt', 'goss'],
                                    'colsample_bytree': (0.2, 1.0, 'uniform'),
                                    'subsample': (0.5, 1.0, 'uniform'),
                                    'min_child_samples': (1, 25),
                                    'min_child_weight': (1e-6, 0.01, 'log-uniform')}}
                    ,
                'catboost': {'model':  CatBoostClassifier(silent=True, loss_function='MultiClass', 
                                                          cat_features=None, class_weights=None, boosting_type='Plain', 
                                                          max_ctr_complexity=2,  thread_count=-1),
                                                          #, task_type="GPU", devices='0:1')}
                             'parameters': {
                                            'depth':(6,15),
                                            'iterations': (500, 1600),
                                            'learning_rate': (1e-7, 1e-1),
                                            'reg_lambda': (1e-5, 10.0),
                                            'l2_leaf_reg':(0.1, 100.),
                                            'bagging_temperature':(1e-8, 1., 'log-uniform'),
                                            'border_count':(1,255),
                                            'rsm':(0.10, 0.8, 'uniform'),
                                            'random_strength':(1e-3, 3.0, 'log-uniform'),
                                        },
                             'additional_necessary_params': ['class_weight', 'cat_features']},

                'random_forest' : {'model': RandomForestClassifier(n_jobs=-1, class_weight='balanced'),
                                   'parameters': {'n_estimators': (100, 1500),
                                                  'max_depth': (6, 12),
                                                  'class_weight': [None, 'balanced'],
                                                  'n_jobs': [-1],
                                                  'max_features': ['auto', 'log2'],
                                                }},
                
                'xgboost': {'model': XGBClassifier(eval_metric = 'f1_macro', n_jobs=-1, objective='multi:softmax', num_class=7),
                            'parameters': {
                                            'learning_rate': (0.0001, 1.0, 'log-uniform'),
                                            'min_child_weight': (0, 10),
                                            'max_depth': (2, 12),
                                            #lta_step': (1, 20),
                                            'subsample': (0.01, 1.0, 'uniform'),
                                            'colsample_bytree': (0.01, 1.0, 'uniform'),
                                            'colsample_bylevel': (0.01, 1.0, 'uniform'),
                                            'reg_lambda': (1e-2, 50., 'log-uniform'),
                                            'reg_alpha': (1e-3, 10.0, 'log-uniform'),
                                            'gamma': (1e-9, 1.0, 'log-uniform'),
                                            'n_estimators': (100, 1600),
                                            'scale_pos_weight': (1e-6, 100., 'log-uniform')
                                        }},
                                    
                'keras_classifier': {}}