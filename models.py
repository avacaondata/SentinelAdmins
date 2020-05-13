##########################################
### AUTHOR: ALEJANDRO VACA SERRANO #######
##########################################


from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
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
from model_trainer import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer, fbeta_score
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
import imblearn as imb
import sklearn as skl


f2_scorer = make_scorer(fbeta_score, beta=2, average="macro")


class UnderSampling(TransformerMixin):
    """
    This way we can perform parametric search of this also!!! 
    With this class, implemented as a sklearn's transformer,
    aims to find the optimal percentage of undersampling
    for the majority class. As RandomUnderSampler() needs a dictionary
    as the sampling strategy for only undersampling one of the classes,
    we need to implement this wrapper to find this optimum undersampling.

    Parameters
    ------------
    llave: int
        The key of the majority class when codified as number 
        (with LabelEncoder for example)
    perc: float
        Float from 0 to 1 indicating the percentage of the class samples
        to leave on the final sample. This is the parameter we want to optimize
        thorugh Bayesian Optimization
    """

    def __init__(self, llave, perc=None):
        super(UnderSampling).__init__()
        self.perc = perc
        self.llave = llave

    def fit(self, X, y=None):
        counter = Counter(y)
        self.under = RandomUnderSampler(
            sampling_strategy={
                k: int(v * self.perc)
                for k, v in dict(counter).items()
                if k == self.llave
            }
        )
        return self.under.fit_resample(X, y)

    def transform(self, X, y=None):
        if y is not None:
            return X, y
        else:
            return X


class OverSampling(TransformerMixin):
    """
    This way we perform parametric seach of this also!!!
    This one was not used in the end, as with the most sophisticated oversampling
    methods from imblearn we were getting worse results.
    """

    def __init__(
        self, llave, n_0=None, n_1=None, n_2=None, n_3=None, n_4=None, n_5=None
    ):
        super(OverSampling).__init__()
        self.ns = [n_0, n_1, n_2, n_3, n_4, n_5]
        self.llave = llave

    def fit(self, X, y=None):
        counter = Counter(y)
        self.dic_smote = {
            k: int(v * n)
            for k, v, n in zip(dict(counter).items(), self.ns)
            if k != self.llave
        }
        self.over = SMOTE(sampling_strategy=self.dic_smote)
        return self.over.fit_resample(X, y)

    def transform(self, X, y=None):
        if y is not None:
            return X, y
        else:
            return X



def lgb_f1_score(y_true, y_pred):
    """
    Callback for LGBM.
    """
    preds = y_pred.reshape(len(np.unique(y_true)), -1)
    preds = preds.argmax(axis=0)
    return "f2", fbeta_score(y_true, preds, beta=2, average="macro"), True


# NOTE: DIC WITH SOME OF THE TRIED MODELS (ONLY THE ONE THAT MADE SENSE TO KEEP)

models_dic = {
    "lgbm": {
        "model": imb.pipeline.Pipeline(
            steps=[
                (
                    "under",
                    RandomUnderSampler(sampling_strategy={5: int(0.11 * 76647)}),
                ),
                (
                    "model",
                    LGBMClassifier(
                        class_weight="balanced",
                        objective="multiclass:softmax",
                        n_jobs=-1,
                        random_state=100,
                        silent=True,
                    ),
                ),
            ]
        ),
        "parameters": {
            "model__reg_alpha": (1e-3, 5.0, "log-uniform"),
            "model__reg_lambda": (1e-2, 50.0, "log-uniform"),
            "model__n_estimators": (600, 3000),
            "model__learning_rate": (5e-4, 1.0, "log-uniform"),
            "model__num_leaves": (25, 60),
            "model__boosting_type": ["gbdt", "goss"],
            "model__colsample_bytree": (0.2, 1.0, "uniform"),
            "model__subsample": (0.5, 1.0, "uniform"),
            "model__min_child_samples": (1, 25),
            "model__min_child_weight": (1e-6, 0.01, "log-uniform"),
        },
    },
    "catboost": {
        "model": CatBoostClassifier(
            silent=True,
            loss_function="MultiClass",
            cat_features=None,
            class_weights=None,
            boosting_type="Plain",
            max_ctr_complexity=2,
            thread_count=-1,
        ),
        "parameters": {
            "depth": (6, 15),
            "iterations": (500, 1600),
            "learning_rate": (1e-7, 1e-1),
            "reg_lambda": (1e-5, 10.0),
            "l2_leaf_reg": (0.1, 100.0),
            "bagging_temperature": (1e-8, 1.0, "log-uniform"),
            "border_count": (1, 255),
            "rsm": (0.10, 0.8, "uniform"),
            "random_strength": (1e-3, 3.0, "log-uniform"),
        },
        "additional_necessary_params": ["class_weight", "cat_features"],
    },
    "random_forest": {
        "model": imb.pipeline.Pipeline(
            steps=[
                (
                    "under",
                    RandomUnderSampler(sampling_strategy={5: int(0.11 * 76647)}),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_jobs=-1, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "parameters": {
            "model__n_estimators": (800, 3500),
            "model__max_depth": (6, 50),
            "model__max_features": (0.10, 1.0, "uniform"),
            "model__max_samples": (0.10, 0.9999, "uniform"),
        },
    },
    "xgboost": {
        "model": imb.pipeline.Pipeline(
            steps=[
                (
                    "under",
                    RandomUnderSampler(sampling_strategy={5: int(0.11 * 76647)}),
                ),
                (
                    "model",
                    XGBClassifier(
                        n_jobs=-1,
                        objective="multi:softmax",
                        num_class=7,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "parameters": {
            "model__learning_rate": (0.08, 1.0, "log-uniform"),
            "model__max_depth": (6, 24),
            "model__subsample": (0.10, 1.0, "uniform"),
            "model__colsample_bytree": (0.10, 1.0, "uniform"),
            "model__reg_lambda": (1e-2, 50.0, "log-uniform"),
            "model__reg_alpha": (1e-3, 10.0, "log-uniform"),
            "model__gamma": (1e-9, 1.0, "log-uniform"),
            "model__n_estimators": (500, 2500),
        },
    },
    "histgradientboosting": {
        "model": BalancedBaggingClassifier(
            base_estimator=HistGradientBoostingClassifier(
                max_iter=800,
                scoring=f2_scorer,
                validation_fraction=0.10,
                n_iter_no_change=50,
                tol=1e-2,
            ),
            n_jobs=-1,
            n_estimators=5,
            sampling_strategy={5: int(0.11 * 76647)},
        ),
        "parameters": {
            "model__base_estimator__learning_rate": (0.001, 1.0, "log-uniform"),
            "model__base_estimator__max_leaf_nodes": (20, 72, "uniform"),
            "model__base_estimator__min_samples_leaf": (2, 25),
            "model__base_estimator__l2_regularization": (0.0, 20.0, "log-uniform"),
        },
    },
    "extratree": {
        "model": imb.pipeline.Pipeline(
            steps=[
                (
                    "under",
                    RandomUnderSampler(sampling_strategy={5: int(0.11 * 76647)}),
                ),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_jobs=-1,
                        class_weight="balanced",
                        random_state=42,
                        bootstrap=True,
                    ),
                ),
            ]
        ),
        "parameters": {
            "model__n_estimators": (600, 3500),
            "model__max_depth": (6, 80),
            "model__max_features": (0.10, 1.0, "uniform"),
            "model__max_samples": (0.10, 0.9999, "uniform"),
            "model__min_samples_split": (2, 20),
        },
    },
    "balanced_rf": {
        "model": BalancedRandomForestClassifier(
            n_jobs=-1, bootstrap=True, sampling_strategy={5: int(0.11 * 76647)},
        ),
        "parameters": {
            "n_estimators": (500, 3500),
            "max_depth": (8, 50),
            "max_features": (0.1, 0.999),
            "max_samples": (0.1, 0.999),
        },
    },
    "keras_classifier": {},
}


# NOTE: THIS USES models_dic TO CREATE A DICTIONARY WITH BOTH
# THE MODEL DECLARED AND THE PARAMETERS THAT WERE FOUND OPTIMAL.

best_models = {
    "lgbm": {
        "model": models_dic["lgbm"]["model"],
        "parameters": {
            "model__reg_alpha": 0.2549997153062324,
            "model__reg_lambda": 0.01074870513247473,
            "model__n_estimators": 2732,
            "model__learning_rate": 0.026775555730953138,
            "model__num_leaves": 59,
            "model__boosting_type": "gbdt",
            "model__colsample_bytree": 0.3331140840638429,
            "model__subsample": 0.7285855646874189,
            "model__min_child_samples": 21,
            "model__min_child_weight": 6.795620025424848e-06,
        },
        "checkpoint": None,
    },
    "histgradientboosting": {
        "model": models_dic["histgradientboosting"]["model"],
        "parameters": {
            "base_estimator__learning_rate": 0.16745509893409027,
            "base_estimator__l2_regularization": 0.3850353881566408,
            "base_estimator__max_leaf_nodes": 66,
            "base_estimator__min_samples_leaf": 18,
        },
        "checkpoint": None,
    },
    "xgboost": {
        "model": models_dic["xgboost"]["model"],
        "parameters": {
            "model__gamma": 0.00037458492373599933,
            "model__learning_rate": 0.21930192955171976,
            "model__max_depth": 24,
            "model__colsample_bytree": 0.1,
            "model__n_estimators": 500,
            "model__reg_alpha": 0.001,
            "model__reg_lambda": 0.01,
            "model__subsample": 1.0,
        },
        "checkpoint": None,
    },
    "random_forest": {
        "model": models_dic["random_forest"]["model"],
        "parameters": {
            "model__max_depth": 19,
            "model__max_features": 0.1,
            "model__max_samples": 0.999,
            "model__n_estimators": 800,
        },
        "checkpoint": None,
    },
    "extratree": {
        "model": models_dic["extratree"]["model"],
        "parameters": {
            "model__n_estimators": 300,
            "model__max_depth": 120,
            "model__max_features": 0.1,
            "model__max_samples": 0.9999,
            "model__min_samples_split": 2,
            "model__class_weight": "balanced",
        },
        "checkpoint": None,
    },
    "balanced_rf": {
        "model": models_dic["balanced_rf"]["model"],
        "parameters": {
            "max_features": 0.6943126024391076,
            "max_samples": 0.7732151017758678,
            "min_samples_split": 3,
            "max_depth": 34,
            "n_estimators": 309,
        },
        "checkpoint": None,
    },
}


# NOTE: DICTIONARY OF STACKING MODELS FOR BAYES SEARCH CV.
stacking_models = {
    "StackingBS1": {
        "model": StackingClassifier(
            estimators=[
                (
                    "lgbm",
                    best_models["lgbm"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["lgbm"]["parameters"]),
                ),
                (
                    "random_forest",
                    best_models["random_forest"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["random_forest"]["parameters"]),
                ),
                (
                    "xgboost",
                    best_models["xgboost"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["xgboost"]["parameters"]),
                ),
                (
                    "extratree",
                    best_models["extratree"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["extratree"]["parameters"]),
                ),
                (
                    "histgradientboosting",
                    best_models["histgradientboosting"]["model"]
                    .set_params(
                        **{"sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["histgradientboosting"]["parameters"]),
                ),
                (
                    "balanced_rf",
                    best_models["balanced_rf"]["model"]
                    .set_params(
                        **{"sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["balanced_rf"]["parameters"]),
                ),
            ],
            final_estimator=imb.pipeline.Pipeline(
                steps=[
                    (
                        "under",
                        RandomUnderSampler(
                            sampling_strategy={5: int(0.11 * (4 / 9) * 76647)}
                        ),
                    ),  # 4/9 because of double cross-validation, 3 fold for BayesSearchCV and 3 fold for final_estimator.
                    ("model", LGBMClassifier(n_jobs=-1, boosting_type="gbdt")),
                ]
            ),
            verbose=1,
            n_jobs=-1,
            cv=3,
        ),
        "parameters": {
            "final_estimator__model__n_estimators": (80, 500),  # 1500
            "final_estimator__model__learning_rate": (5e-3, 5e-2, "log-uniform"),
            "final_estimator__model__num_leaves": (16, 80),
            "final_estimator__model__reg_lambda": (1e-2, 15.0, "log-uniform"),
            "final_estimator__model__reg_alpha": (1e-4, 1.0, "log-uniform"),
            "final_estimator__model__colsample_bytree": (0.2, 1.0, "uniform"),
            "final_estimator__model__subsample": (0.5, 1.0, "uniform"),
            "final_estimator__model__min_child_samples": (1, 25),
            "final_estimator__model__min_child_weight": (1e-6, 0.01, "log-uniform"),
        },
    },
    "StackingAlex1": {
        "model": StackingClassifier(
            estimators=[
                (
                    "lgbm",
                    best_models["lgbm"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["lgbm"]["parameters"]),
                ),
                (
                    "random_forest",
                    best_models["random_forest"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["random_forest"]["parameters"]),
                ),
                (
                    "xgboost",
                    best_models["xgboost"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["xgboost"]["parameters"]),
                ),
                (
                    "extratree",
                    best_models["extratree"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["extratree"]["parameters"]),
                ),
            ],
            final_estimator=imb.pipeline.Pipeline(
                steps=[
                    (
                        "under",
                        RandomUnderSampler(
                            sampling_strategy={5: int(0.11 * (4 / 9) * 76647)}
                        ),
                    ),  # 4/9 because of double cross-validation, 3 fold for BayesSearchCV and 3 fold for final_estimator.
                    ("model", LGBMClassifier(n_jobs=-1, boosting_type="gbdt")),
                ]
            ),
            verbose=1,
            n_jobs=-1,
            cv=3,
        ),
        "parameters": {
            "final_estimator__model__n_estimators": (80, 500),  # 1500
            "final_estimator__model__learning_rate": (1e-2, 0.8, "log-uniform"),
            "final_estimator__model__num_leaves": (16, 72),
            "final_estimator__model__reg_lambda": (1e-2, 20.0, "log-uniform"),
            "final_estimator__model__reg_alpha": (1e-4, 1.0, "log-uniform"),
            "final_estimator__model__colsample_bytree": (0.2, 1.0, "uniform"),
            "final_estimator__model__subsample": (0.5, 1.0, "uniform"),
            "final_estimator__model__min_child_samples": (1, 25),
            "final_estimator__model__min_child_weight": (1e-6, 0.01, "log-uniform"),
        },
    },
    "StackingAlex2": {
        "model": StackingClassifier(
            estimators=[
                (
                    "lgbm",
                    best_models["lgbm"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["lgbm"]["parameters"]),
                ),
                (
                    "random_forest",
                    best_models["random_forest"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["random_forest"]["parameters"]),
                ),
                (
                    "xgboost",
                    best_models["xgboost"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["xgboost"]["parameters"]),
                ),
                (
                    "extratree",
                    best_models["extratree"]["model"]
                    .set_params(
                        **{"under__sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["extratree"]["parameters"]),
                ),
                (
                    "histgradientboosting",
                    best_models["histgradientboosting"]["model"]
                    .set_params(
                        **{"sampling_strategy": {5: int(0.11 * 76647 * (2 / 3))}}
                    )
                    .set_params(**best_models["histgradientboosting"]["parameters"]),
                ),
            ],
            final_estimator=imb.pipeline.Pipeline(
                steps=[
                    (
                        "under",
                        RandomUnderSampler(
                            sampling_strategy={5: int(0.11 * (4 / 9) * 76647)}
                        ),
                    ),  # 4/9 because of double cross-validation, 3 fold for BayesSearchCV and 3 fold for final_estimator.
                    ("model", LGBMClassifier(n_jobs=-1, boosting_type="gbdt")),
                ]
            ),
            verbose=1,
            n_jobs=-1,
            cv=3,
        ),
        "parameters": {
            "final_estimator__model__n_estimators": (80, 500),  # 1500
            "final_estimator__model__learning_rate": (1e-2, 0.8, "log-uniform"),
            "final_estimator__model__num_leaves": (16, 72),
            "final_estimator__model__reg_lambda": (1e-2, 20.0, "log-uniform"),
            "final_estimator__model__reg_alpha": (1e-4, 1.0, "log-uniform"),
            "final_estimator__model__colsample_bytree": (0.2, 1.0, "uniform"),
            "final_estimator__model__subsample": (0.5, 1.0, "uniform"),
            "final_estimator__model__min_child_samples": (1, 25),
            "final_estimator__model__min_child_weight": (1e-6, 0.01, "log-uniform"),
        },
    },
}


# NOTE: DICTIONARY WITH THE FINAL MODELS WITH THE BEST PARAMETERS FOR
# BOTH THE BASE ESTIMATORS AND THE FINAL ESTIMATORS, FOUND AFTER SEVERAL
# BAYES SEARCH CV EXPERIMENTS AS EXPLAINED IN THE DOCUMENT.

FINAL_MODELS = {
    "StackingAlex_FULL": {
        "model": StackingClassifier(
            estimators=[
                (
                    "lgbm",
                    best_models["lgbm"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["lgbm"]["parameters"]),
                ),
                (
                    "random_forest",
                    best_models["random_forest"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["random_forest"]["parameters"]),
                ),
                (
                    "xgboost",
                    best_models["xgboost"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["xgboost"]["parameters"]),
                ),
                (
                    "extratree",
                    best_models["extratree"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["extratree"]["parameters"]),
                ),
                (
                    "histgradientboosting",
                    best_models["histgradientboosting"]["model"]
                    .set_params(**{"sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["histgradientboosting"]["parameters"]),
                ),
                (
                    "balanced_rf",
                    best_models["balanced_rf"]["model"]
                    .set_params(**{"sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["balanced_rf"]["parameters"]),
                ),
            ],
            final_estimator=imb.pipeline.Pipeline(
                steps=[
                    (
                        "under",
                        RandomUnderSampler(
                            sampling_strategy={5: int(0.11 * (2 / 3) * 90173)}
                        ),
                    ),
                    (
                        "model",
                        LGBMClassifier(n_jobs=-1, boosting_type="gbdt").set_params(
                            **{
                                "colsample_bytree": 0.9074472342809521,
                                "learning_rate": 0.008296054702313257,
                                "min_child_samples": 6,
                                "min_child_weight": 0.003388459033965211,
                                "n_estimators": 161,
                                "num_leaves": 51,
                                "reg_alpha": 0.002895693625773787,
                                "reg_lambda": 0.05932653205105829,
                                "subsample": 0.977678197120508,
                            }
                        ),
                    ),
                ]
            ),
            verbose=1,
            n_jobs=-1,
            cv=3,
        )
    },
    "StackingAlex1": {
        "model": StackingClassifier(
            estimators=[
                (
                    "lgbm",
                    best_models["lgbm"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["lgbm"]["parameters"]),
                ),
                (
                    "random_forest",
                    best_models["random_forest"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["random_forest"]["parameters"]),
                ),
                (
                    "xgboost",
                    best_models["xgboost"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["xgboost"]["parameters"]),
                ),
                (
                    "extratree",
                    best_models["extratree"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["extratree"]["parameters"]),
                ),
            ],
            final_estimator=imb.pipeline.Pipeline(
                steps=[
                    (
                        "under",
                        RandomUnderSampler(
                            sampling_strategy={5: int(0.11 * (2 / 3) * 90173)}
                        ),
                    ),
                    (
                        "model",
                        LGBMClassifier(n_jobs=-1, boosting_type="gbdt").set_params(
                            **{
                                "colsample_bytree": 0.6621023581919685,
                                "learning_rate": 0.005833323610324078,
                                "min_child_samples": 20,
                                "min_child_weight": 0.0007100926648543407,
                                "n_estimators": 257,
                                "num_leaves": 17,
                                "reg_alpha": 0.0003566122503710693,
                                "reg_lambda": 0.33140807899378955,
                                "subsample": 0.5063060185977877,
                            }
                        ),
                    ),
                ]
            ),
            verbose=1,
            n_jobs=-1,
            cv=3,
        )
    },
    "StackingAlex2": {
        "model": StackingClassifier(
            estimators=[
                (
                    "lgbm",
                    best_models["lgbm"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["lgbm"]["parameters"]),
                ),
                (
                    "random_forest",
                    best_models["random_forest"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["random_forest"]["parameters"]),
                ),
                (
                    "xgboost",
                    best_models["xgboost"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["xgboost"]["parameters"]),
                ),
                (
                    "extratree",
                    best_models["extratree"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["extratree"]["parameters"]),
                ),
                (
                    "histgradientboosting",
                    best_models["histgradientboosting"]["model"]
                    .set_params(**{"sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["histgradientboosting"]["parameters"]),
                ),
            ],
            final_estimator=imb.pipeline.Pipeline(
                steps=[
                    (
                        "under",
                        RandomUnderSampler(
                            sampling_strategy={5: int(0.11 * (2 / 3) * 90173)}
                        ),
                    ),  # 4/9 because of double cross-validation, 3 fold for BayesSearchCV and 3 fold for final_estimator.
                    (
                        "model",
                        LGBMClassifier(n_jobs=-1, boosting_type="gbdt").set_params(
                            **{
                                "colsample_bytree": 0.2708810667489143,
                                "learning_rate": 0.0106399168456851,
                                "min_child_samples": 22,
                                "min_child_weight": 1.1856486008343247e-06,
                                "n_estimators": 108,
                                "num_leaves": 21,
                                "reg_alpha": 0.00020260202982016795,
                                "reg_lambda": 2.6657123325317276,
                                "subsample": 0.8425430132465751,
                            }
                        ),
                    ),
                ]
            ),
            verbose=1,
            n_jobs=-1,
            cv=3,
        )
    },
    "StackingArmando1": {
        "model": StackingClassifier(
            estimators=[
                (
                    "lgbm",
                    best_models["lgbm"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["lgbm"]["parameters"])
                    .set_params(**{"model__class_weight": None}),
                ),
                (
                    "random_forest",
                    best_models["random_forest"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["random_forest"]["parameters"])
                    .set_params(**{"model__class_weight": None}),
                ),
                (
                    "xgboost",
                    best_models["xgboost"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["xgboost"]["parameters"])
                    .set_params(**{"model__class_weight": None}),
                ),
                (
                    "extratree",
                    best_models["extratree"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["extratree"]["parameters"])
                    .set_params(**{"model__class_weight": None}),
                ),
                (
                    "histgradientboosting",
                    best_models["histgradientboosting"]["model"]
                    .set_params(**{"sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["histgradientboosting"]["parameters"]),
                ),
                (
                    "balanced_rf",
                    best_models["balanced_rf"]["model"]
                    .set_params(**{"sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["balanced_rf"]["parameters"]),
                ),
            ],
            final_estimator=imb.pipeline.Pipeline(
                steps=[
                    (
                        "under",
                        RandomUnderSampler(
                            sampling_strategy={5: int(0.11 * (2 / 3) * 90173)}
                        ),
                    ),
                    (
                        "model",
                        LGBMClassifier(n_jobs=-1, boosting_type="gbdt").set_params(
                            **{
                                "colsample_bytree": 0.5890592300112927,
                                "learning_rate": 0.010674780636071638,
                                "min_child_samples": 23,
                                "min_child_weight": 1.1871906994001223e-05,
                                "n_estimators": 111,
                                "num_leaves": 57,
                                "reg_alpha": 0.000645335030510169,
                                "reg_lambda": 0.015199866773896266,
                                "subsample": 0.5198182920037712,
                            }
                        ),
                    ),
                ]
            ),
            verbose=1,
            n_jobs=-1,
            cv=3,
        )
    },
    "StackingArmando2": {
        "model": StackingClassifier(
            estimators=[
                (
                    "lgbm",
                    best_models["lgbm"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["lgbm"]["parameters"]),
                ),
                (
                    "random_forest",
                    best_models["random_forest"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["random_forest"]["parameters"]),
                ),
                (
                    "xgboost",
                    best_models["xgboost"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["xgboost"]["parameters"]),
                ),
                (
                    "extratree",
                    best_models["extratree"]["model"]
                    .set_params(**{"under__sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["extratree"]["parameters"]),
                ),
                (
                    "balanced_rf",
                    best_models["balanced_rf"]["model"]
                    .set_params(**{"sampling_strategy": {5: int(0.11 * 90173)}})
                    .set_params(**best_models["balanced_rf"]["parameters"]),
                ),
            ],
            final_estimator=imb.pipeline.Pipeline(
                steps=[
                    (
                        "under",
                        RandomUnderSampler(
                            sampling_strategy={5: int(0.11 * (2 / 3) * 90173)}
                        ),
                    ),
                    (
                        "model",
                        LGBMClassifier(n_jobs=-1, boosting_type="gbdt").set_params(
                            **{
                                "colsample_bytree": 0.786725858205575,
                                "learning_rate": 0.010881709103331574,
                                "min_child_samples": 3,
                                "min_child_weight": 0.008842318157479623,
                                "n_estimators": 121,
                                "num_leaves": 18,
                                "reg_alpha": 0.5366709568674902,
                                "reg_lambda": 0.2561467982877725,
                                "subsample": 0.9467575888239212,
                            }
                        ),
                    ),
                ]
            ),
            verbose=1,
            n_jobs=-1,
            cv=3,
        )
    },
}


def build_stacking(
    models,
    base_model="LogisticRegression",
    base_model_params=None,
    cv=5,
    passthrough=False,
):
    """
    Function to build a simple stacking composed of models loaded in above dicts.

    Parameters
    -------------------
    models: list
        Models to use as base estimators.
    base_model: str
        Model to use as final estimator.
    base_model_params: dict
        Dict containing the parameters for the final estimator.
    cv: int
        The number of splits for a StratifiedKFold (k).
    passthrough: bool
        Whether or not to fit the final estimator with the data as well as
        with the base estimators' predictions.
    
    Returns
    -------------------
    A StackingClassifier. 
    """
    print("1")
    print(base_model_params)
    print(type(base_model_params))
    base_model_params = dict(base_model_params)
    base_models = [
        (m, best_models[m]["model"].set_params(**best_models[m]["parameters"]))
        for m in models
    ]
    print("2")
    if base_model == "LogisticRegression":
        final_estimator = imb.pipeline.Pipeline(
            steps=[
                (
                    "under",
                    RandomUnderSampler(
                        sampling_strategy={5: int(0.11 * (4 / 5) * 76647)}
                    ),
                ),
                ("model", LogisticRegression().set_params(**base_model_params)),
            ]
        )
    elif base_model == "LGBM":
        final_estimator = imb.pipeline.Pipeline(
            steps=[
                (
                    "under",
                    RandomUnderSampler(
                        sampling_strategy={5: int(0.11 * (4 / 5) * 76647)}
                    ),
                ),
                ("model", LGBMClassifier(**base_model_params)),
            ]
        )
    print(final_estimator._estimator_type)
    print("getting stacking")
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=final_estimator,
        n_jobs=-1,
        passthrough=passthrough,
        verbose=1,
    )
    return stacking
