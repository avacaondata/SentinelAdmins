##########################################
### AUTHOR: ALEJANDRO VACA SERRANO #######
##########################################


from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
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
from sklearn.metrics import make_scorer, fbeta_score, f1_score  # Evaluation functions
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from models import UnderSampling, OverSampling

NAME = "HISTGB"
N_ITER = 40
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)
MODE = "INDIVIDUAL"
pd.options.display.max_rows = 999

f2_scorer = make_scorer(fbeta_score, beta=2, average="macro")
f05_scorer = make_scorer(fbeta_score, beta=0.5, average="macro")



params_rf = {
    "n_estimators": (1000, 2000),
    "max_features": ["auto", "log2"],
    "min_samples_split": (2, 15),
}

rf = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)


params_lgbm = {
    "reg_alpha": (1e-3, 5.0, "log-uniform"),
    "reg_lambda": (1e-2, 50.0, "log-uniform"),
    "n_estimators": (600, 4000),
    "learning_rate": (5e-3, 1.0, "log-uniform"),
    "num_leaves": (25, 70),
    "boosting_type": ["gbdt", "goss"],
    "colsample_bytree": (0.1, 1.0, "uniform"),
    "subsample": (0.1, 1.0, "uniform"),
    "min_child_samples": (1, 25),
    "min_child_weight": (1e-6, 0.1, "log-uniform"),
}

lgbm = LGBMClassifier(
    class_weight="balanced",
    objective="multiclass:softmax",
    n_jobs=-1,
    random_state=100,
    silent=True,
)


params_catboost = {
    "depth": (6, 30),
    "iterations": (500, 1600),
    #'learning_rate': (1e-7, 1e-1),
    "reg_lambda": (1e-5, 10.0),
    #'l2_leaf_reg':(0.1, 100.),
    "bagging_temperature": (1e-8, 1.0, "log-uniform"),
    "border_count": (1, 255),
    #'rsm':(0.10, 0.8, 'uniform'),
    "random_strength": (1e-3, 3.0, "log-uniform"),
}


params_etc = {
    "n_estimators": (1500, 3000),
    "max_features": ["auto", "log2"],
    "min_samples_split": (2, 5),
}

etc = ExtraTreesClassifier(n_jobs=-1, random_state=100, class_weight="balanced")

params_gb = {
    "n_estimators": (1500, 2500),
    "loss": ["deviance", "exponential"],
    "learning_rate": (1e-2, 0.2, "log-uniform"),
    "min_samples_split": (2, 10),
    "max_depth": (8, 50),
    "max_features": ["auto", "log2"],
}




def print_confusion_matrix(
    confusion_matrix, class_names, figsize=(10, 7), fontsize=14, normalize=True
):
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
        confusion_matrix = (
            confusion_matrix.astype("float")
            / confusion_matrix.sum(axis=1)[:, np.newaxis]
        )
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names,)
    fig = plt.figure(figsize=figsize)
    fmt = ".2f" if normalize else "d"
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return fig


def get_classes_order_catboost(X_train, y_train):
    cat = CatBoostClassifier(
        iterations=10, depth=2, learning_rate=0.05, loss_function="MultiClass"
    )
    cat.fit(X_train, y_train)
    return cat.classes_



def lgb_f1_score(y_true, y_pred):
            preds = y_pred.reshape(len(np.unique(y_true)), -1)
            preds = preds.argmax(axis = 0)
            return 'f2', fbeta_score(y_true, preds, beta=2, average='macro'), True



def main():
    mlflow.start_run(run_name=NAME)

    if "X_train.pkl" not in os.listdir():
        print("procesando los datos")
        X, y, encoder = preprocess_data("TOTAL_TRAIN.csv", process_cat=False)
        print(X.shape)

        with open(f"label_encoder_{NAME}.pkl", "wb") as f:
            pickle.dump(encoder, f)
        print(
            f"##################### The shape of X is {X.shape} #######################"
        )
        y = y.astype("int")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=15, stratify=y
        )
        with open("X_train.pkl", "wb") as f:
            pickle.dump(X_train, f)
        with open("X_test.pkl", "wb") as f:
            pickle.dump(X_test, f)
        with open("y_train.pkl", "wb") as f:
            pickle.dump(y_train, f)
        with open("y_test.pkl", "wb") as f:
            pickle.dump(y_test, f)

        print(X_train.shape)

    else:
        with open("X_train.pkl", "rb") as f:
            X_train = pickle.load(f)
        with open("X_test.pkl", "rb") as f:
            X_test = pickle.load(f)
        with open("y_train.pkl", "rb") as f:
            y_train = pickle.load(f)
        with open("y_test.pkl", "rb") as f:
            y_test = pickle.load(f)
        with open(f"label_encoder_XGB1704.pkl", "rb") as f:
            encoder = pickle.load(f)
        print("######### ajustando cat encoder ############")
        
    cols_cat = ["ruido", "CODIGO_POSTAL", "ZONA_METROPOLITANA", "CALIDAD_AIRE"]
    cols_float = [col for col in X_train.columns if col not in cols_cat]
    X_train[cols_float] = X_train[cols_float].astype("float")
    X_test[cols_float] = X_test[cols_float].astype("float")
    
    labs_names = [c for c in encoder.classes_]

    
    model = LGBMClassifier(
        class_weight="balanced",
        objective="multiclass:softmax",
        n_jobs=-1,
        random_state=100,
        silent=True,
    )


    if MODE != "INDIVIDUAL":
        params = {
            "reg_alpha": (1e-3, 5.0, "log-uniform"),
            "reg_lambda": (1e-2, 50.0, "log-uniform"),
            "n_estimators": (600, 4500),
            "learning_rate": (5e-3, 1.0, "log-uniform"),
            "num_leaves": (20, 80),
            "boosting_type": ["gbdt", "goss"],
            "colsample_bytree": (0.1, 1.0, "uniform"),
            "subsample": (0.1, 1.0, "uniform"),
            "min_child_samples": (1, 25),
            "min_child_weight": (1e-6, 0.1, "log-uniform"),
        }

        
        print(params)


        cb = CatBoostEncoder(cols=cols_cat)
        X_train = cb.fit_transform(X_train, y_train)
        X_test = cb.transform(X_test)
        fit_params = {
            ### fit params ###
            "eval_set": [(X_test, y_test)],
            "eval_metric": lgb_f1_score,
            "early_stopping_rounds": 300,
        }


        pipeline = Pipeline(
            steps=[("clas_encoder", CatBoostEncoder(cols=cols_cat)), ("model", model)]
        )

        best_model = BayesSearchCV(
            model,
            params,
            n_iter=N_ITER,
            n_points=1,
            cv=cv,
            scoring=f2_scorer,
            random_state=100,
            optimizer_kwargs={"n_initial_points": 10},
            fit_params=fit_params,
        )

    def on_step(optim_result):
        score = best_model.best_score_
        results = best_model.cv_results_
        try:
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"results_{NAME}.csv", header=True, index=False)
            print(
                f"############ Llevamos {results_df.shape[0]} pruebas #################"
            )
            print(f"los resultados del cv de momento son {results_df}")
        except:
            print("Unable to convert cv results to pandas dataframe")
        mlflow.log_metric("best_score", score)
        with open(f"./best_{NAME}_params.pkl", "wb") as f:
            pickle.dump(best_model.best_params_, f)
        
        print("best score: %s" % score)
        if score >= 0.98:
            print("Interrupting!")
            return True


    print("ajustando modelo")
    if MODE != "INDIVIDUAL":
        print(X_train.dtypes)
        best_model.fit(X_train, y_train, callback=[on_step])
        with open(f"./best_{NAME}_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        preds = best_model.predict(X_test)
    else:
        if NAME not in os.listdir():
            os.mkdir(NAME)
        
        cat_encoder = CatBoostEncoder(cols=cols_cat)
        X_train = cat_encoder.fit_transform(X_train, y_train)
        X_test = cat_encoder.transform(X_test)
        best_model = BalancedBaggingClassifier(
            base_estimator=HistGradientBoostingClassifier(
                max_iter=3000,
                random_state=42,
                learning_rate=0.1,
                max_leaf_nodes=54,
                min_samples_leaf=2,
                scoring=f2_scorer,
                validation_fraction=0.1,
                n_iter_no_change=50,
            ),
            n_estimators=5,
            random_state=42,
            n_jobs=-1,
            max_features=0.7,
            sampling_strategy={5: int(dict(Counter(y_train))[5] * 0.11)},
        )
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)
        print(
            f'F1 SCORE IS {f1_score(y_test, preds, average="macro")}, precision is {precision_score(y_test, preds, average="macro")}, recall is {recall_score(y_test, preds, average="macro")}, accuracy is {accuracy_score(y_test, preds)}'
        )
        print(f"F2 SCORE IS {fbeta_score(y_test, preds, average='macro', beta=2)}")
        print(f"F05 SCORE IS {fbeta_score(y_test, preds, average='macro', beta=2)}")
        cm = confusion_matrix(y_test, preds)
        grafico_conf_matrix = print_confusion_matrix(cm, class_names=labs_names)
        grafico_conf_matrix.savefig(f"{NAME}/norm_NO_PIPELINE")

        with open(f"best_model_{NAME}.pkl", "wb") as f:
            pickle.dump(best_model, f)
        
    print("loggeando movidas")
    mlflow.log_metrics(
        metrics={
            "f1": f1_score(y_test, preds, average="macro"),
            "precision": precision_score(y_test, preds, average="macro"),
            "recall": recall_score(y_test, preds, average="macro"),
            "accuracy": accuracy_score(y_test, preds),
            "f05": fbeta_score(y_test, preds, beta=0.5, average="macro"),
            "f2": fbeta_score(y_test, preds, beta=2, average="macro"),
        }
    )
    if MODE != "INDIVIDUAL":
        best_params = best_model.best_params_
        for param in best_params.keys():
            mlflow.log_param(param, best_params[param])
    cm = confusion_matrix(y_test, preds)
    grafico_conf_matrix = print_confusion_matrix(cm, class_names=labs_names)
    grafico_conf_matrix.savefig(NAME)
    grafico_norm = print_confusion_matrix(cm, class_names=labs_names, normalize=False)
    grafico_norm.savefig(f"{NAME}_no_norm")
    mlflow.end_run()


if __name__ == "__main__":
    main()
