##########################################
### AUTHOR: ALEJANDRO VACA SERRANO #######
##########################################


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
from sklearn.metrics import make_scorer, fbeta_score
from imblearn.combine import SMOTETomek, SMOTEENN
import argparse
from models import models_dic, build_stacking, stacking_models, best_models
from category_encoders.cat_boost import CatBoostEncoder
import xgboost
import random
import imblearn as imb
from sklearn.ensemble import StackingClassifier


f2_scorer = make_scorer(fbeta_score, beta=2, average="macro")
f05_scorer = make_scorer(fbeta_score, beta=0.5, average="macro")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)


def get_cv_iterable(X_train, y_train):
    it = []
    sk = StratifiedKFold(n_splits=3, shuffle=True)
    for train_index, test_index in sk.split(X_train, y_train):
        X_test1, y_test1 = X_train.iloc[test_index, :], y_train[test_index]
        select_from = X_test1.index[y_test1 == 5].tolist()
        other_indices = [ind for ind in X_test1.index if ind not in select_from]
        select = random.choices(select_from, k=int(0.11 * len(select_from)))
        total_select = other_indices + select
        it.append((train_index, total_select))
    return it


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        required=True,
        help="The model to use; see models.py for available models",
        type=str,
    )

    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        required=True,
        help="The name to put to the model saved and the mlflow run",
        type=str,
    )

    parser.add_argument(
        "-uo",
        "--use_old",
        dest="use_old",
        required=False,
        help="Whether or not to use the matrices and encoders saved",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "-i",
        "--iter",
        dest="iter",
        help="n iter in bayes search cv",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-stack",
        "--stacking",
        dest="stacking",
        help="Whether to stack or not to stack, that's the question",
        type=bool,
        required=False,
        default=True,
    )

    args = parser.parse_args()

    def on_step(optim_result):
        score = best_model.best_score_
        results = best_model.cv_results_
        try:
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"results_{args.name}.csv", header=True, index=False)
            print(
                f"############ Llevamos {results_df.shape[0]} pruebas #################"
            )
            print(f"los resultados del cv de momento son {results_df}")
        except:
            print("Unable to convert cv results to pandas dataframe")
        mlflow.log_metric("best_score", score)
        with open(f"./best_{args.name}_params.pkl", "wb") as f:
            pickle.dump(best_model.best_params_, f)
        print("best score: %s" % score)
        if score >= 0.98:
            print("Interrupting!")
            return True

    mlflow.start_run(run_name=args.name)

    if not args.use_old:
        print("procesando los datos")
        X, y, encoder = preprocess_data("TOTAL_TRAIN.csv", process_cat=False)
        cols_cat = ["ZONA_METROPOLITANA", "CODIGO_POSTAL", "ruido", "CALIDAD_AIRE"]
        cols_float = [col for col in X.columns if col not in cols_cat]
        X[cols_float] = X[cols_float].astype("float")
        print(X.shape)

        with open(f"label_encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=15, stratify=y
        )
        print(X_train.shape)
        with open("X_train.pkl", "wb") as f:
            pickle.dump(X_train, f)
        with open("X_test.pkl", "wb") as f:
            pickle.dump(X_test, f)
        with open("y_train.pkl", "wb") as f:
            pickle.dump(y_train, f)
        with open("y_test.pkl", "wb") as f:
            pickle.dump(y_test, f)
    else:
        with open("X_train.pkl", "rb") as f:
            X_train = pickle.load(f)
        with open("X_test.pkl", "rb") as f:
            X_test = pickle.load(f)
        with open("y_train.pkl", "rb") as f:
            y_train = pickle.load(f)
        with open("y_test.pkl", "rb") as f:
            y_test = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)

        cols_cat = ["ZONA_METROPOLITANA", "CODIGO_POSTAL", "ruido", "CALIDAD_AIRE"]
        cols_float = [col for col in X_train.columns if col not in cols_cat]
        X_train[cols_float] = X_train[cols_float].astype("float")
        X_test[cols_float] = X_test[cols_float].astype("float")

    cat_encoder = CatBoostEncoder(cols=cols_cat)
    X_train = cat_encoder.fit_transform(X_train, y_train)
    X_test = cat_encoder.transform(X_test)
    if "Oeste" in X_train.columns:
        X_train = X_train.drop("Oeste", axis=1)
        X_test = X_test.drop("Oeste", axis=1)
    labs_names = [c for c in encoder.classes_]
    if not args.stacking:
        model = models_dic[args.model]["model"]
        params = models_dic[args.model]["parameters"]
    else:
        model = stacking_models[args.model]["model"]
        params = stacking_models[args.model]["parameters"]

    counter = dict(Counter(y_train))
    if not args.stacking:
        samp_strategy = {5: int(0.11 * counter[5])}
        model.set_params(**{"model__sampling_strategy": samp_strategy})

    for col in X_train.columns:
        if col not in X_test.columns:
            print(f"{col} NO ESTÁ EN TEST")
    for col in X_test.columns:
        if col not in X_train.columns:
            print(f"{col} NOT IN TRAIN")
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    good_colnames = []
    i = 0
    for col in X_train.columns:
        if not col.isascii():
            print(f"La columna {col} no es ascii")
            good_colnames.append(f"{i}_not_ascii")
            i += 1
        else:
            good_colnames.append(col)
    good_colnames = [
        "".join(c if c.isalnum() else "_" for c in str(x)) for x in good_colnames
    ]
    X_train.columns = good_colnames
    X_test.columns = good_colnames
    counter2 = dict(Counter(y_test))
    samp_strategy2 = {5: int(0.11 * counter2[5])}
    X_test, y_test = RandomUnderSampler(sampling_strategy=samp_strategy2).fit_resample(
        X_test, y_test
    )
    X_train = X_train.sort_index(axis=1)
    X_test = X_test.sort_index(axis=1)

    cols = X_train.columns
    X_test = X_test[cols]

    iterable_cv = get_cv_iterable(X_train, y_train)

    best_model = BayesSearchCV(
        model,
        params,
        n_iter=args.iter,
        n_points=1,
        cv=iterable_cv,
        scoring="accuracy",
        random_state=100,
        optimizer_kwargs={"n_initial_points": 10},
        refit=False,
    )

    print("ajustando modelo")

    best_model.fit(X_train, y_train, callback=[on_step])

    with open(f"./best_{args.name}_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    params = best_model.best_params_

    model = StackingClassifier(
        estimators=[
            (
                "lgbm",
                best_models["lgbm"]["model"]
                .set_params(**{"under__sampling_strategy": {5: int(0.11 * 76647)}})
                .set_params(**best_models["lgbm"]["parameters"]),
            ),
            (
                "random_forest",
                best_models["random_forest"]["model"]
                .set_params(**{"under__sampling_strategy": {5: int(0.11 * 76647)}})
                .set_params(**best_models["random_forest"]["parameters"]),
            ),
            (
                "xgboost",
                best_models["xgboost"]["model"]
                .set_params(**{"under__sampling_strategy": {5: int(0.11 * 76647)}})
                .set_params(**best_models["xgboost"]["parameters"]),
            ),
            (
                "extratree",
                best_models["extratree"]["model"]
                .set_params(**{"under__sampling_strategy": {5: int(0.11 * 76647)}})
                .set_params(**best_models["extratree"]["parameters"]),
            ),
            (
                "histgradientboosting",
                best_models["histgradientboosting"]["model"]
                .set_params(**{"sampling_strategy": {5: int(0.11 * 76647)}})
                .set_params(**best_models["histgradientboosting"]["parameters"]),
            ),
            (
                "balanced_rf",
                best_models["balanced_rf"]["model"]
                .set_params(**{"sampling_strategy": {5: int(0.11 * 76647)}})
                .set_params(**best_models["balanced_rf"]["parameters"]),
            ),
        ],
        final_estimator=imb.pipeline.Pipeline(
            steps=[
                (
                    "under",
                    RandomUnderSampler(
                        sampling_strategy={5: int(0.11 * (2 / 3) * 76647)}
                    ),
                ),
                (
                    "model",
                    LGBMClassifier(n_jobs=-1, boosting_type="gbdt").set_params(
                        **{
                            k.replace("final_estimator__model__", ""): v
                            for k, v in params.items()
                        }
                    ),
                ),
            ]
        ),
        verbose=1,
        n_jobs=-1,
        cv=3,
    )

    best_model = model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
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

    best_params = params
    for param in best_params.keys():
        mlflow.log_param(param, best_params[param])
    cm = confusion_matrix(y_test, preds)
    grafico_conf_matrix = print_confusion_matrix(cm, class_names=labs_names)
    grafico_conf_matrix.savefig(args.name)
    grafico_norm = print_confusion_matrix(cm, class_names=labs_names, normalize=False)
    grafico_norm.savefig(f"{args.name}_no_norm")
    mlflow.end_run()
