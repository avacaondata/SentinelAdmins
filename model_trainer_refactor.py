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
from sklearn.metrics import make_scorer, fbeta_score  # Evaluation functions
from imblearn.combine import SMOTETomek, SMOTEENN
import argparse
from models import models_dic


f2_scorer = make_scorer(fbeta_score, beta=2, average="macro")
f05_scorer = make_scorer(fbeta_score, beta=0.5, average="macro")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)


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
        required=True,
        help="Whether or not to use the matrices and encoders saved",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-i",
        "--iter",
        dest="iter",
        help="n iter in bayes search cv",
        type=int,
        default=100,
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
        with open(f"./totalbs_{args.name}_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        print("best score: %s" % score)
        if score >= 0.98:
            print("Interrupting!")
            return True

    mlflow.start_run(run_name=args.name)

    if not args.use_old:
        print("procesando los datos")
        X, y, encoder = preprocess_data("TOTAL_TRAIN.csv", process_cat=True)
        X = X.astype("float")
        print(X.shape)

        with open(f"label_encoder_{args.name}.pkl", "wb") as f:
            pickle.dump(encoder, f)
        y = y.astype("int")
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
        with open("label_encoder_LIGHTGBM_VARS_FINAL.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open("best_LIGHTGBM_VARS_FINAL_params.pkl", "rb") as f:
            params = pickle.load(f)
    labs_names = [c for c in encoder.classes_]

    model = models_dic[args.model]["model"]
    params = models_dic[args.model]["parameters"]
    best_model = BayesSearchCV(
        model,
        params,
        n_iter=args.iter,
        n_points=1,
        cv=cv,
        scoring="f1_macro",
        random_state=100,
        optimizer_kwargs={"n_initial_points": 20},
    )
    good_colnames = []
    i = 0
    for col in X_train.columns:
        if not col.isascii():
            print(f"La columna {col} no es ascii")
            good_colnames.append(f"{i}_not_ascii")
            i += 1
        else:
            good_colnames.append(col)
    X_train.columns = good_colnames
    print("ajustando modelo")
    best_model.fit(X_train, y_train, callback=[on_step])
    with open(f"./best_{args.name}_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
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

    best_params = best_model.best_params_
    for param in best_params.keys():
        mlflow.log_param(param, best_params[param])
    cm = confusion_matrix(y_test, preds)
    grafico_conf_matrix = print_confusion_matrix(cm, class_names=labs_names)
    grafico_conf_matrix.savefig(args.name)
    grafico_norm = print_confusion_matrix(cm, class_names=labs_names, normalize=False)
    grafico_norm.savefig(f"{args.name}_no_norm")
    mlflow.end_run()
