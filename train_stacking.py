##########################################
### AUTHOR: ALEJANDRO VACA SERRANO #######
##########################################


from models import *
import pickle
from model_trainer_refactor import *
import argparse
from category_encoders.one_hot import OneHotEncoder
import json
from urllib.parse import unquote
from sklearn.metrics import confusion_matrix
import sys
import os

SAVE_DIR = "stacking_models"

if SAVE_DIR not in os.listdir():
    os.mkdir(SAVE_DIR)


def save_obj(obj, name):
    print(f"### SAVING {name} ###")
    with open(f"./{SAVE_DIR}/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def load_obj(name):
    print(f"### LOADING {name} ###")
    with open(f"./{SAVE_DIR}/{name}.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj


def transform_types_X(X):
    cols_cat = ["ruido", "CODIGO_POSTAL", "ZONA_METROPOLITANA", "CALIDAD_AIRE"]
    cols_float = [col for col in X.columns if col not in cols_cat]
    X[cols_float] = X[cols_float].astype("float64")
    return X


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--models",
        dest="models",
        required=False,
        help="The models to use; see models.py for available models",
        type=str,
    )
    parser.add_argument(
        "--model",
        dest="model",
        required=True,
        default=None,
        help="when training a individual model, the name of the model",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        required=True,
        help="The name to put to the experiment",
        type=str,
    )
    parser.add_argument(
        "-fm",
        "--final_model",
        dest="final_model",
        default="LogisticRegression",
        required=False,
        help="The final model to use in the stacking",
        type=str,
    )
    parser.add_argument(
        "-fmp",
        "--final_model_parameters",
        dest="final_model_parameters",
        required=False,
        help="The parameters for the final estimator",
    )
    parser.add_argument(
        "-cv",
        "--cv",
        dest="cv",
        required=False,
        default=3,
        help="the cross validation splits",
        type=int,
    )
    parser.add_argument(
        "-pt",
        "--passthrough",
        dest="passthrough",
        default=False,
        required=False,
        help="whether or not to pass the X_train also to the final estimator",
        type=bool,
    )
    parser.add_argument(
        "-enc",
        "--encoder",
        dest="encoder",
        default="CatBoost",
        required=False,
        type=str,
        help="The category encoder to use",
    )
    parser.add_argument(
        "-ut",
        "--under_test",
        help="Whether or not to undersample the test",
        required=False,
        default=True,
        type=bool,
    )
    parser.add_argument(
        "-pret", "--pretrained", required=False, default=True, type=bool,
    )

    print("parse args")
    args = parser.parse_args()

    mlflow.start_run(run_name=args.name)
    if not args.pretrained:
        print("to str")
        jsonString = unquote(args.final_model_parameters)
        print("getting dict of params")
        params = dict(json.loads(jsonString))
        print(params)

        params.update(
            {
                "n_estimators": int(params["n_estimators"]),
                "n_leaves": int(params["n_leaves"]),
                "n_jobs": -1,
                "reg_lambda": float(params["reg_lambda"]),
                "colsample_bytree": float(params["colsample_bytree"]),
            }
        )
        print(params)
        args.models = args.models.split(",")
        print(args)
        stacking = build_stacking(
            models=args.models,
            base_model=args.final_model,
            base_model_params=params,
            cv=args.cv,
            passthrough=args.passthrough,
        )
        print(stacking)
    else:

        stacking = FINAL_MODELS[args.model]["model"]

    print("LOADING FILES AND TRANSFORMING TYPES")
    X_train, X_test = load_obj("X_train"), load_obj("X_test")
    if "Oeste" in X_train.columns:
        X_train = X_train.drop("Oeste", axis=1)
        X_test = X_test.drop("Oeste", axis=1)
    X_train = transform_types_X(X_train)
    X_test = transform_types_X(X_test)
    y_train, y_test = load_obj("y_train"), load_obj("y_test")
    encoder = load_obj("label_encoder")
    print("CHANGING COLUMN NAMES")
    X_train.columns = [
        "".join(c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns
    ]
    X_test.columns = [
        "".join(c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns
    ]
    if args.encoder == "CatBoost":
        cat_encoder = CatBoostEncoder()
    elif args.encoder == "OneHot":
        cat_encoder = OneHotEncoder()
    print("HACIENDO CATEGORICAL ENCODER")
    X_train = cat_encoder.fit_transform(X_train, y_train)
    X_test = cat_encoder.transform(X_test)
    print("FITTING STACKING")
    stacking.fit(X_train, y_train)
    save_obj(stacking, f"{args.name}")
    X_test, y_test = RandomUnderSampler(
        sampling_strategy={5: int(0.11 * 13526)}
    ).fit_resample(X_test, y_test)
    preds = stacking.predict(X_test)
    save_obj(preds, f"{args.name}_preds")
    print(
        f"F1 SCORE {f1_score(y_test, preds , average='macro')}, F2 SCORE {fbeta_score(y_test, preds, average='macro', beta=2)},F05 SCORE {fbeta_score(y_test, preds, average='macro', beta=0.5)}, PRECISION IS {precision_score(y_test, preds, average='macro')},RECALL IS {recall_score(y_test, preds, average='macro')}, ACCURACY IS {accuracy_score(y_test, preds)}"
    )
    cm = confusion_matrix(y_test, preds, normalize="true")
    fig = print_confusion_matrix(cm, class_names=encoder.classes_)  # , figsize=(10, 8))
    fig.savefig(f"./{SAVE_DIR}/{args.name}_CONFUSION_MATRIX.png")

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
