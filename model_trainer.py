from sklearn.ensemble import RandomForestClassifier
from preprocessing import *
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
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

NAME = 'rf_100BS'
N_ITER = 100

params = {'n_estimators': (100, 1500),
            'max_depth': (2, 10),
            'class_weight': ['balanced'],
            'n_jobs': [-1],
            'max_features': ['auto', 'log2'],
            }

model = RandomForestClassifier()


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
    mlflow.start_run(run_name=NAME)
    print('procesando los datos')
    X, y = preprocess_data('Modelar_UH2020.txt', process_cat=True)
    y=y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    setlabs = [l for l in set(y_train)]
    tag2idx = {i: l for l, i in enumerate(setlabs)}

    best_model = BayesSearchCV(
                model,
                params,
                n_iter = N_ITER,
                cv=3,
                scoring='f1_macro',
                random_state=42,
                )

    def on_step(optim_result):
        score = best_model.best_score_
        mlflow.log_metric('best_score', score)
        print("best score: %s" % score)
        if score >= 0.98:
            print('Interrupting!')
            return True

    print('ajustando modelo')
    best_model.fit(X_train, y_train, callback=on_step)
    with open(f'best_rf_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print('loggeando movidas')
    mlflow.log_artifact('best_rf_model.pkl')
    best_params = best_model.best_params_
    for param in best_params.keys():
         mlflow.log_param(param, best_params[param])
    preds = best_model.predict(X_test)
    preds_proba = best_model.predict_proba(X_test)
    cm = confusion_matrix(y_test, preds)
    grafico_conf_matrix = print_confusion_matrix(cm, class_names = setlabs)
    grafico_conf_matrix.savefig(NAME)
    grafico_norm = print_confusion_matrix(cm, class_names = setlabs, normalize=False)
    grafico_norm.savefig(f'{NAME}_no_norm')
    mlflow.log_metrics(metrics={'f1': f1_score(y_test, preds, average='macro'),
                           'precision': precision_score(y_test, preds, average='macro'),
                           'recall': recall_score(y_test, preds, average='macro'),
                           'accuracy': accuracy_score(y_test, preds)})
    mlflow.log_artifact(f'./{NAME}.png')
    mlflow.log_artifact(f'./{NAME}_no_norm.png')
    mlflow.sklearn.log_model(best_model.best_estimator_, 'random_forest_model')
    mlflow.end_run()


if __name__ == '__main__':
    main()
