import argparse
import pandas as pd
import numpy as np
import os
import joblib

# encoding
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

# training model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import random
import itertools
from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier

# pyts
import pyts
from pyts.multivariate.classification import MultivariateClassifier
from pyts.classification import TimeSeriesForest
from pyts.classification import TSBF
# from pyts.classification import KNeighborsClassifier

# visualizations
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



def encode(df):
    """
    Returns a numpy array of shape (num_timesteps_total, 16) where categorical features are one-hot encoded.
    """
    enc = OneHotEncoder()
    distraction_enc = enc.fit_transform(df[['FirstDistraction']]).toarray()
    diversion_enc = enc.fit_transform(df[['FirstDiversion']]).toarray()
    encoded = np.hstack((distraction_enc, diversion_enc))

    numerical = df[['TimeFocused', 'TimeDiverted', 'TimeDistracted', 'NumDiverted', 'NumDistracted', 'NumNotifs', 'AvgFriendsPerMin']].values
    encoded_arr = np.hstack((numerical, encoded))

    return encoded_arr


def pad(df, encoded_arr):
    """
    Returns a numpy array of shape (num_sessions, 16, 16).
    """
    grouped = df.groupby('Session')
    padded = []
    
    for session, group in grouped:
        seq = encoded_arr[group.index]

        if len(seq) < 16:
            padding = np.zeros((16 - len(seq), encoded_arr.shape[1]))
            seq = np.vstack((seq, padding))

        padded.append(seq)
    
    data = np.array(padded)
    
    return data


def get_labels(df):
    """
    Returns a 1D numpy array of shape (num_sessions,). 
    """
    labels = df.groupby('Session')['Type'].first().values
    return labels


def split(data, labels, test_size=0.5, random_state=11):
    """
    Returns train/validation/test sets in a test_size/(test_size/2)/(test_size/2) ratio.
    """
    # stratify=labels ensures same proportion of classes in train/val/test datasets as in original dataset
    X_train, X_, y_train, y_ = train_test_split(data, labels, test_size=test_size, random_state=random_state, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=test_size, random_state=random_state, stratify=y_)

    return X_train, X_val, X_test, y_train, y_val, y_test


def score_baseline(X_train, X_val, y_train, y_val, X_test, y_test, random_state):
    """
    Returns baseline scores of 2 dummy classifiers.
    """
    dummy_strat = MultivariateClassifier(DummyClassifier(strategy='stratified', random_state=random_state))
    dummy_strat.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    dummy_strat.score(X_test, y_test)
    print(f"Dummy classifier (stratified) score: {dummy_strat.score(X_test, y_test):.4f}")

    dummy_uni = MultivariateClassifier(DummyClassifier(strategy='uniform', random_state=random_state))
    dummy_uni.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    dummy_uni.score(X_test, y_test)
    print(f"Dummy classifier (uniform) score: {dummy_uni.score(X_test, y_test):.4f}")

    return


def optimize_tsf(f, X_train, y_train, n_iter, n_jobs, random_state):
    """
    Runs Bayesian optimization and returns TimeSeriesForest estimator that has been fitted on training set with found best hyperparameters.
    """
    param_space_tsf = {
        "estimator__n_estimators": (100, 1000),  # int (default = 500)
        "estimator__n_windows": (1, 5),  # int or float (default = 1.)
        "estimator__min_window_size": (0.1, 1.0),  # int or float (default = 1)
        "estimator__criterion": ["gini", "entropy"],  # str (default = “entropy”)
        "estimator__max_depth": list(range(5, 50)) + [None],  # integer or None (default = None)
        "estimator__min_samples_split": (2, 10),  #  int or float (default = 2)
        "estimator__min_samples_leaf": (1, 5),  # int or float (default = 1)
        # "estimator__min_weight_fraction_leaf": 0.0, # float (default = 0.)
        "estimator__max_features": ["sqrt", "log2", None],  # int, float, str or None (default = “sqrt”)
        "estimator__max_leaf_nodes": list(range(5, 50)) + [None],  # int or None (default = None)
        # "estimator__min_impurity_decrease": 0.0,  # float (default = 0.)
        "estimator__bootstrap": [True, False],  # bool (default = True)
    }

    bayes_tsf = BayesSearchCV(
        MultivariateClassifier(TimeSeriesForest()),
        param_space_tsf,
        n_iter=n_iter,
        cv=10,  # 10-fold cross-validation
        n_jobs=n_jobs, 
        random_state=random_state,
        verbose=1
    )
    bayes_tsf.fit(X_train, y_train)

    results_tsf = pd.DataFrame(bayes_tsf.cv_results_)
    folder = os.path.dirname(f) # get the folder of the original dataset
    results_tsf.to_csv(f"{folder}/results_tsf.csv", index=False)
    print(f"Saved Bayes optimization results as {folder}/results_tsf.csv")

    return bayes_tsf


def optimize_tsbf(f, X_train, y_train, n_iter, n_jobs, random_state):
    """
    Runs Bayesian optimization and returns TSBF estimator that has been fitted on training set with found best hyperparameters.
    """
    param_space_tsbf = {
        "estimator__n_estimators": (100, 1000),  # int (default = 500)
        "estimator__min_subsequence_size": (0.1, 1.0),  # int or float (default = 0.5)
        # "estimator__min_interval_size": (0.1, 1.0),  # int or float (default = 0.1)
        # "estimator__n_subsequences": list(range(0.1, 1.0)) + ['auto'], # ‘auto’, int or float (default = ‘auto’)
        "estimator__bins": (2, 10),  # int or array-like (default = 10)
        "estimator__criterion": ["gini", "entropy"],  # str (default = “entropy”)
        "estimator__max_depth": list(range(5, 50)) + [None], # integer or None (default = None)
        "estimator__min_samples_split": (2, 10),  # int or float (default = 2)
        "estimator__min_samples_leaf": (1, 5),  # int or float (default = 1)
        # "estimator__min_weight_fraction_leaf": (0.0, 0.5),  # float (default = 0.)
        "estimator__max_features": ["sqrt", "log2", None],  # int, float, str or None (default = “sqrt”)
        "estimator__max_leaf_nodes": list(range(5, 50)) + [None],  # int or None (default = None)
        # "estimator__min_impurity_decrease": (0.0, 0.5),  # float (default = 0.)
        # "estimator__bootstrap": [True, False],  # bool (default = True)
    }

    bayes_tsbf = BayesSearchCV(
        MultivariateClassifier(TSBF()),
        param_space_tsbf,
        n_iter=n_iter,  
        cv=10, # 10-fold cross-validation
        n_jobs=n_jobs, 
        random_state=random_state,
        verbose=1
    )
    bayes_tsbf.fit(X_train, y_train)

    results_tsbf = pd.DataFrame(bayes_tsbf.cv_results_)
    folder = os.path.dirname(f)
    results_tsbf.to_csv(f"{folder}/results_TSBF.csv", index=False)
    print(f"Saved Bayes optimization results as {folder}/results_TSBF.csv")

    return bayes_tsbf


def fit_best(X_train, X_val, y_train, y_val, X_test, y_test, model, estimator):
    """
    Refits estimator with best hyperparameters on training+validation set and returns refitted estimator.
    """
    best_params = model.best_params_
    print(f"{estimator} best parameters found: {best_params}")

    best_score = model.best_score_
    print(f"{estimator} best score on training+val set: {best_score:.4f}")

    # strip "estimator__" prefix from parameter names
    cleaned_params = {key.replace("estimator__", ""): value for key, value in best_params.items()}
    
    if (estimator == "TimeSeriesForest"):
        final_model = MultivariateClassifier(TimeSeriesForest(**cleaned_params))
    elif (estimator == "TSBF"):
        final_model = MultivariateClassifier(TSBF(**cleaned_params))
    
    final_model.fit(
        np.concatenate((X_train, X_val)),
        np.concatenate((y_train, y_val))
    )
    test_accuracy = final_model.score(X_test, y_test)
    print(f"{estimator} best model's score on test set: {test_accuracy:.4f}")
    
    return final_model


def evaluate(labels, predicted, folder, estimator):
    """
    Saves estimator's confusion matrix as a heatmap figure.
    """
    cm = confusion_matrix(labels, predicted)
    classes = ['A', 'B', 'C', 'D']

    plt.close('all')

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='flare')

    plt.xlabel('Predicted Player Types')
    plt.ylabel('True Player Types')
    plt.title('')
    plt.savefig(f"{folder}cm_{estimator}.png")

    print(f"saved confusion matrix to {os.path.dirname(f)}/cm_{estimator}.png \n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify datasets using pyts models."
    )
    parser.add_argument(
        "-f",
        required=True,
        type=str,
        help="filepath to the dataset to be classified"
    )
    parser.add_argument(
        "-n",
        required=False,
        type=int,
        default=10,
        help="(optional) number of iterations for bayes optimization"
    )
    parser.add_argument(
        "-m",
        required=False,
        type=str,
        nargs='*',
        choices=['all', 'tsf', 'tsbf', 'knn'], # removed knn
        default=['all'],
        help="specify a model to use for classification, or 'all' to run all available models"
    )
    parser.add_argument(
        "-j",
        required=False,
        type=int,
        default=-1,
        help="(optional) number of jobs to run in parallel (default of -1 means use all cores, set to 1 for no parallelization--for debugging)"
    )
    parser.add_argument(
        "-seed",
        required=False,
        type=int,
        default=None,
        help="(optional) number to use as the state for random uniform sampling (if not specified, uses scipy.stats distributions)"
    )
    parser.add_argument(
        "-p",
        required=False,
        type=str,
        default=None,
        help="(for demo purposes) folder of models with predetermined best params as .pkl files"
    )
    args = parser.parse_args()

    f = args.f # filename to time series dataset
    n = args.n
    m = args.m # a list of strings
    j = args.j
    random_state = args.seed
    pkl = args.p

    df = pd.read_csv(f)
    encoded_arr = encode(df)
    data = pad(df, encoded_arr)
    labels = get_labels(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(data, labels)

    score_baseline(X_train, X_val, y_train, y_val, X_test, y_test, random_state)

    
    if pkl is None:
        if 'all' in m or 'tsf' in m:
            bayes_tsf = optimize_tsf(f, X_train, y_train, n, j, random_state)

            toSave = input("Save TimeSeriesForest model as .pkl? [y/n]")
            if toSave == 'y':
                joblib.dump(bayes_tsf, f'{os.path.dirname(f)}/bayes_tsf.pkl')
                print(f"Saved TimeSeriesForest model as {os.path.dirname(f)}/bayes_tsf.pkl")
            else:
                print("okie *thumbs up*")

            final_model_tsf = fit_best(X_train, X_val, y_train, y_val, X_test, y_test, bayes_tsf, 'TimeSeriesForest')

            y_pred_tsf = final_model_tsf.predict(X_test)
            evaluate(y_test, y_pred_tsf, f'{os.path.dirname(f)}/', 'tsf')


        if 'all' in m or 'tsbf' in m:
            bayes_tsbf = optimize_tsbf(f, X_train, y_train, n, j, random_state)

            toSave = input("Save TSBF model as .pkl? [y/n]")
            if toSave == 'y':
                joblib.dump(bayes_tsbf, f'{os.path.dirname(f)}/bayes_TSBF.pkl')
                print(f"Saved TSBF model as {os.path.dirname(f)}/bayes_TSBF.pkl")
            else:
                print("okie *thumbs up*")
            
            final_model_tsbf = fit_best(X_train, X_val, y_train, y_val, X_test, y_test, bayes_tsbf, 'TSBF')
            
            y_pred_tsbf = final_model_tsbf.predict(X_test)
            evaluate(y_test, y_pred_tsbf, f'{os.path.dirname(f)}/', 'TSBF')

    
    # if p option is set, load demo model with predetermined best params and score on test set
    else:
        model_tsf = joblib.load(f'{pkl}bayes_tsf.pkl')
        final_model_tsf = fit_best(X_train, X_val, y_train, y_val, X_test, y_test, model_tsf, 'TimeSeriesForest')
        y_pred_tsf = final_model_tsf.predict(X_test)
        evaluate(y_test, y_pred_tsf, f'{os.path.dirname(f)}/', 'tsf')

        model_tsbf = joblib.load(f'{pkl}bayes_TSBF.pkl')
        final_model_tsbf = fit_best(X_train, X_val, y_train, y_val, X_test, y_test, model_tsbf, 'TSBF')
        y_pred_tsbf = final_model_tsbf.predict(X_test)
        evaluate(y_test, y_pred_tsbf, f'{os.path.dirname(f)}/', 'TSBF')

    print("done :D")


    

