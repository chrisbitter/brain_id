import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn import metrics
import argparse
import datetime
import pickle

from features import FeatureExtractor
from models import BaselineClf, MLModel, RiemannMDM
from train.dataset import EEGDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/datasets/csv/",
        help="Path to the dataset directory.",
    )
    parser.add_argument(  # accept list of subject sub-01 sub-02 ...
        "--subjects",
        type=str,
        nargs="+",
        default=["sub-01"],
        help="List of subject (e.g., sub-01 sub-02 ...).",
    )
    parser.add_argument(
        "--n_times",
        type=int,
        default=1,
        help="Number of times to run the training.",
    )
    return parser.parse_args()


def evaluate(model_dir, X_test, y_test):
    """Evaluate the model on the test set."""

    # load the model
    print(f"Loading model from {model_dir}...")
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    # get model predictions
    y_pred = model.predict(X_test)

    # get all relevant metrics for classification
    accuracy = metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    classification_report = metrics.classification_report(y_test, y_pred)

    # save the results
    results_file = os.path.join(model_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix))
        f.write("\nClassification Report:\n")
        f.write(classification_report)


def optimize(X_train, y_train, param_list):
    """hyper parameter optimization using grid search.

    Args:
        X_train (np.ndarray): training data
        y_train (np.ndarray): training labels
        param_list (dict): dictionary of parameters to optimize

    Returns:
        dict: best parameters found during optimization
    """

    # create the model
    model = MLModel()

    # create the parameter grid
    param_grid = {"model__" + k: v for k, v in param_list.items()}

    # perform grid search
    scoring = ["accuracy", "balanced_accuracy", "f1_micro", "f1_macro"]
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring=scoring, refit="f1_macro", n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train, y_train)

    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    return grid_search.best_params_


def train_model(X_train, y_train, hpo_parameters):

    # hyperparameter optimization
    best_params = optimize(X_train, y_train, hpo_parameters)

    # load best hyper-parameters
    model = MLModel(
        feature_kwargs=best_params.get("feature_kwargs", {}), 
        model_kwargs=best_params.get("model_kwargs", {})
    )

    # fit final model
    model.fit(X_train, y_train)
    return model


def main():
    # parse command line arguments
    args = parse_args()

    # create the dataset
    ds = EEGDataset(args.data_path, args.subjects, test_split=0.2)
    X_train, y_train, X_test, y_test = ds.get_dataset(
        time_window=0.5, overlap=0.50, balance=1.0
    )
    print("dataset classes: ", np.unique(y_train, return_counts=True))

    # train & optimize the model
    hpo_params = {
        "feature_kwargs__pca_components": [0.90, 0.80, 0.60],
        "model_kwargs__kernel": ["linear", "rbf"],
        "model_kwargs__C": [0.1, 1.0, 10.0, 100.0],
    }
    model = train_model(X_train, y_train, hpo_parameters=hpo_params)

    # save results
    class_name = model.model.__class__.__name__
    ckpt_dir = "checkpoints/train_ml/"
    model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id += f"_{class_name}"
    model_dir = os.path.join(ckpt_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # save the model
    print(f"Saving model to {model_dir}...")
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # evaluate the model on the test set
    # evaluate(model_dir, X_test, y_test)


if __name__ == "__main__":
    main()
