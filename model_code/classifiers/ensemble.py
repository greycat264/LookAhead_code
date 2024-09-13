import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from tqdm import tqdm
from pickle import load
from collections import Counter
from itertools import combinations
from catboost import CatBoostClassifier
from model_mlp import mlp_load, mlp_predict, mlp_evaluate

from data_loader import DataLoader

PARAMS = {"feature_set": "all"}


class EnsembleClassifier:
    def __init__(self, classifiers, classifier_names):
        self.classifiers = classifiers

        self.size_ = len(classifiers)
        classifiers_names = []

        for classifier in classifiers:
            classifiers_names.append(classifier["name"])

    def __majority_voting(self, pred_Y):
        return Counter(pred_Y).most_common(1)[0][0]

    def __classifier_predict(self, classifier, val_X):
        match classifier["name"]:
            case "MLP":
                return mlp_predict(classifier["model"], val_X.to_numpy())

            case "LightGBM":
                lgb_pred_Y_prob = classifier["model"].predict(
                    val_X.to_numpy(), num_iteration=classifier["model"].best_iteration
                )
                lgb_pred_Y = [
                    1 if pred_y_prob >= 0.5 else 0 for pred_y_prob in lgb_pred_Y_prob
                ]
                return lgb_pred_Y

            case "CatBoost":
                return classifier["model"].predict(val_X)

            case "XGBoost":
                return (classifier["model"].predict(xgb.DMatrix(val_X)) > 0.5).astype(
                    int
                )

        return classifier["model"].predict(val_X)

    def predict(self, val_X):
        pred_Y = np.array([])

        # Produce predictions for validation set using each classifier
        for classifier in self.classifiers:
            pred_y = self.__classifier_predict(classifier, val_X)
            pred_Y = np.append(pred_Y, pred_y)

        # Combine the predictions via majority voting
        pred_Y = pred_Y.reshape(len(self.classifiers), -1).T
        pred_Y = np.apply_along_axis(self.__majority_voting, 1, pred_Y)

        return pred_Y

    def rmse_evaluate(self, val_X, val_Y):
        pred_Y = self.predict(val_X)
        rmse = np.sqrt(np.mean((val_Y - pred_Y) ** 2))

        return rmse

    def print_classifiers(self, f_zeta=None):
        classifiers = ""

        for classifier in self.classifiers:
            classifiers += classifier["name"] + ", "
        classifiers = classifiers[:-2]

        if f_zeta is not None:
            print(f"Classifiers: {classifiers},\t f(zeta): {f_zeta}")
        else:
            print(classifiers)

    def size(self):
        return self.size_


def __pso_objective_func(ensemble_classifier, val_X, val_Y, classifier_names):
    rmse = ensemble_classifier.rmse_evaluate(val_X, val_Y)
    set_size = ensemble_classifier.size()

    return rmse + set_size


def load_classifiers():
    classifiers = [
        {"name": "Logistic Regression", "model": None},
        {"name": "Decision Trees", "model": None},
        {"name": "Random Forest", "model": None},
        {
            "name": "CatBoost",
            "model": CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                loss_function="MultiClass",
                eval_metric="Accuracy",
                allow_writing_files=False,
            ),
        },
        {"name": "LightGBM", "model": None},
        # {"name": "MLP", "model": None},
        {
            "name": "XGBoost",
            "model": xgb.Booster(),
        },
    ]

    # Get the names of the classifiers
    classifier_names = []
    for classifier in classifiers:
        classifier_names.append(classifier["name"])

    # Logistic Regression
    with open("../models/logistic_regression.pkl", "rb") as f:
        classifiers[0]["model"] = load(f)

    # Decision Trees Classifier
    with open("../models/decision_tree.pkl", "rb") as f:
        classifiers[1]["model"] = load(f)

    # Random Forest Classifier
    with open("../models/random_forest.pkl", "rb") as f:
        classifiers[2]["model"] = load(f)

    # CatBoost Classifier
    classifiers[3]["model"].load_model("../models/catboost.cbm")

    # LightGBM Classifier
    classifiers[4]["model"] = lgb.Booster(model_file="../models/lightgbm.txt")

    # # MLP Classifier
    # classifiers[5]["model"] = mlp_load("../models/mlp.pth")

    # XGBoost Classifier
    classifiers[5]["model"].load_model("../models/xgboost.ubj")

    return classifiers, classifier_names


def load_val_dataset():
    dataloader = DataLoader(
        "../dataset/features/attack_269_new_fund_populated.csv",
        "../dataset/features/normal_13000_new_fund_populated.csv",
    )
    dataloader.form_datasets(
        PARAMS["feature_set"], use_transformer=False, test_set_only=True
    )
    dataloader.standardize()

    return dataloader.test_X(), dataloader.test_Y()


if __name__ == "__main__":
    classifiers, classifier_names = load_classifiers()
    val_X, val_Y = load_val_dataset()

    num_classifiers = len(classifiers)

    # Generate all possible combinations of classifiers
    classifiers_combinations = []
    for i in range(1, num_classifiers + 1):
        classifiers_combinations.extend(list(combinations(range(num_classifiers), i)))

    optimal_obj_value = None
    optimal_classifier = None

    for combination in classifiers_combinations:
        ensemble_classifier = EnsembleClassifier(
            [classifiers[i] for i in combination],
            [classifier_names[i] for i in combination],
        )

        obj_value = __pso_objective_func(
            ensemble_classifier, val_X, val_Y, classifier_names
        )
        ensemble_classifier.print_classifiers(obj_value)

        if optimal_obj_value is None or obj_value < optimal_obj_value:
            optimal_obj_value = obj_value
            optimal_classifier = ensemble_classifier

    print("\nOptimal:")
    optimal_classifier.print_classifiers(optimal_obj_value)
