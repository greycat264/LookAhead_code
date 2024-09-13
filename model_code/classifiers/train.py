import torch.nn as nn
import lightgbm as lgb

from tqdm import tqdm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_loader import DataLoader as DatasetLoader
from model_mlp import mlp_train
from model_helpers import (
    plot_importance_bar_graphs,
    save_xgb_false_predictions,
    save_lgb_false_predictions,
)
from train_helpers import save_classification_reports, save_models

# Output files
REPORT_FILE_PATH = "../results/classification_report.txt"
XGB_FALSE_PREDICTIONS_PATH = "../results/xgb_false_predictions.csv"
XGB_IMPORTANCE_FIGURE_PATH = "../results/top_feature_importance_xgb.png"

LGB_IMPORTANCE_FIGURE_PATH = "../results/top_feature_importance_lgb.png"
LGB_FALSE_PREDICTIONS_PATH = "../results/lgb_false_predictions.csv"
CATBOOST_IMPORTANCE_FIGURE_PATH = "../results/top_feature_importance_cat.png"

# Parameters
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_error",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}

PARAMS = {
    "feature_set": "all",  # One of FEATURE_SETS
    "oversample": True,  # Balance the dataset
    "write_importance": True,  # Output importance figures
    "standardize": True,  # Standardize the data
    "lightgbm_epochs": 100,  # Number of epochs for LightGBM
    "mlp_epochs": 10,  # Number of epochs for MLP
}


if __name__ == "__main__":
    # Load the data
    dataloader = DatasetLoader(
        "../dataset/features/attack_269_new_fund_populated.csv",
        "../dataset/features/normal_13000_new_fund_populated.csv",
    )
    dataloader.form_datasets(PARAMS["feature_set"], use_transformer=False)
    dataloader.print_counts()

    if PARAMS["oversample"]:
        dataloader.oversample()

    if PARAMS["standardize"]:
        dataloader.standardize()

    # Get the dataset
    train_X, train_Y = dataloader.train_X(), dataloader.train_Y()
    test_X, test_Y = dataloader.test_X(), dataloader.test_Y()
    train_XY_lgbm = dataloader.train_XY_lgbm()
    test_df = dataloader.test_DF()

    # Create the models
    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=0),
        "Decision Trees Classifier": DecisionTreeClassifier(
            criterion="gini", random_state=0
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, criterion="gini", random_state=0
        ),
        "XGBoost Classifier": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
        ),
        "CatBoost Classifier": CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            allow_writing_files=False,
        ),
    }

    classifiers_predictions = []

    # Train and evaluate the classifiers
    print("Now training and evaluating the classifiers.")
    for i, classifier in enumerate(tqdm(classifiers.values())):
        classifier.fit(train_X, train_Y)

        pred_Y = classifier.predict(test_X)
        classifiers_predictions.append(pred_Y)

    # Train and evaluate the LightGBM classifier
    print("Now training and evaluating the LightGBM classifier.")
    bst = lgb.train(LIGHTGBM_PARAMS, train_XY_lgbm, PARAMS["lightgbm_epochs"])

    lgb_pred_Y_prob = bst.predict(test_X.to_numpy(), num_iteration=bst.best_iteration)
    lgb_pred_Y = [1 if pred_y_prob >= 0.5 else 0 for pred_y_prob in lgb_pred_Y_prob]

    # Train and evaluate the MLP classifier
    print("Now training and evaluating the MLP classifiers.")
    mlp, mlp_pred_Y = mlp_train(PARAMS["mlp_epochs"], train_X, train_Y, test_X, test_Y)

    # Write evaluation results to the log file
    save_classification_reports(
        classifiers,
        classifiers_predictions,
        lgb_pred_Y,
        mlp_pred_Y,
        test_Y,
        PARAMS,
        REPORT_FILE_PATH,
    )

    # Plot the importance figures if enabled
    if PARAMS["write_importance"]:
        plot_importance_bar_graphs(
            classifiers,
            bst,
            train_X,
            train_X.columns,
            XGB_IMPORTANCE_FIGURE_PATH,
            LGB_IMPORTANCE_FIGURE_PATH,
            CATBOOST_IMPORTANCE_FIGURE_PATH,
        )

    # Output false predictions
    save_xgb_false_predictions(
        classifiers, test_X, test_Y, test_df, XGB_FALSE_PREDICTIONS_PATH
    )
    save_lgb_false_predictions(bst, test_X, test_Y, test_df, LGB_FALSE_PREDICTIONS_PATH)

    # Save trained models
    save_models(classifiers, mlp, bst)
