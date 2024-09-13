import shap
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

RENAME_MAP = {
    "verity": "Verified",
    "avg_token_call": "AvgTokenCall",
    "isErcContract": "TokenContract",
    "fund_from_label_transfer": "FundSourceTransfer",
    "fund_from_label_Bridge": "FundSourceBridge",
    "fund_from_label_common": "FundSourceSafe",
    "fund_from_label_uncommon": "FundSourceAnonymous",
    "txDataLen": "InputDataLength",
    "tokenCallFrequency": "TokenCallProportion",
    "publicFucNumber": "PublicFuncNumber",
    "publicFucFrequency": "PublicFuncProportion",
    "flashloanFuncNumber": "FlashloanNumber",
    "gasUsed": "GasUsed",
    "publicFuncNumber": "PublicFuncNumber",
    "nonce": "Nonce",
    "totalFuncNumber": "TotalFuncNumber",
    "callFlowAnalysisConfidence": "CallFlowAnalysisScore",
}


def get_classification_report(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        target_names=["Benign", "Adversarial"],
        output_dict=True,
    )
    TP = conf_matrix[0][0]
    FP = conf_matrix[1][0]
    FN = conf_matrix[0][1]
    TN = conf_matrix[1][1]
    precision = report["Adversarial"]["precision"]
    # False Positive Rate (FPR) = FP / (FP + TN)
    FPR = FP / (FP + TN)
    # True Positive Rate (TPR) = TP / (TP + FN)
    TPR = report["Adversarial"]["recall"]
    f1_score = report["Adversarial"]["f1-score"]
    auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, TPR, f1_score, FPR


def __write_xgboost_importance(xgboost_classifier, columns, figure_path):
    feature_importance = xgboost_classifier.feature_importances_
    feature_names = columns
    feature_importance_list = list(zip(feature_names, feature_importance))

    # Sort features by importance
    feature_importance_list.sort(key=lambda x: x[1], reverse=True)
    top_features = feature_importance_list[:15]

    # Create a horizontal bar plot of the top 15 features
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    top_feature_names, top_feature_importances = zip(*top_features)
    ax.barh(top_feature_names, top_feature_importances)
    ax.set_xlabel("Feature Importance", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")


def __write_lgb_importance(lgb_classifier, columns, figure_path):
    feature_importances = lgb_classifier.feature_importance(importance_type="split")
    new_columns = columns
    for i, item in enumerate(new_columns):
        if item in RENAME_MAP:
            new_columns[i] = RENAME_MAP[item]
    feature_names = new_columns

    # Sort features by importance
    sorted_feature_importance = sorted(
        zip(feature_names, feature_importances), key=lambda x: x[1]
    )

    # Print all features
    print("\nLightGBM features:")
    print(feature_names)
    print(sorted_feature_importance)

    # Print top 15 features
    print("\nLightGBM top 15 features:")
    top_features = sorted_feature_importance[-15:]
    for feature_name, importance in top_features:
        print(f"{feature_name}: {importance}")

    # Create a horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    top_feature_names, top_feature_importances = zip(*top_features)
    ax.barh(top_feature_names, top_feature_importances)
    ax.set_xlabel("Feature Importance", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")


def __write_catboost_importance(catboost_model, columns, figure_path):
    feature_importances = catboost_model.get_feature_importance()
    feature_names = columns

    # Sort features by importance
    sorted_feature_importance = sorted(
        zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True
    )

    # Print top 15 features
    print("\nCatBoost top 15 features:")
    top_features = sorted_feature_importance[:15]
    for feature_name, importance in top_features:
        print(f"{feature_name}: {importance}")

    # Create a horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    top_feature_names, top_feature_importances = zip(*top_features)

    ax.barh(top_feature_names, top_feature_importances)
    ax.set_xlabel("Feature Importance", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)

    plt.savefig(figure_path, dpi=300, bbox_inches="tight")


def __write_lgb_shap_importance(lgb_classifier, train_X, figure_path):
    t_data_new = train_X.rename(columns=RENAME_MAP)

    explainer = shap.TreeExplainer(lgb_classifier, approximate=True)
    X_sample = t_data_new[:]
    shap_values = explainer.shap_values(X_sample)
    base_value = explainer.expected_value

    plt.figure(dpi=300)
    shap.summary_plot(shap_values, X_sample, alpha=0.4, max_display=15, show=False)
    plt.tight_layout()

    # add suffix to figure path filename
    figure_path = figure_path.replace(".png", "_shap.png")
    plt.savefig(figure_path)


def plot_importance_bar_graphs(
    classifiers, bst, train_X, train_X_columns, xgb_path, lgb_path, catboost_path
):
    print("Creating importance figures.")
    columns = train_X_columns.tolist()
    __write_xgboost_importance(
        classifiers["XGBoost Classifier"], train_X_columns, xgb_path
    )
    __write_lgb_importance(bst, columns, lgb_path)
    __write_lgb_shap_importance(bst, train_X, lgb_path)
    __write_catboost_importance(
        classifiers["CatBoost Classifier"], train_X_columns, catboost_path
    )


def save_xgb_false_predictions(classifiers, test_X, test_Y, test_df, csv_path):
    xgb_predictions = classifiers["XGBoost Classifier"].predict(test_X)
    false_ids = []
    for idx, pred in enumerate(xgb_predictions):
        if pred != test_Y[idx]:
            false_ids.append(idx)
    test_df.iloc[false_ids].to_csv(csv_path, index=False)


def save_lgb_false_predictions(bst, test_X, test_Y, test_df, csv_path):
    lgb_predictions_prob = bst.predict(
        test_X.to_numpy(), num_iteration=bst.best_iteration
    )
    lgb_predictions = [
        1 if pred_prob >= 0.5 else 0 for pred_prob in lgb_predictions_prob
    ]
    false_ids = []
    for idx, pred in enumerate(lgb_predictions):
        if pred != test_Y[idx]:
            false_ids.append(idx)
    test_df.iloc[false_ids].to_csv(csv_path, index=False)
