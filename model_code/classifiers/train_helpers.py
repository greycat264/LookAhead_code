import torch
import pickle

from model_helpers import get_classification_report

def save_classification_reports(classifiers, classifiers_pred_Y, lgb_pred_Y, mlp_pred_Y, test_Y, params, report_path):
		print("Writing results to log file.")
		with open(report_path, "w+") as f:
				f.write(f"Oversampling enabled => {params["oversample"]}\n")
				f.write(f"Feature set => {params["feature_set"]}\n")
				f.write(f"Standardization enabled => {params["standardize"]}\n")
				f.write(
						"Model Name                              acc    prec   recall f1     FPR    \n"
				)

				# Classifier results
				for classifier_idx in range(len(classifiers.values())):
						accuracy, precision, tpr, f1_score, fpr = get_classification_report(
								test_Y, classifiers_pred_Y[classifier_idx]
						)
						
						f.write(f"{list(classifiers.keys())[classifier_idx]:<40}")
						f.write(f"{accuracy:.4f} {precision:.4f} {tpr:.4f} {f1_score:.4f} {fpr:.4f}\n")
				
				# LightGBM results
				accuracy, precision, tpr, f1_score, fpr = get_classification_report(test_Y, lgb_pred_Y)
				f.write(f"{"LightGBM":<40}")
				f.write(f"{accuracy:.4f} {precision:.4f} {tpr:.4f} {f1_score:.4f} {fpr:.4f}\n")
				
				# MLP results
				accuracy, precision, tpr, f1_score, fpr = get_classification_report(test_Y, mlp_pred_Y)
				f.write(f"{"MLP":<40}")
				f.write(f"{accuracy:.4f} {precision:.4f} {tpr:.4f} {f1_score:.4f} {fpr:.4f}\n")
				
				f.write("==========================================================================\n")

def save_models(classifiers, mlp, bst):
    with open("../models/logistic_regression.pkl", "wb") as f:
        pickle.dump(classifiers["Logistic Regression"], f, protocol=5)
    with open("../models/decision_tree.pkl", "wb") as f:
        pickle.dump(classifiers["Decision Trees Classifier"], f, protocol=5)
    with open("../models/random_forest.pkl", "wb") as f:
        pickle.dump(classifiers["Random Forest Classifier"], f, protocol=5)
    torch.save(mlp.state_dict(), "../models/mlp.pth")
    bst.save_model("../models/lightgbm.txt")
    classifiers["XGBoost Classifier"].save_model("../models/xgboost.ubj")
    classifiers["CatBoost Classifier"].save_model("../models/catboost.cbm")
    print("\nModels saved.")
