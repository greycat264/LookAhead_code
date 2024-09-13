import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
from collections import Counter
from features import (
    CREATOR_ATTRIBUTE_FEATURES,
    TRANSACTION_DATA_FEATURES,
    TRANSACTION_FEATURES,
    CONTRACT_FEATURES,
    COMBINED_FEATURES,
    ALL_FEATURES,
)

# Turn off warnings
pd.options.mode.chained_assignment = None

# Import the transformer model
transformer_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../transformer")
)
sys.path.insert(0, transformer_dir)

from model import (
    TransformerModel,
    TRANSFORMER_MODEL_PATH,
    TRANSFORMER_VECTORIZE_LAYER_PATH,
)

FEATURE_SETS = {
    "transaction_features": TRANSACTION_FEATURES,
    "contract_features": CONTRACT_FEATURES,
    "combined": COMBINED_FEATURES,
    "all": ALL_FEATURES,
    "custom": CREATOR_ATTRIBUTE_FEATURES
    + TRANSACTION_DATA_FEATURES
    + CONTRACT_FEATURES,
}

CONTRACT_TYPE = {"adversarial": 1, "benign": 0}


class DataLoader:
    def __init__(self, attack_csv_path, benign_csv_path):
        self.attack_csv_path = attack_csv_path
        self.benign_csv_path = benign_csv_path
        self.train_X_df = None
        self.train_Y_df = None
        self.test_X_df = None
        self.test_Y_df = None
        self.test_df = None

    def __format_features(self, data_frame, feature_arr):
        features = feature_arr + ["tx_hash"]
        data_frame = data_frame[features]

        if "verity" in data_frame.columns:
            data_frame["verity"] = data_frame["verity"].astype(int)

        if "nonce" in data_frame.columns:
            data_frame["nonce"] = [
                int(item, 16) for item in data_frame["nonce"].tolist()
            ]

        if "value" in data_frame.columns:
            data_frame["value"] = [
                int(item, 16) for item in data_frame["value"].tolist()
            ]
            data_frame["value"] = [
                1 if value > 0 else 0 for value in data_frame["value"].tolist()
            ]

        if "fund_from_label" in data_frame.columns:
            data_frame = pd.get_dummies(
                data_frame, columns=["fund_from_label"], prefix="fund_from_label"
            )

        return data_frame

    def __populate_cfa_confidence(self, data_frame, texts_csv_path):
        texts_df = pd.read_csv(texts_csv_path)
        texts_df = texts_df.drop("contract_type", axis=1)

        # Merge contract_texts df with current df according to contract_address
        data_frame = pd.merge(data_frame, texts_df, on="contract_address", how="inner")

        # Unmatched contracts should get a confidence of 0
        data_frame["callFlowAnalysisConfidence"] = 0.0

        # Use the transformer model to predict the confidence
        transformer = TransformerModel(
            saved_model_path=TRANSFORMER_MODEL_PATH,
            saved_vectorize_layer_path=TRANSFORMER_VECTORIZE_LAYER_PATH,
        )

        data_frame["callFlowAnalysisConfidence"] = transformer.predict(
            data_frame["contract_text"]
        )

        return data_frame

    def form_datasets(
        self, feature_set_choice="all", use_transformer=False, test_set_only=False
    ):
        self.test_set_only = test_set_only

        attack_df = pd.read_csv(self.attack_csv_path)
        normal_df = pd.read_csv(self.benign_csv_path)
        normal_df = pd.concat([normal_df] * 10, ignore_index=True)

        attack_df.loc[attack_df["fund_from_label"] == "Seed", "fund_from_label"] = (
            "unknown"
        )
        normal_df.loc[normal_df["fund_from_label"] == "Seed", "fund_from_label"] = (
            "unknown"
        )

        # Transformer inference to produce callFlowAnalysisConfidence if required
        if use_transformer:
            attack_df = self.__populate_cfa_confidence(
                attack_df,
                "../dataset/contracts/code2text/attack_263_decompiled_texts.csv",
            )
            normal_df = self.__populate_cfa_confidence(
                normal_df,
                "../dataset/contracts/code2text/normal_12000_decompiled_texts.csv",
            )

        # Format the features
        feature_array = FEATURE_SETS[feature_set_choice]
        attack_df_formatted = self.__format_features(attack_df, feature_array)
        normal_df_formatted = self.__format_features(normal_df, feature_array)

        # Assign labels
        attack_df_formatted["tag"] = [CONTRACT_TYPE["adversarial"]] * len(
            attack_df_formatted
        )
        normal_df_formatted["tag"] = [CONTRACT_TYPE["benign"]] * len(
            normal_df_formatted
        )

        attack_num, normal_num = len(attack_df_formatted), len(normal_df_formatted)

        # Form the training dataset (last 80% of attack contracts, first 80% of benign contracts)
        train_df, attack_train_df, benign_train_df = None, None, None
        attack_train_df = attack_df_formatted.iloc[int(attack_num * 0.2) :]
        benign_train_df = normal_df_formatted[: int(normal_num * 0.8)]
        train_df = pd.concat([attack_train_df, benign_train_df]).reset_index(drop=True)

        # Form the test dataset (first 20% of attack contracts, last 20% of benign contracts)
        attack_test_df = attack_df_formatted.iloc[: int(attack_num * 0.2)]
        benign_test_df = normal_df_formatted.iloc[int(normal_num * 0.8) :]
        self.test_df = pd.concat([attack_test_df, benign_test_df]).reset_index(
            drop=True
        )

        # Extract train and test labels
        self.train_X_df = train_df.drop("tag", axis=1)
        self.train_X_df = self.train_X_df.drop("tx_hash", axis=1)
        self.train_Y_df = train_df["tag"]

        self.test_X_df = self.test_df.drop("tag", axis=1)
        self.test_X_df = self.test_X_df.drop("tx_hash", axis=1)
        self.test_Y_df = self.test_df["tag"]

    def oversample(self):
        assert self.train_X_df is not None, "Call form_datasets() first"
        assert self.train_Y_df is not None, "Call form_datasets() first"

        print("Oversampling the training dataset:")

        # Print the class distribution
        counter = Counter(self.train_Y_df)
        print(counter)

        oversample = ADASYN(sampling_strategy="auto", random_state=42)
        self.train_X_df, self.train_Y_df = oversample.fit_resample(
            self.train_X_df, self.train_Y_df
        )

        # Print the new class distribution
        counter = Counter(self.train_Y_df)
        print(counter)

    def standardize(self):
        scaler = StandardScaler()

        train_X = scaler.fit_transform(self.train_X_df)
        if not self.test_set_only:
            self.train_X_df = pd.DataFrame(train_X, columns=self.train_X_df.columns)

        test_X = scaler.transform(self.test_X_df)
        self.test_X_df = pd.DataFrame(test_X, columns=self.test_X_df.columns)

        print("Dataset standardized.")

    def train_XY_lgbm(self):
        assert self.train_X_df is not None, "Call form_datasets() first"
        assert self.train_Y_df is not None, "Call form_datasets() first"
        assert (
            self.test_set_only is False
        ), "Call form_datasets() with test_set_only=False"

        return lgb.Dataset(self.train_X_df, label=self.train_Y_df)

    def print_counts(self):
        print("Train count: ", len(self.train_Y_df))
        print("Test count: ", len(self.test_Y_df))

    def train_X(self):
        assert self.train_X_df is not None, "Call form_datasets() first"
        assert (
            self.test_set_only is False
        ), "Call form_datasets() with test_set_only=False"

        return self.train_X_df

    def train_Y(self):
        assert self.train_Y_df is not None, "Call form_datasets() first"
        assert (
            self.test_set_only is False
        ), "Call form_datasets() with test_set_only=False"

        return self.train_Y_df

    def test_X(self):
        assert self.test_X_df is not None, "Call form_datasets() first"
        return self.test_X_df

    def test_Y(self):
        assert self.test_Y_df is not None, "Call form_datasets() first"
        return self.test_Y_df

    def test_DF(self):
        assert self.test_df is not None, "Call form_datasets() first"
        return self.test_df
