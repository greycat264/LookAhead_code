import pandas as pd
import numpy as np

from tensorflow import cast, string, bool


class DataLoader:
    def __init__(self, attack_csv_path, benign_csv_path):
        self.attack_csv_path = attack_csv_path
        self.benign_csv_path = benign_csv_path
        self.train_X_arr = []
        self.train_Y_arr = []
        self.test_X_arr = []
        self.test_Y_arr = []
        self.dataset_X_arr = []

    def form_datasets(self):
        attack_df = pd.read_csv(self.attack_csv_path)
        benign_df = pd.read_csv(self.benign_csv_path)

        # Form the dataset for text vectorization
        dataset_df = pd.concat([attack_df, benign_df]).reset_index(drop=True)
        self.dataset_X_arr = dataset_df["contract_text"].to_numpy()

        # Form the training dataset (last 80% of attack contracts, first 80% of benign contracts)
        attack_train_df = attack_df.iloc[int(0.2 * len(attack_df)) :]
        benign_train_df = benign_df.iloc[: int(0.8 * len(benign_df))]

        # Oversample the attack contracts
        attack_train_rows = np.arange(len(attack_train_df))
        attack_train_choices = np.random.choice(attack_train_rows, len(benign_train_df))
        attack_train_df = attack_train_df.iloc[attack_train_choices]
        print("Attack contracts oversampled to: ", len(attack_train_df))

        train_df = pd.concat([attack_train_df, benign_train_df]).reset_index(drop=True)
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        self.train_X_arr = train_df["contract_text"].to_numpy()
        self.train_Y_arr = train_df["contract_type"].to_numpy()

        # Form the test dataset (first 20% of attack contracts, last 20% of benign contracts)
        attack_test_df = attack_df.iloc[: int(0.2 * len(attack_df))]
        benign_test_df = benign_df.iloc[int(0.8 * len(benign_df)) :]

        test_df = pd.concat([attack_test_df, benign_test_df]).reset_index(drop=True)

        self.test_X_arr = test_df["contract_text"].to_numpy()
        self.test_Y_arr = test_df["contract_type"].to_numpy()

    def print_counts(self):
        print("Train count: ", len(self.train_Y_arr))
        print("Test count: ", len(self.test_Y_arr))

    def train_X(self):
        assert self.train_X_arr is not None, "Call form_datasets() first"
        return self.train_X_arr

    def train_Y(self):
        assert self.train_Y_arr is not None, "Call form_datasets() first"
        return self.train_Y_arr

    def test_X(self):
        assert self.test_X_arr is not None, "Call form_datasets() first"
        return self.test_X_arr

    def test_Y(self):
        assert self.test_Y_arr is not None, "Call form_datasets() first"
        return self.test_Y_arr

    def dataset_X(self):
        assert self.dataset_X_arr is not None, "Call form_datasets() first"
        return self.dataset_X_arr
