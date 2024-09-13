import numpy as np

from random import randint
from data_loader import DataLoader
from model import (
    TransformerModel,
    TRANSFORMER_MODEL_PATH,
    TRANSFORMER_VECTORIZE_LAYER_PATH,
)

if __name__ == "__main__":
    # Load the data
    dataloader = DataLoader(
        "../dataset/contracts/code2text/attack_263_decompiled_texts.csv",
        "../dataset/contracts/code2text/normal_12000_decompiled_texts.csv",
    )
    dataloader.form_datasets()

    test_X, test_Y = dataloader.test_X(), dataloader.test_Y()

    # Randomly choose a sample from the test_X
    sample_idx = randint(0, len(test_X))
    test_x = test_X[sample_idx : sample_idx + 1]
    test_y = test_Y[sample_idx : sample_idx + 1]

    # Load the model
    model = TransformerModel(
        saved_model_path=TRANSFORMER_MODEL_PATH,
        saved_vectorize_layer_path=TRANSFORMER_VECTORIZE_LAYER_PATH,
    )

    # Evaluate the model on the sample
    print(model.predict(test_x), test_y)
