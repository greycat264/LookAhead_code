from data_loader import DataLoader
from model import TransformerModel

if __name__ == "__main__":
    # Load the data
    dataloader = DataLoader(
        "../dataset/contracts/code2text/attack_263_decompiled_texts.csv",
        "../dataset/contracts/code2text/normal_12000_decompiled_texts.csv",
    )
    dataloader.form_datasets()
    dataloader.print_counts()

    train_X, train_Y = dataloader.train_X(), dataloader.train_Y()
    test_X, test_Y = dataloader.test_X(), dataloader.test_Y()
    dataset_X = dataloader.dataset_X()

    # Create the model
    model = TransformerModel(dataset_X=dataset_X)

    # Train the model (early stopping is enabled)
    model.train(10, train_X, train_Y, False, False, True, test_X, test_Y)
