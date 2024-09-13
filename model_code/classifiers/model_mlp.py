import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def mlp_predict(model, test_X, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert the data to PyTorch tensors
    features_tensor = torch.FloatTensor(test_X)
    labels_tensor = torch.FloatTensor(([0] * len(test_X)).to_numpy()).view(-1, 1)

    # Load the data
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    model.eval()
    y_pred_list = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)

            # Forward
            outputs = model(inputs)

            # Convert the predictions to binary
            predictions = (outputs > 0.5).float()
            y_pred_list.append(predictions.cpu().numpy())

    # Concatenate the predictions
    y_pred = np.concatenate(y_pred_list)
    return y_pred


def mlp_evaluate(model, features, labels, device, batch_size=32):
    # Convert the data to PyTorch tensors
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.FloatTensor(labels).view(-1, 1)

    # Load the data
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    model.eval()
    y_pred_list = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)

            # Forward
            outputs = model(inputs)

            # Convert the predictions to binary
            predictions = (outputs > 0.5).float()
            y_pred_list.append(predictions.cpu().numpy())

    # Concatenate the predictions
    y_pred = np.concatenate(y_pred_list)
    return y_pred


def mlp_train(mlp_epochs, X_train_scaled, y_train, X_test_scaled, y_test):
    X_train_to_model, y_train_to_model = X_train_scaled, y_train

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_to_model.to_numpy())
    y_train_tensor = torch.FloatTensor(y_train_to_model.to_numpy()).view(-1, 1)

    # Load the data
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Create the model
    input_size = X_train_to_model.shape[1]
    hidden_size = 128
    output_size = 1
    print("MLP input size: ", input_size)
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = mlp_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "for MLP training")
    model.to(device)

    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Back propagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1:3}/{num_epochs}")

    y_pred = mlp_evaluate(model, X_test_scaled.to_numpy(), y_test.to_numpy(), device)
    return model, y_pred


def mlp_load(model_path, input_size=43, hidden_size=128, output_size=1):
    model = MLP(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model
