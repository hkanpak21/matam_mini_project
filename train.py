import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path, delimiter='\t', header=None)
    return data

# Split the data into features and labels
def split_features_labels(data):
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    return features, labels

# Load the training and validation data
train_data = load_data('DATA_split/DATA_4sn_trn.txt')
val_data = load_data('DATA_split/DATA_4sn_val.txt')
test_data = load_data('DATA_split/DATA_4sn_tst.txt')

# Split the training and validation data into features and labels
train_features, train_labels = split_features_labels(train_data)
val_features, val_labels = split_features_labels(val_data)

# Convert the data to PyTorch tensors
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)

# Create TensorDatasets
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# The test data only has features, and the labels are all zeros
test_features_tensor = torch.tensor(test_data.values[:, :-1], dtype=torch.float32)

# Define the neural network model
class EarthquakePredictor(nn.Module):
    def __init__(self):
        super(EarthquakePredictor, self).__init__()
        self.fc1 = nn.Linear(200, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the model, loss function, and optimizer
model = EarthquakePredictor()
criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}')

# Prediction on the test set
model.eval()
with torch.no_grad():
    test_predictions = model(test_features_tensor).squeeze().numpy()

# Ensure compatibility with the original data type
test_predictions = test_predictions.astype(float)

# Save predictions to file
test_data.iloc[:, -1] = test_predictions
test_data.to_csv('DATA_split/DATA_4sn_tst_pred.txt', sep='\t', header=None, index=False)

print("Predictions saved to DATA_split/DATA_4sn_tst_pred.txt")
