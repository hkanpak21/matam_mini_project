import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

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

# Display the shapes of the datasets
print("Train features shape:", train_features_tensor.shape)
print("Train labels shape:", train_labels_tensor.shape)
print("Validation features shape:", val_features_tensor.shape)
print("Validation labels shape:", val_labels_tensor.shape)
print("Test features shape:", test_features_tensor.shape)
