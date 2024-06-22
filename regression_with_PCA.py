import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
train_data = pd.read_csv('DATA_split/DATA_4sn_trn.txt', delimiter='\t', header=None)
val_data = pd.read_csv('DATA_split/DATA_4sn_tst_pred.txt', delimiter='\t', header=None)

# Combine the datasets
combined_data = pd.concat([train_data, val_data], ignore_index=True)
combined_features = combined_data.iloc[:, :-1].values
combined_labels = combined_data.iloc[:, -1].values

# Standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(combined_features)

# Split standardized features and labels into training and validation sets
train_sample_size = train_data.shape[0]
train_features = standardized_features[:train_sample_size, :]
val_features = standardized_features[train_sample_size:, :]
train_labels = combined_labels[:train_sample_size]
val_labels = combined_labels[train_sample_size:]

# Perform PCA on the standardized data
pca = PCA()
combined_features_pca = pca.fit_transform(standardized_features)

# Analyze explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.grid(True)
plt.show()

# Determine the number of components to keep (e.g., 95% of variance explained)
num_components_to_keep = np.argmax(cumulative_explained_variance >= 0.95) + 1

# Adjust combined_features_pca to keep only the desired number of components
combined_features_pca = combined_features_pca[:, :num_components_to_keep]

# Split PCA features into training and validation sets
train_features_pca = combined_features_pca[:train_sample_size, :]
val_features_pca = combined_features_pca[train_sample_size:, :]

# Fit the linear regression model using the PCA-transformed training data
regressor_pca = LinearRegression()
regressor_pca.fit(train_features_pca, train_labels)

# Predict on training data
train_predictions_pca = regressor_pca.predict(train_features_pca)
# Predict on validation data
val_predictions_pca = regressor_pca.predict(val_features_pca)

# Calculate Mean Squared Error (MSE) for PCA-based regression
train_mse_pca = mean_squared_error(train_labels, train_predictions_pca)
val_mse_pca = mean_squared_error(val_labels, val_predictions_pca)

print("PCA-based Regression Training MSE: ", train_mse_pca)
print("PCA-based Regression Validation MSE: ", val_mse_pca)

# Display PCA-based regression model coefficients
print("PCA-based Regression Coefficients:", regressor_pca.coef_)

# Fit the linear regression model using the raw standardized training data
regressor_mlr = LinearRegression()
regressor_mlr.fit(train_features, train_labels)

# Predict on training data
train_predictions_mlr = regressor_mlr.predict(train_features)
# Predict on validation data
val_predictions_mlr = regressor_mlr.predict(val_features)

# Calculate Mean Squared Error (MSE) for raw data regression
train_mse_mlr = mean_squared_error(train_labels, train_predictions_mlr)
val_mse_mlr = mean_squared_error(val_labels, val_predictions_mlr)

print("Raw Data Regression Training MSE: ", train_mse_mlr)
print("Raw Data Regression Validation MSE: ", val_mse_mlr)

# Display raw data regression model coefficients
print("Raw Data Regression Coefficients:", regressor_mlr.coef_)