import os 
# Configure CPU cores to avoid performance warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# Read the dataset containing network traffic data (normal and anomaly records)
dataset = pd.read_csv('C:/Users/Ahmed/Koleya/5th Term/Introduction To AI/Section/12 Project/Sample_KDD.csv')

# Separate features and labels
# X contains all columns except the last one (features)
X = dataset.iloc[:, :-1]
# y contains only the last column (labels: 'normal' or 'anomaly')
y = dataset.iloc[:, -1]

# PART A: Split data into training and testing sets
# Using stratified split to maintain the same ratio of normal/anomaly in both sets
# 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Print distribution of normal and anomaly cases in both sets
print(f"Training dataset - Normal: {y_train.value_counts()['normal']}, "
      f"Anomaly: {y_train.value_counts()['anomaly']}")
print(f"Testing dataset - Normal: {y_test.value_counts()['normal']}, "
      f"Anomaly: {y_test.value_counts()['anomaly']}")

# PART B: Train and evaluate a Perceptron classifier
perceptron = Perceptron()
perceptron.fit(X_train, y_train)  # Train the model
y_pred = perceptron.predict(X_test)  # Make predictions
# Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Perceptron Classification Accuracy: {accuracy * 100:.2f}%")

# PART C: Perform K-Means clustering with different k values
k_values = [2, 3, 4, 5]
for k in k_values:
    # Initialize and fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Analyze cluster composition
    print(f"\nK-Means Clustering with k={k}")
    cluster_labels = pd.DataFrame({'Cluster': clusters, 'Class': y})
    
    # Calculate percentages of normal/anomaly records in each cluster
    for cluster in range(k):
        cluster_data = cluster_labels[cluster_labels['Cluster'] == cluster]
        normal_count = cluster_data[cluster_data['Class'] == 'normal'].shape[0]
        anomaly_count = cluster_data[cluster_data['Class'] == 'anomaly'].shape[0]
        total_count = cluster_data.shape[0]
        
        # Calculate and display percentages
        normal_percentage = (normal_count / total_count) * 100
        anomaly_percentage = (anomaly_count / total_count) * 100
        print(f"Cluster {cluster}: Normal: {normal_percentage:.2f}%, "
              f"Anomaly: {anomaly_percentage:.2f}%")