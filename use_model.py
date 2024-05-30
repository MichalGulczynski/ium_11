import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
import sys
import os

def calculate_metrics(result):        
    rmse = np.sqrt(mean_squared_error(result["Real"], result["Predictions"]))
    f1 = f1_score(result["Real"], result["Predictions"], average='macro')
    accuracy = accuracy_score(result["Real"], result["Predictions"])
    
    filename = 'metrics_df.csv'
    if os.path.exists(filename):
        metrics_df = pd.read_csv(filename)
        new_row = pd.DataFrame({'RMSE': [rmse], 'F1 Score': [f1], 'Accuracy': [accuracy]})
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    else:
        metrics_df = pd.DataFrame({'RMSE': [rmse], 'F1 Score': [f1], 'Accuracy': [accuracy]})

    
    metrics_df.to_csv(filename, index=False)

np.set_printoptions(threshold=20)

file_path = 'model.pkl'
with open(file_path, 'rb') as file:
    model = pickle.load(file)
print("Model został wczytany z pliku:", file_path)

test_df = pd.read_csv("docker_test_dataset.csv")

Y_test = test_df[['playlist_genre']]
X_test = test_df.drop(columns='playlist_genre')
Y_test = np.ravel(Y_test)

scaler = StandardScaler()
numeric_columns = X_test.select_dtypes(include=['int', 'float']).columns
X_test_scaled = scaler.fit_transform(X_test[numeric_columns])

Y_pred = model.predict(X_test_scaled)

result = pd.DataFrame({'Predictions': Y_pred, "Real": Y_test})
result.to_csv("spotify_genre_predictions.csv", index=False)

calculate_metrics(result)