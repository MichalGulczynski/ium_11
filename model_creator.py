import pandas as pd
import os
import numpy as np
import shutil
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn. preprocessing import LabelEncoder
import pickle

def datasets_preparation():
       df_1 = pd.read_csv("spotify_songs.csv")
       df_2 = pd.read_csv("Spotify_Dataset.csv", sep=";")

       df_1 = df_1.dropna()
       df_2 = df_2.dropna()
       df_2 = df_2.rename(columns={'Title': 'track_name'})

       columns_to_remove_df_1 = ['track_id', 'track_album_id', 'track_album_name', 'track_album_release_date',
                            'playlist_id', 'playlist_subgenre']
       columns_to_remove_df_2 = ['Date','# of Artist', 'Artist (Ind.)', '# of Nationality',
              'Nationality', 'Continent', 'Points (Total)',
              'Points (Ind for each Artist/Nat)', 'id', 'Song URL']

       df_1 = df_1.drop(columns=columns_to_remove_df_1)
       df_2 = df_2.drop(columns=columns_to_remove_df_2)
       df_1 = df_1.drop_duplicates(subset=['track_name'])
       df_2 = df_2.drop_duplicates(subset=['track_name'])

       le = LabelEncoder()

       unique_names_df2 = df_2['track_name'].unique()
       diff_df = df_1[~df_1['track_name'].isin(unique_names_df2)]
       diff_df = diff_df.iloc[:10000]

       #diff_df = pd.concat([diff_df, df_1.iloc[:20]], ignore_index=True)
       diff_df['track_artist'] = le.fit_transform(diff_df.track_artist)
       diff_df['playlist_name'] = le.fit_transform(diff_df.playlist_name)
       diff_df['playlist_genre'] = le.fit_transform(diff_df.playlist_genre)

       #df_1 = df_1.iloc[20:]
       
       if "docker_test_dataset.csv" not in os.listdir():
              diff_df.to_csv("docker_test_dataset.csv", index=False)

       result_df = pd.merge(df_1, df_2, on='track_name', how='inner') 
       result_df = result_df.drop_duplicates(subset=['track_name'])
       columns_to_remove_result_df = ['Rank', 'Artists', 'Danceability', 'Energy', 'Loudness',
       'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence']
       result_df = result_df.drop(columns=columns_to_remove_result_df)

       result_df['track_artist'] = le.fit_transform(result_df.track_artist)
       result_df['playlist_name'] = le.fit_transform(result_df.playlist_name)
       result_df['playlist_genre'] = le.fit_transform(result_df.playlist_genre)
       
       return result_df

result_df = datasets_preparation()
Y = result_df[['playlist_genre']]
X = result_df.drop(columns='playlist_genre')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=float(sys.argv[1]), random_state=42)


Y_train = np.ravel(Y_train)
Y_test = np.ravel(Y_test)

scaler = StandardScaler()
numeric_columns = X_train.select_dtypes(include=['int', 'float']).columns
X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled = scaler.transform(X_test[numeric_columns])

model = LogisticRegression(max_iter=int(sys.argv[2]))
model.fit(X_train_scaled, Y_train)


Y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

file_path = 'model.pkl'

if os.path.exists(file_path):
    os.remove(file_path)

if file_path not in os.listdir("./"):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

print("Model został zapisany do pliku:", file_path)






