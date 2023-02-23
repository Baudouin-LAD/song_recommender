#%%Libraries
import spotipy
import pandas as pd
#%%Login Spotify API
import config
from spotipy.oauth2 import SpotifyClientCredentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= config.client_id,
                                                           client_secret= config.client_secret))
#%%Creating dataframe from playlist tool
def playlist_songs(url):
 # Get playlist ID from URL
    playlist_id = url.split("/")[-1]

    # Get playlist tracks
    track_uris = []
    offset = 0
    while True:
        results = sp.playlist_tracks(playlist_id, offset=offset)
        tracks = results['items']
        if not tracks:
            break
        track_uris += [track['track']['uri'] for track in tracks]
        offset += len(tracks)

    return track_uris

def song_features(track_uris):
    track_features = [sp.audio_features(track) for track in track_uris]
    return track_features
    
def data_generator(features):
    columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature']
    song_features_df = pd.DataFrame(columns=columns)
    for track in features:
        track_uri = track[0]['uri']
        danceability = track[0]['danceability']
        energy = track[0]['energy']
        key = track[0]['key']
        loudness = track[0]['loudness']
        mode = track[0]['mode']
        speechiness = track[0]['speechiness']
        acousticness = track[0]['acousticness']
        instrumentalness = track[0]['instrumentalness']
        liveness = track[0]['liveness']
        valence = track[0]['valence']
        tempo = track[0]['tempo']
        type_ = track[0]['type']
        id_ = track[0]['id']
        track_href = track[0]['track_href']
        analysis_url = track[0]['analysis_url']
        duration_ms = track[0]['duration_ms']
        time_signature = track[0]['time_signature']
    
        # append a new row to the DataFrame with the track features
        song_features_df = song_features_df.append({'uri': track_uri, 'danceability': danceability, 'energy': energy, 'key': key, 'loudness': loudness, 'mode': mode, 'speechiness': speechiness, 'acousticness': acousticness, 'instrumentalness': instrumentalness, 'liveness': liveness, 'valence': valence, 'tempo': tempo, 'type': type_, 'id': id_, 'track_href': track_href, 'analysis_url': analysis_url, 'duration_ms': duration_ms, 'time_signature': time_signature}, ignore_index=True)
    return song_features_df
   


        
    
    
tracks_p=playlist_songs("https://open.spotify.com/playlist/5tbtgaIAg9YI17pgxb0TXB")
features = song_features(tracks_p)
data=data_generator(features)
data.set_index('id', inplace=True)


#%%Slicing Dataframe before scaling
data_to_scale = data.iloc[:,0:11]
#%%Scaling Data

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
scaler.fit(data_to_scale)
data_scaled = scaler.transform(data_to_scale)
data_scaled_df = pd.DataFrame(data_scaled, columns = data_to_scale.columns)


#%%Saving Scaler using pickle
import pickle


with open("/Users/Baudouin/Ironhack/song_recommender/scaler.pickle", "wb") as f:
    pickle.dump(scaler,f)

#%% choosing K
K = range(2, 40)
inertia = []

for k in K:
    print("Training a K-Means model with {} clusters! ".format(k))
    print()
    kmeans = KMeans(n_clusters=k,
                    random_state=1234)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('inertia')
plt.xticks(np.arange(min(K), max(K)+1, 1.0))
plt.title('Elbow Method showing the optimal k')
#We choose K=10`

#%%fitting  k means cluster
kmeans = KMeans(n_clusters=10, random_state=1234)
kmeans.fit(data_scaled)

from yellowbrick.cluster import SilhouetteVisualizer
model = KMeans(10, random_state=42)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(data_scaled)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure
#It seems the model is good enough

#%%Assigning original data to clusters
clusters = kmeans.predict(data_scaled)
pd.Series(clusters).value_counts().sort_index()
data["cluster"]=clusters

#%%Saving model using pickle
with open("/Users/Baudouin/Ironhack/song_recommender/model.pickle", "wb") as f:
    pickle.dump(model,f)








