#%%Libraries
import requests
from bs4 import BeautifulSoup as bs
import spotipy
import json
import pandas as pd
#%%Login Spotify API
import config
from spotipy.oauth2 import SpotifyClientCredentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= config.client_id,
                                                           client_secret= config.client_secret))
#%%Creating dataframe from playlist tool
def extractor(playlist_id):
    # create an empty data frame to store the song features
    tracks_df = pd.DataFrame(columns=['name', 'artists', 'id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'])
    
    # set up the initial parameters for the playlist tracks request
    playlist_tracks = sp.playlist_tracks(playlist_id)
    tracks = playlist_tracks['items']
    while playlist_tracks['next']:
        playlist_tracks = sp.next(playlist_tracks)
        tracks.extend(playlist_tracks['items'])
    
    # retrieve the song features for each track and append them to the data frame
    for track in tracks:
        track_id = track['track']['id']
        track_features = sp.audio_features(track_id)[0]
        track_row = {'name': track['track']['name'],
                     'artists': track['track']['artists'][0]['name'],
                     'id': track_id,
                     'danceability': track_features['danceability'],
                     'energy': track_features['energy'],
                     'key': track_features['key'],
                     'loudness': track_features['loudness'],
                     'mode': track_features['mode'],
                     'speechiness': track_features['speechiness'],
                     'acousticness': track_features['acousticness'],
                     'instrumentalness': track_features['instrumentalness'],
                     'liveness': track_features['liveness'],
                     'valence': track_features['valence'],
                     'tempo': track_features['tempo'],
                     'duration_ms': track_features['duration_ms'],
                     'time_signature': track_features['time_signature']}
        tracks_df = tracks_df.append(track_row, ignore_index=True)
    
    return tracks_df


