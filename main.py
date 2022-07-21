# Description: Build a song recommendation engine from a large csv table of top songs

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings

df_old = pd.read_csv('top10s.csv', sep=',', encoding='latin-1')
df_new = pd.read_csv('spotify_dataset.csv', sep=',', encoding='latin-1')

# Renaming the all the important columns to be the same for both datasets
df_new.rename(columns={"Song Name": "title"}, inplace=True)
df_new.rename(columns={"Artist": "artist"}, inplace=True)
df_new.rename(columns={"Genre": "top genre"}, inplace=True)
df_new.rename(columns={"Energy": "nrgy"}, inplace=True)
df_new.rename(columns={"Liveness": "live"}, inplace=True)

# Multiplying the 'nrgy' and 'live' column of the new dataset by 100 to be a whole number
warnings.filterwarnings("ignore")
for i in range(0, len(df_new['title'])):
    # Some parts of the csv file had blank information, so we skip over that
    if i in {35, 163, 464, 530, 636, 654, 750, 784, 876, 1140, 1538}:
        continue
    else:
        df_new['nrgy'][i] = round(float(df_new['nrgy'][i]) * 100)
        df_new['live'][i] = round(float(df_new['live'][i]) * 100)

# Since we had two csv files, we concatenate all the important columns
df_final = pd.concat([df_new, df_old], ignore_index=True)

# Adding a song_id column to the new dataset
df_final["song_id"] = range(0, 2159)

# Create a list of all the important columns that we use to find similar songs
columns = ['artist', 'top genre', 'nrgy', 'live', 'title']

# Check for any missing values
df_final[columns].isnull().values.any()


# Create a function to combine all the old song values of the important columns and values into a single string
def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(data['artist'][i] + ' ' + data['top genre'][i] + ' ' + str(data['nrgy'][i]) +
                                  ' ' + str(data['live'][i]) + ' ' + data['title'][i])
    return important_features


# Add the new values of the important columns into a new column in the data
df_final['important_features'] = get_important_features(df_final)

# Convert the text to a matrix of token counts
cm = CountVectorizer().fit_transform(df_final['important_features'])

# Get the cosine similarity from the count matrix
cs = cosine_similarity(cm)

# Print the cosine_similarity
# print(cs)

# Get the title of the song that the user likes
title = input("Please choose a song: ")

# Get the song's id
song_id = df_final[df_final.title == title]['song_id'].values[0]

# Create a list of enumerations for the similarity scores
scores = list(enumerate(cs[song_id]))

# Sort the list to show the most similar songs first and least similar songs last
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
sorted_scores = sorted_scores[1:]

# Print the first seven items, or the seven songs most closely related to the chosen song.
j = 0
print('The top most recommended songs to "' + title + '" are:')

for items in sorted_scores:
    song_title = df_final[df_final.song_id == items[0]]['title'].values[0]
    artist_title = df_final[df_final.song_id == items[0]]['artist'].values[0]
    if j != 0:
        print(j, song_title + " by " + artist_title)
    j = j + 1
    if j > 7:
        break
