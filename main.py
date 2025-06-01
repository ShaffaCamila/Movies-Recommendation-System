# %% [markdown]
# # Proyek Sistem Rekomendasi: [Movie Recommender System Dataset]
# - **Nama:** [Naia Shaffa Camila]
# - **Email:** [naiashaffa@gmail.com]
# - **ID Dicoding:** [MC015D5X2145]

# %% [markdown]
# ### Import Library

# %%
import os
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')

import pandas as pd
import numpy as np 
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

# %% [markdown]
# ### Data Load

# %%
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files('gargmanas/movierecommenderdataset', path='data/', unzip=True)

# %% [markdown]
# ### EDA

# %%
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# %%
movies.info()

# %%
count_no_genres = movies[movies['genres'].str.contains(r'\(no genres listed\)')].shape[0]
print(f"Jumlah film dengan '(no genres listed)': {count_no_genres}")


# %%
movies

# %%
movies.duplicated(subset=['movieId']).sum()

# %%
movies.isnull().sum()

# %% [markdown]
# Insights : 
# - Terdapat 9742 baris (records) dalam dataset movies.
# - Terdapat 3 kolom yaitu: movieId, title, genres
# - Tidak terdapat duplikat berdasarkan banyak baris dalam DataFrame movies yang memiliki movieId yang sama.
# - Tidak memiliki missing value.

# %%
ratings.info()

# %%
ratings

# %%
ratings.describe()

# %%
ratings.duplicated(subset=['userId', 'movieId']).sum()

# %%
ratings.isnull().sum()

# %% [markdown]
# Insights :
# - Terdapat 100.836 baris (records) dalam dataset
# - Terdapat 4 kolom yaitu: userId, moviesId, rating, dan timestamp
# - Rating memiliki nilai terendah 0.5 dan tertinggi 5.
# - Tidak terdapat duplikat berdasarkan banyak baris dalam DataFrame ratings yang memiliki userId dan movieId yang sama.
# - Tidak memiliki missing value.

# %%
print("Jumlah user dalam dataset: ", ratings.userId.nunique())
print("-"*50)

# Number of Movies in the dataset:
print("Jumlah film dalam dataset:", movies.title.nunique())
print("-"*71)

# Unique of Rating points in the dataset:
print("Nilai unik pada kolom rating:", ratings.rating.unique())

# %%
# Visualisasi distribusi genre film
genre_count = movies['genres'].str.split('|', expand=True).stack().value_counts()
plt.figure(figsize=(10,6))
sns.barplot(x=genre_count.values, y=genre_count.index, palette='muted')
plt.title('Distribusi Genre Film')
plt.xlabel('Jumlah Film')
plt.ylabel('Genre')
plt.show()

# %%
# Visualisasi distribusi rating
plt.figure(figsize=(10,6))
sns.countplot(x='rating', data=ratings, palette='viridis')
plt.title('Distribusi Rating Film')
plt.xlabel('Rating')
plt.ylabel('Jumlah Rating')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# ## Content-based Filtering

# %% [markdown]
# Metode ini merekomendasikan berdasarkan kesamaan fitur (genre).

# %% [markdown]
# ### Data Preparation

# %% [markdown]
# Memakai movies dataset karena content-based filtering fokus pada fitur genre

# %%
movies_df = movies.copy()

# %%
movies_df.shape

# %%
movies[movies['genres'].str.contains('(no genres listed)')]

# %%
count_no_genres = movies_df[movies_df['genres'].str.contains(r'\(no genres listed\)')].shape[0]
print(f"Jumlah film dengan '(no genres listed)': {count_no_genres}")

# %%
genre_dummies = movies_df['genres'].str.get_dummies(sep='|')
movies_df = pd.concat([movies_df, genre_dummies], axis=1)

# %% [markdown]
# Hapus baris yang memiliki genre '(non genres listed)'

# %%
# Menghapus baris dengan genre '(no genres listed)'
movies_df = movies_df[~movies_df['genres'].str.contains(r'\(no genres listed\)')]

# %%
# Cek
movies_df.shape

# %% [markdown]
# Catatan :
# 
# Hasil sudah sesuai.
# - Data awal sebanyak 9742 baris
# - Jumlah film dengan '(no genres listed)': 34
# - Setelah baris dengan genre '(no genres listed)' dihapus : 9708

# %% [markdown]
# Menghapus kolom '(no genres listed)'

# %%
# Menghapus kolom '(no genres listed)'
movies_df.drop('(no genres listed)', axis=1, inplace=True)

# %%
movies_df.shape

# %%
movies_df.columns

# %% [markdown]
# Catatan :
# - kolom '(no genres listed)' berhasil dihapus.
# - Terdapat 22 kolom.

# %% [markdown]
# ### Model Development

# %%
from sklearn.metrics.pairwise import cosine_similarity

genre_features = movies_df.iloc[:, 3:]

cosine_sim = cosine_similarity(genre_features)
cosine_sim

# %%
def similarity_score_precision(title, cosine_sim, df, top_n=10, threshold=0.5):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    scores = [score for _, score in sim_scores]
    relevant = [s for s in scores if s >= threshold]
    return len(relevant) / top_n

def precision_genre(base_genres, recommended_genres):
    base_set = set(base_genres.split('|'))
    count = 0
    for rec_genre in recommended_genres:
        rec_set = set(rec_genre.split('|'))
        if base_set & rec_set:
            count += 1
    return count / len(recommended_genres)


# %%
def recommend_movies(title, cosine_sim, df, top_n=10):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_indices = [i[0] for i in sim_scores]
    return df.iloc[recommended_indices][['title', 'genres']]

# %%
# Pilih judul film yang ingin diuji
movie_title = "Toy Story (1995)"

movie_genre = movies_df.loc[movies_df['title'] == movie_title, 'genres'].values[0]
movie_id = movies_df.loc[movies_df['title'] == movie_title, 'movieId'].values[0]
genre_list = movie_genre.split('|')

recommend_df = recommend_movies(movie_title, cosine_sim, movies_df)

print("REKOMENDASI FILM BERDASARKAN GENRE:")
print("-" * 50)
print(f"Judul: {movie_title}")
print(f"MovieId: {movie_id}")
print(f"Genre: {movie_genre}")
print("-" * 50)
print("TOP 10 REKOMENDASI FILM :")
recommend_df


# %%
def genre_precision(input_genres, rec_genres):
    input_set = set(input_genres.split('|'))
    rec_set = set(rec_genres.split('|'))
    intersection = input_set.intersection(rec_set)
    precision = len(intersection) / len(rec_set) if len(rec_set) > 0 else 0
    return precision

target_title = "Toy Story (1995)"
target_idx = movies_df[movies_df['title'] == target_title].index[0]
target_genres = movies_df.loc[target_idx, 'genres']

similarities = cosine_sim[target_idx]

sim_scores = list(enumerate(similarities))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

recommendations = []

for i, score in sim_scores:
    rec_title = movies_df.iloc[i]['title']
    rec_genres = movies_df.iloc[i]['genres']
    prec = genre_precision(target_genres, rec_genres)
    recommendations.append({
        'Rekomendasi Film': rec_title,
        'Cosine Similarity': round(score, 4),
        'Genre Rekomendasi': rec_genres,
        'Precision Genre': round(prec, 4)
    })

df_recommendations = pd.DataFrame(recommendations)

print(f"Objek Film Pencarian: {target_title}")
print(f"Genre Film Pencarian: {target_genres}")
print("="*70)
print(df_recommendations)

# %%
precision_sim = similarity_score_precision(movie_title, cosine_sim, movies_df, top_n=10, threshold=0.5)
precision_gen = precision_genre(movie_genre, recommend_df['genres'].values)

print("\nEvaluasi Presisi:")
print("Presisi berdasarkan similarity score:", round(precision_sim, 2))
print("Presisi berdasarkan genre:", round(precision_gen, 2))

# %% [markdown]
# Insight :
# - Cosine Similarity: skor kemiripan berdasarkan encoding genre.
# - Precision Genre: proporsi genre rekomendasi yang sama dengan film input, dari 0 sampai 1.
# - Kalau precision genre 1.0 artinya genre rekomendasi 100% sama dengan genre film input.

# %% [markdown]
# ## Collaborative Filtering

# %% [markdown]
# Metode ini merekomendasikan berdasarkan items dari pengguna lain.

# %% [markdown]
# ### Data Preparation

# %% [markdown]
# #### Merge dataset

# %%
df = pd.merge(ratings, movies, on='movieId')

# %%
df.info()

# %% [markdown]
# Membentuk User-Item Matrix

# %%
user_item_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')

# %%
user_item_matrix.head()

# %%
user_item_matrix = user_item_matrix.fillna(0)

# %%
user_item_matrix.head()

# %% [markdown]
# ### Model Development

# %% [markdown]
# ##### Menggunakan NearestNeighbors algorithm

# %%
from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute', n_jobs=-1)
knn.fit(user_item_matrix.values.T)

# %%
avg_ratings = df.groupby('movieId')['rating'].mean().to_dict()

def get_similar_movies(movie_id, movie_titles, user_item_matrix, avg_ratings, n=10):
    if movie_id not in user_item_matrix.columns:
        return f"Movie ID {movie_id} not found in matrix."

    movie_idx = list(user_item_matrix.columns).index(movie_id)
    distances, indices = knn.kneighbors(
        user_item_matrix.values.T[movie_idx].reshape(1, -1), n_neighbors=n+1
    )

    similar_movies = []
    for i in range(1, len(distances[0])):
        sim_movie_id = user_item_matrix.columns[indices[0][i]]
        distance = distances[0][i]
        similarity = 1 - distance
        avg_rating = avg_ratings.get(sim_movie_id, np.nan)
        similar_movies.append((sim_movie_id, movie_titles.get(sim_movie_id, "Unknown Title"), similarity, round(avg_rating, 2)))

    return pd.DataFrame(similar_movies, columns=["movieId", "title", "similarity", "avg_rating"]).sort_values(by="similarity", ascending=False)


# %%
target_movie_id = 1
movie_info = movies[movies['movieId'] == target_movie_id]

if not movie_info.empty:
    target_title = movie_info.iloc[0]['title']
    target_genres = movie_info.iloc[0]['genres']
    target_rating = round(avg_ratings.get(target_movie_id, np.nan), 2)

    similar = get_similar_movies(target_movie_id, movies.set_index('movieId')['title'].to_dict(), user_item_matrix, avg_ratings, n=10)

    print(f"🎯 Objek Film Pencarian")
    print(f"   Judul   : {target_title}")
    print(f"   Genre   : {target_genres}")
    print(f"   Rating  : {target_rating}\n")

    print("🔍 10 Film yang Mirip:")
    print(similar)
else:
    print(f"Film dengan movieId = {target_movie_id} tidak ditemukan.")

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error

def predict_rating(user_id, movie_id, user_item_matrix, knn_model):
    if movie_id not in user_item_matrix.columns or user_id not in user_item_matrix.index:
        return np.nan

    movie_idx = list(user_item_matrix.columns).index(movie_id)
    distances, indices = knn_model.kneighbors(user_item_matrix.values.T[movie_idx].reshape(1, -1), n_neighbors=6)

    total_sim = 0
    weighted_sum = 0

    for i in range(1, len(distances[0])):  
        sim_movie_idx = indices[0][i]
        sim_movie_id = user_item_matrix.columns[sim_movie_idx]
        similarity = 1 - distances[0][i]

        if not np.isnan(user_item_matrix.loc[user_id, sim_movie_id]):
            rating = user_item_matrix.loc[user_id, sim_movie_id]
            weighted_sum += similarity * rating
            total_sim += similarity

    if total_sim == 0:
        return np.nan
    return weighted_sum / total_sim

sample_df = df.sample(100, random_state=42)

predicted_ratings = []
actual_ratings = []

for row in sample_df.itertuples():
    pred = predict_rating(row.userId, row.movieId, user_item_matrix, knn)
    if not np.isnan(pred):
        predicted_ratings.append(pred)
        actual_ratings.append(row.rating)

rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
mae = mean_absolute_error(actual_ratings, predicted_ratings)

print("Evaluasi Model KNN Collaborative Filtering")
print("RMSE:", round(rmse, 4))
print("MAE :", round(mae, 4))


# %% [markdown]
# Insight :
# - Menggunakan algortima NearestNeighbors
# - Similarity di kisaran ~0.53 - 0.57 menunjukkan film-film ini mirip secara pola rating user 


