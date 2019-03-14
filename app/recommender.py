import pandas as pd
import collections
from scipy.sparse import *
from sklearn.neighbors import NearestNeighbors


# Generate numerical aliases for keys in a hashmap
def map_ids(items, return_inverse=False):
    item_to_num = {}
    num_to_item = {}
    num = 0
    for item in items:
        if item not in item_to_num:
            item_to_num[item] = num
            num_to_item[num] = item
            num += 1
    if return_inverse:
        return item_to_num, num_to_item
    else:
        return item_to_num


# generate sparse matrix representation
def create_matrix(rows, cols, df, row_ids, col_ids):
    matrix = lil_matrix((rows, cols))
    for tup in df.iterrows():
        row = row_ids[tup[1][0]]
        col = col_ids[tup[1][1]]
        rating = max(int(tup[1][2]), 1) # ghetto workaround for storing explicit 0 vals in sparse matrix
        matrix[row, col] = rating
    return matrix.todense()


# Generate user vector to run algorithms on
def create_user_vector(cols, ratings, col_ids):
    matrix = lil_matrix((1, cols))
    for movie in ratings:
        col = col_ids[movie]
        matrix[0, col] = ratings[movie]
    return matrix.todense()


# Recommender class to have persisestent training data to recommend on
class MovieRecommender:
    def __init__(self, strategy="knn", metric="cosine", num_neighbors=15):
        self.mc_ratings = pd.read_csv("mc_ratings.csv")
        self.critics = self.mc_ratings['critic_url'].values
        self.movies = self.mc_ratings['movie_url'].values

        self.num_critics = len(set(self.critics))
        self.num_movies = len(set(self.movies))
        self.critic_ids = map_ids(self.critics)
        self.movie_to_num, self.num_to_movie = map_ids(self.movies, return_inverse=True)

        self.matrix = create_matrix(self.num_critics, self.num_movies, self.mc_ratings, self.critic_ids, self.movie_to_num)
        print("Matrix built")

        if strategy == "knn":
            self.recommend = self.knn_recommend
            self.knn = NearestNeighbors(n_neighbors=num_neighbors, metric=metric).fit(self.matrix)
        else:
            raise ValueError("Bad strategy")
    
    # Generate recommendations from given ratings using KNN
    def knn_recommend(self, user_ratings):
        user_vector = create_user_vector(self.num_movies, user_ratings, self.movie_to_num)
        neighbors = self.knn.kneighbors(X=user_vector, return_distance=False)[0]
        
        predictions = {}

        for movie_id in range(self.matrix.shape[1]):
            movie_rating = 0
            counter = 0

            for critic_id in neighbors:
                if self.matrix[critic_id, movie_id] > 0:
                    counter += 1
                    movie_rating += self.matrix[critic_id, movie_id]

            if counter > 0:
                predictions[self.num_to_movie[movie_id]] = movie_rating / counter

        for movie in user_ratings:
            if movie in predictions:
                del predictions[movie]

        print(predictions)
        return predictions