import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from datetime import datetime, timedelta


def vector_cosine(X, Y, dense=True):
    if not dense:
        X = X.toarray()[0]
        Y = Y.toarray()[0]

    X_mag = np.sqrt(X.dot(X))
    Y_mag = np.sqrt(Y.dot(Y))

    if X_mag == 0 or Y_mag == 0:
        return -1 # Hack to make null vectors maximally distant
    
    return X.dot(Y) / (X_mag * Y_mag)


# Recommender class to have persistent training data to recommend on
class MovieRecommender:
    def __init__(self,
                 strategy="knn",
                 metric="cosine",
                 k=20,
                 user_based=False,
                 ratings_path="ratings.csv",
                 movies_path="movies.csv",
                 df_critic_str="critic_url", # TODO: delete these when we switch to loading data from DB 
                 df_movie_str="movie_url"
                ):
        ratings, critics, movies = self.read_ratings(ratings_path)
        self.movies_dates = self.read_movies(movies_path)
        self.user_based = user_based

        self.num_critics = len(critics)
        self.num_movies = len(movies)
        
        # hashmaps to translate from unique url -> id and back
        self.critic_to_num, self.num_to_critic = self.map_ids(critics)
        self.movie_to_num, self.num_to_movie = self.map_ids(movies)

        _, self.processed_matrix = self.create_matrix(ratings)
        print("Ratings Matrix built")
        similarity_matrix = cosine_similarity(self.processed_matrix, dense_output=False)
        print("Similarity Matrix built")
        
        # save k closest neighbors to each row so prediction is fast
        self.k_neighbors = {}
        num_rows = self.num_critics if user_based else self.num_movies
        for row_id in range(num_rows):
            similarity_row = similarity_matrix[row_id].toarray()[0] # Turn row into numpy array for argpartition
            self.k_neighbors[row_id] = set(np.argpartition(similarity_row * -1, k)[:k+1])

        if strategy == "knn":
            self.recommend = self.knn_recommend
        else:
            raise ValueError("Bad strategy")
    

    # Read the CSV of sparse critic ratings
    def read_ratings(self, ratings_path):
        ratings = []
        critics = set()
        movies = set()

        with open(ratings_path, encoding="utf-8") as ratings_file:
            ratings_reader = csv.reader(ratings_file)
            next(ratings_reader)

            for row in ratings_reader:
                ratings.append([row[0], row[1], row[2]])
                critics.add(row[0])
                movies.add(row[1])

        return ratings, list(critics), list(movies)

    
    # Read the csv of movie data, including date
    def read_movies(self, movies_path):
        movies = dict()

        with open(movies_path, encoding="utf-8") as movies_file:
            movies_reader = csv.reader(movies_file)
            next(movies_reader)

            for row in movies_reader:
                movie_id = row[0]
                if row[2] != "":
                    releaseDate = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
                else:
                    releaseDate = datetime(1, 1, 1) # A long, long time ago

                movies[movie_id] = releaseDate
        
        return movies


    # Generate numerical aliases for keys in a hashmap
    def map_ids(self, items):
        item_to_num = {}
        num_to_item = {}
        num = 0

        for item in items:
            item_to_num[item] = num
            num_to_item[num] = item
            num += 1
                
        return item_to_num, num_to_item

    # generate ratings matrix
    def create_matrix(self, data):
        # set rows as our CF strategy. User_based sets critics as rows, where as item based sets movies
        if (self.user_based):
            num_rows = self.num_critics
            num_cols = self.num_movies
            row_ids = self.critic_to_num
            col_ids = self.movie_to_num
            # this is the element in the tuple that the critic_url is at when iterating over the dataframe.
            # these can be deleted once loading data from db
            tup_id_row = 0 
            tup_id_col = 1
        else:
            num_rows = self.num_movies
            num_cols = self.num_critics
            row_ids = self.movie_to_num
            col_ids = self.critic_to_num
            tup_id_row = 1
            tup_id_col = 0
        
        ratings_matrix = np.full((num_rows, num_cols), np.nan)

        ## TODO: CHANGE THIS WHEN LOADING DATA FROM DB
        for record in data:
            row = row_ids[record[tup_id_row]]
            col = col_ids[record[tup_id_col]]
            rating = max(int(record[2]), 1) # ghetto workaround for storing explicit 0 vals
            ratings_matrix[row, col] = rating

        # mean center each movie based on its ratings
        mean_centered = self.preprocess_ratings_matrix(ratings_matrix, num_rows)
        
        # set nans back to 0
        ratings_matrix[np.isnan(ratings_matrix)] = 0

        ratings_matrix = sparse.csr_matrix(ratings_matrix)

        return ratings_matrix, mean_centered

    # preprocessing strategy before we compute similarity matrix
    def preprocess_ratings_matrix(self, ratings_matrix, num_rows):
        # mean center each movie based on its ratings
        mean_centered = ratings_matrix - np.nanmean(ratings_matrix, axis = 1).reshape(num_rows, 1)

        # set nans back to 0
        mean_centered[np.isnan(mean_centered)] = 0
        
        return sparse.csr_matrix(mean_centered)

    # Generate user vector to run algorithms on
    def create_user_vector(self, cols, ratings, col_ids):
        vector = np.zeros(shape=(cols))
        for movie in ratings:
            if movie in col_ids:
                vector[col_ids[movie]] = ratings[movie]
        return vector

    
    # TODO: extend to user based predictions
    #       add new_movie_ids block that separates recommendations
    #       remove movies with very few ratings??
    #       use baseline estimate with CF
    
    # Generate recommendations from given ratings using KNN
    def knn_recommend(self, user_ratings, rec_count=100):
        # Get the current time for filtering new movies
        curr_time = datetime.now()
        max_time_diff = timedelta(days=90)

        # get ids for movies that user has rated
        user_vector = self.create_user_vector(self.num_movies, user_ratings, self.movie_to_num)
        movies_rated_by_user = np.nonzero(user_vector)[0]
        
        preds = np.zeros(shape=(self.num_movies))
        
        # make a prediction for every movie by using weighted avg of related movies that
        # user has rated. if they haven't rated any related movies fall back to default (right now 0)
        for movie_id in range(self.num_movies):
            running_sum = 0
            running_denom = 0

            for rated_movie in movies_rated_by_user:
                # keep track of weighted average of movies that this user has rated as we go
                if rated_movie in self.k_neighbors[movie_id]:
                    similarity = vector_cosine(self.processed_matrix[movie_id], self.processed_matrix[rated_movie], dense=False)

                    if similarity < 0: # don't use negatively related movies in the weighted avg
                        continue
                    user_rating = user_ratings[self.num_to_movie[rated_movie]]
                    running_sum += (similarity * user_rating)
                    running_denom += similarity

            # if the user has not rated any of the similar movies, just predict avg rating
            # otherwise, take the weighted average of movies they had rated.
            if running_denom != 0:
                weighted_avg = running_sum / running_denom
                pred = weighted_avg
            else:
                pred = 0

            preds[movie_id] = pred

        all_recs = {}
        new_recs = {}
        movies = np.argpartition(preds * -1, rec_count)[:rec_count+1]
        for movie_id in movies:
            movie_codename = self.num_to_movie[movie_id]
            # don't include movies user has already rated
            if self.num_to_movie[movie_id] not in user_ratings:
                all_recs[movie_codename] = preds[movie_id]

                # Check if the movie is new enough to include
                try: # Stupid rotten tomatoes with different movie codenames
                    if (curr_time - self.movies_dates[movie_codename[7:]]) < max_time_diff:
                        new_recs[movie_codename] = preds[movie_id]
                except KeyError:
                    pass

        return all_recs, new_recs
