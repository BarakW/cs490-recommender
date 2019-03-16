import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import *

# Recommender class to have persistent training data to recommend on
class MovieRecommender:
    def __init__(self,
                 strategy="knn",
                 metric="cosine",
                 k=20,
                 user_based=False,
                 file_path="clean_RT.csv",
                 df_critic_str="critic_url", # TODO: delete these when we switch to loading data from DB 
                 df_movie_str="movie_url"
                ):
        self.ratings, self.critics, self.movies = self.read_ratings(file_path)
        self.user_based = user_based

        self.num_critics = len(self.critics)
        self.num_movies = len(self.movies)
        
        # hashmaps to translate from unique url -> id and back
        self.critic_to_num, self.num_to_critic = self.map_ids(self.critics)
        self.movie_to_num, self.num_to_movie = self.map_ids(self.movies)

        self.ratings_matrix, self.processed_matrix = self.create_matrix(self.ratings)
        print("Ratings Matrix built")
        self.similarity_matrix = cosine_similarity(self.processed_matrix)
        print("Similarity Matrix built")
        
        # global mean rating for all movies
        self.global_mean = np.mean(self.ratings_matrix[np.nonzero(self.ratings_matrix)])
        
        # save mean ratings for each row
        self.mean_ratings = {}
        # save k closest neighbors to each row so prediction is fast
        self.k_neighbors = {}
        num_rows = self.num_critics if user_based else self.num_movies
        for row_id in range(num_rows):
            self.mean_ratings[row_id] = np.mean(self.ratings_matrix[row_id, np.nonzero(self.ratings_matrix[row_id])])
            self.k_neighbors[row_id] = set(np.argpartition(self.similarity_matrix[row_id] * -1, k)[:k+1])

        if strategy == "knn":
            self.recommend = self.knn_recommend
        else:
            raise ValueError("Bad strategy")
    

    # Read the CSV of sparse critic ratings
    def read_ratings(self, file_path):
        ratings = []
        critics = set()
        movies = set()

        with open(file_path) as ratings_file:
            ratings_reader = csv.reader(ratings_file)
            next(ratings_reader)

            for row in ratings_reader:
                ratings.append([row[0], row[1], row[2]])
                critics.add(row[0])
                movies.add(row[1])

        return ratings, list(critics), list(movies)
    
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
        for record in self.ratings:
            row = row_ids[record[tup_id_row]]
            col = col_ids[record[tup_id_col]]
            rating = max(int(record[2]), 1) # ghetto workaround for storing explicit 0 vals
            ratings_matrix[row, col] = rating

        # mean center each movie based on its ratings
        mean_centered = self.preprocess_ratings_matrix(ratings_matrix, num_rows)
        
        # set nans back to 0
        ratings_matrix[np.isnan(ratings_matrix)] = 0

        return ratings_matrix, mean_centered

    # preprocessing strategy before we compute similarity matrix
    def preprocess_ratings_matrix(self, ratings_matrix, num_rows):
        # mean center each movie based on its ratings
        mean_centered = ratings_matrix - np.nanmean(ratings_matrix, axis = 1).reshape(num_rows, 1)

        # set nans back to 0
        mean_centered[np.isnan(mean_centered)] = 0
        
        return mean_centered

    # Generate user vector to run algorithms on
    def create_user_vector(self, cols, ratings, col_ids):
        vector = np.zeros(shape=(cols))
        for movie in ratings:
            if movie in col_ids:
                vector[col_ids[movie]] = ratings[movie]
        return vector

    # return the most similar rows to a given item.
    # can be used to find most similar users or most similar movies to a given item
    def get_similar_rows(self, url, similar_count):
        # set correct hashmaps to use for looking up id and url
        if self.user_based:
            row_ids = self.critic_to_num
            id_to_url = self.num_to_critic
        else:
            row_ids = self.movie_to_num
            id_to_url = self.num_to_movie

        # get the highest similarity values from the similarity matrix
        item_id = row_ids[url]
        neighbors = np.argsort(self.similarity_matrix[item_id] * -1)[:similar_count+1]
        sim_values = self.similarity_matrix[item_id, neighbors]

        # pretty return the similarity values with their item
        sims = {}
        for neighbor_idx, item_id in enumerate(neighbors):
            sims[id_to_url[item_id]] = sim_values[neighbor_idx]
        return sims

    def most_reviewed_movies(self, num_movies_to_get):
        # get number of ratings for every movie
        num_ratings_per_movie = np.zeros(shape = (self.num_movies))
        for movie_id in range(self.num_movies):
            num_ratings_per_movie[movie_id] = len(np.nonzero(self.ratings_matrix[movie_id])[1])

        # sort by most reviewed, and return with movie name
        most_reviewed_movies = np.argsort(num_ratings_per_movie * -1)[:num_movies_to_get+1]
        
        mvs = {}
        for m in most_reviewed_movies:
            mvs[self.num_to_movie[m]] = num_ratings_per_movie[m]
        return mvs

    
    # TODO: extend to user based predictions
    #       add new_movie_ids block that separates recommendations
    #       remove movies with very few ratings??
    #       use baseline estimate with CF
    
    # Generate recommendations from given ratings using KNN
    def knn_recommend(self, user_ratings, new_movie_ids=[], rec_count=30):
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
                    similarity = self.similarity_matrix[movie_id, rated_movie]

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
                pred = 0#self.mean_ratings[movie_id]

            preds[movie_id] = pred

        recs = {}
        movies = np.argpartition(preds * -1, rec_count)[:rec_count+1]
        for movie_id in movies:
            # don't include movies user has already rated
            if self.num_to_movie[movie_id] in user_ratings:
                continue
            recs[self.num_to_movie[movie_id]] = preds[movie_id]
        return recs
