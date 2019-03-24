import csv
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime


cred = credentials.Certificate("./firebase_credentials.json")
movie_night = firebase_admin.initialize_app(cred)
db = firestore.client()


# Load all the critics into a dict without ratings
def get_critics(file_name):
    critics = dict()

    critics_file = open(file_name, encoding="utf-8")
    critics_reader = csv.reader(critics_file)
    next(critics_reader)

    for row in critics_reader:
        critic_id = row[1][8:]
        critic_name = row[0]

        if critic_id not in critics:
            critics[critic_id] = {"name": critic_name, "ratings":{}}

    critics_file.close()
    return critics

# Load all the movies into a dict without ratings
def get_movies(file_name):
    movies = dict()

    movies_file = open(file_name, encoding="utf-8")
    movies_reader = csv.reader(movies_file)
    next(movies_reader)

    for row in movies_reader:
        movie_id = row[1][3:]
        movie_name = row[2]

        if movie_id not in movies:
            movies[movie_id] = {"name": movie_name, "ratings":{}}

    movies_file.close()
    return movies

# Load all the rating data into a critics map and movies map
def populate_ratings(file_name):
    critics = get_critics("critics.csv")
    movies = get_movies("reviews.csv")

    RT_file = open(file_name, encoding="utf-8")
    RT_reader = csv.reader(RT_file)
    next(RT_reader)

    for row in RT_reader:
        critic_id = row[0][7:]
        movie_id = row[1][3:]
        if critic_id in critics and movie_id in movies:
            try:
                numer, denom = row[4].split("/")
                numer_rating = int(100 * float(numer) / float(denom))
                critics[critic_id]["ratings"][movie_id] = numer_rating
                movies[movie_id]["ratings"][critic_id] = numer_rating
            except:
                pass

    RT_file.close()

    culled_critics = {}
    for critic in critics:
        if len(critics[critic]["ratings"]) > 0:
            culled_critics[critic] = critics[critic]
    critics = culled_critics

    culled_movies = {}
    for movie in movies:
        if len(movies[movie]["ratings"]) > 0:
            culled_movies[movie] = movies[movie]
    movies = culled_movies

    for critic_id in critics:
        critic_ref = db.collection("critics").document(critic_id)
        critic_ref.set({
            "name": critics[critic_id]["name"],
        })
        
        batch = db.batch()
        count = 0
        for movie_id in critics[critic_id]["ratings"]:
            rating_ref = critic_ref.collection("ratings").document(movie_id)
            
            if count >= 500:
                batch.commit()
                batch = db.batch()
                count = 0

            batch.set(rating_ref, {
                "name": movies[movie_id]["name"],
                "score": critics[critic_id]["ratings"][movie_id]
            })
            count += 1
        batch.commit()

    for movie_id in movies:
        movie_ref = db.collection("movies").document(movie_id)
        movie_ref.set({
            "name": movies[movie_id]["name"],
        })
        
        batch = db.batch()
        count = 0
        for critic_id in movies[movie_id]["ratings"]:
            rating_ref = movie_ref.collection("ratings").document(critic_id)
            
            if count >= 500:
                batch.commit()
                batch = db.batch()
                count = 0

            batch.set(rating_ref, {
                "name": critics[critic_id]["name"],
                "score": movies[movie_id]["ratings"][critic_id]
            })
            count += 1
        batch.commit()

populate_ratings("reviews.csv")