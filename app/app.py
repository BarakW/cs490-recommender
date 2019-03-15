from flask import Flask
from flask import request
from flask import Response
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from recommender import MovieRecommender

app = Flask(__name__)

cred = credentials.Certificate("./firebase_credentials.json")
movie_night = firebase_admin.initialize_app(cred)

db = firestore.client()

recommender = MovieRecommender()

"""
Responds to post requests to generate recommendations.
Generated recommendations are stored under the user in firebase.
"""
def generate_recommendations(stale_recs_snapshot, changes, readtime):
    for user_snapshot in stale_recs_snapshot:
        user_ref = user_snapshot.reference

        user_doc = user_ref.get()
        
        if not user_doc:
            return
        elif user_doc.to_dict()["stale"] == False:
            return
        
        user_ratings = user_doc.to_dict()["ratings"]
        movies = list(user_ratings.keys())
        for movie in movies:
            user_ratings["/movie/" + movie] = user_ratings[movie]
            del user_ratings[movie]

        user_recommendations = recommender.recommend(user_ratings)

        movies = list(user_recommendations.keys())
        for movie in movies:
            user_recommendations[movie[7:]] = user_recommendations[movie]
            del user_recommendations[movie]

        user_ref.update({
            "recommendations": user_recommendations,
            "stale": False
        })

stale_recommendations_query = db.collection("users").where("stale", "==", True)
query_watch = stale_recommendations_query.on_snapshot(generate_recommendations)