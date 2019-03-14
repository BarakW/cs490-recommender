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
@app.route("/recommend/<userid>", methods=["POST"])
def generate_recommendations(userid):
    user_ref = db.collection("users").document(userid)

    try:
        user_doc = user_ref.get()
    except:
        return Response(status=400)
    
    user_ratings = user_doc.to_dict()["ratings"]
    user_recommendations = recommender.recommend(user_ratings)

    user_ref.update({
        "recommendations": user_recommendations
    })

    return Response(status=200)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)