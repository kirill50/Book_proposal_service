# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

# import the necessary packages
import dill
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import os

dill._dill._reverse_typemap['ClassType'] = type
# import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# function to get recommendations
def book_recommendations(title, n=5):
    test_tfidf = model.transform(title)
    cosine_sim_titles = cosine_similarity(test_tfidf, tfidf_matrix)
    ind = np.argpartition(cosine_sim_titles, -n)
    candidates = titles_train.iloc[ind[0][:n]].values
    return candidates


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)


# load pipeline
modelpath = "./models/book_pipeline.dill"
load_model(modelpath)

# read train set to have titles (we are going to recommend some of them)
data = pd.read_csv("./models/books.csv")
titles_train = data['original_title']
# getting a matrix for original titles
tfidf_matrix = model.transform(data)


@app.route("/", methods=["GET"])
def general():
    return """Welcome to book proposal aps. Please use 'http://<address>/predict' to POST"""


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        original_title = ""
        request_json = flask.request.get_json()
        if request_json["original_title"]:
            original_title = request_json['original_title']
        # logger.info(f'{dt} Data: original_title={original_title}')
        # if request_json["company_profile"]:
        #	company_profile = request_json['company_profile']

        # if request_json["benefits"]:
        #	benefits = request_json['benefits']

        try:
            # preds = model.transform(pd.DataFrame({"original_title": [original_title]}))
            # getting top 5 titles
            predicted_titles = book_recommendations(pd.DataFrame({"original_title": [original_title]}))
        except AttributeError as e:
            # logger.warning(f'{dt} Exception: {str(e)}')
            data['predictions'] = str(e)
            data['success'] = False
            return flask.jsonify(data)

        data["predictions"] = "; ".join(predicted_titles)
        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    print(data)
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=True, port=port)