import os
import json
import logging
import cherrypy

from flask import Flask, Blueprint

from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS

from paste.translogger import TransLogger


"""Simple online music artists recommender system.

This demo implements a online music recommender system on the dataset Audioscrobbler(small).
Spark MLlib and flask are employed to realize the target.
Basic functions includes:
    - Given a user_id, return top top_count artists for this user;
    - Given a user_id and a certain artist_id, predict potential visit numbers.

Example:
    Just start the recsys with following command:

        $ bash start_recsys.sh

    Then visit websites using using formats as below:

        http://127.0.0.1:5000/1059637/visit/top/5

        http://127.0.0.1:5000/1059637/visit/1007903

    Results will show on the page.

Todo:
    * Add more recommend methods
    * Try on larger datasets
    * Add a detailed evalution script

"""


class RecSysEngine:
    """define the music recommendation engine
    """

    def __train_model(self):
        """Train the ALS model with the current dataset
        """
        logger.info("Training the ALS model...")
        self.model = ALS.trainImplicit(self.UserArtistVisit_RDD, self.rank, seed=self.seed,
                                       iterations=self.iterations, lambda_=self.regularization_parameter)
        logger.info("ALS model trained!")

    def get_artist_for_user_id(self, user_id, artist_id):
        """Predict visit counts with a user_id and an artist_id
        """
        UserVisit_RDD = self.sc.parallelize(
            artist_id).map(lambda x: (user_id, x))
        # Get predicted visits
        PredictArtisitVisit_RDD = self.model.predictAll(
            UserVisit_RDD).map(lambda x: (x.product, x.rating))
        predicted_Visit = self.ArtistVisit_RDD.join(
            PredictArtisitVisit_RDD).map(lambda r: (r[1][0], r[1][1])).collect()

        return predicted_Visit

    def get_top_artists(self, user_id, top_count):
        """Recommends top_count top unvisited artists to user_id
        """
        # Get predicted artists
        RecProducts = self.model.recommendProducts(user_id, top_count)
        ArtistVisitDict = self.ArtistVisit_RDD.collectAsMap()
        artists_list = map(lambda x: (
            ArtistVisitDict.get(x.product)), RecProducts)

        return artists_list

    def __init__(self, sc, dataset_path):
        """Init the recsys engine with Spark context and dataset path
        """

        logger.info("Starting up the RecSys Engine: ")

        self.sc = sc

        # Load user_artist_data data
        logger.info("Loading artists data...")
        user_artist_visit_path = os.path.join(
            dataset_path, 'user_artist_data.txt')
        UserArtistVisit_raw_RDD = self.sc.textFile(user_artist_visit_path)
        self.UserArtistVisit_RDD = UserArtistVisit_raw_RDD.map(lambda x: x.split(" ")).map(
            lambda x: (int(x[0]), int(x[1]), int(x[2]))).cache()

        # Load artist_alias data
        ArtistAlias = self.sc.textFile(
            os.path.join(dataset_path, 'artist_alias.txt'))
        ArtistAliasDict = ArtistAlias.map(lambda x: x.split("\t")).map(
            lambda x: (int(x[0]), int(x[1]))).collectAsMap()

        # Load artist_data
        logger.info("Loading visits data...")
        artist_data_path = os.path.join(dataset_path, 'artist_data.txt')
        ArtistVisit_raw_RDD = self.sc.textFile(artist_data_path)
        self.ArtistVisit_RDD = ArtistVisit_raw_RDD.map(lambda x: x.split(
            "\t")).map(lambda x: (int(x[0]), x[1])).cache()

        # Remove duplicated data in UserArtistVisit_RDD
        self.UserArtistVisit_RDD = self.UserArtistVisit_RDD.map(lambda x: (
            x[0], ArtistAliasDict[x[1]], x[2]) if x[1] in ArtistAliasDict.keys() else x).cache()

        # Train the model
        self.rank = 10
        self.seed = 300
        self.iterations = 20
        self.regularization_parameter = 0.1
        self.__train_model()


"""config flask route
"""

main = Blueprint('main', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@main.route("/")
def hello():
    return "Welcome to Online Music Recommendation System!"


@main.route("/<int:user_id>/visit/top/<int:count>", methods=["GET"])
def top_artists(user_id, count):
    logger.debug("User %s TOP artists requested", user_id)
    top_artists = recsys.get_top_artists(user_id, count)
    return json.dumps(top_artists)


@main.route("/<int:user_id>/visit/<int:artist_id>", methods=["GET"])
def user_artist(user_id, artist_id):
    logger.debug("User %s requested for artist %s", user_id, artist_id)
    visits = recsys.get_artist_for_user_id(user_id, [
        artist_id])
    return json.dumps(visits)


def create_app(spark_context, dataset_path):
    global recsys

    recsys = RecSysEngine(spark_context, dataset_path)

    app = Flask(__name__)
    app.register_blueprint(main)
    return app


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""init and start service
"""


def run_server():
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)
    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')
    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 5000,
        'server.socket_host': '0.0.0.0'
    })
    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == "__main__":

    # Load data and init spark context
    dataset_path = os.path.join('datasets', 'Audioscrobbler-small')

    conf = SparkConf().setAppName("Simple-Online-Music-RecSys")
    sc = SparkContext(conf=conf)

    app = create_app(sc, dataset_path)

    # start web server
    run_server()
