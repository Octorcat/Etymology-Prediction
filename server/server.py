from flask import Flask, Response, send_from_directory
from flask_restx import Resource, Api

from flask_cors import CORS
from services.classification.classifier import EtymologyPrediction

api = Api()

app = Flask(__name__)
api.init_app(app)
CORS(app)


@app.route("/")
def index() -> Response:
    return send_from_directory("../client/dist", "index.html")


@app.route("/<path:path>")
def serve_static_files(path: str) -> Response:
    return send_from_directory("../client/dist", path)


@api.route("/etymology/<string:word>")
class EtymologyPredictionRouter(Resource):
    def get(self, word: str) -> dict:
        word_etymology_pred_probs = EtymologyPrediction.predict(word)
        return {"etymology": word_etymology_pred_probs}
