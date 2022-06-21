from flask import Flask, Response, send_from_directory
from flask_restx import Resource, Api
from flask_cors import CORS
from services.classification.classifier import EtymologyPrediction
from services.classification.model.load_model import ALL_CHARS as ALLOWED_CHARS


app = Flask(__name__)
CORS(app)

CLIENT_DIST_DIR = "../client/dist"


@app.route("/")
def index() -> Response:
    return send_from_directory(CLIENT_DIST_DIR, "index.html")


@app.route("/<path:path>")
def serve_static_files(path: str) -> Response:
    return send_from_directory(CLIENT_DIST_DIR, path)


api = Api(app, doc="/doc")


@api.route("/etymology/<string:word>")
class EtymologyPredictionRouter(Resource):
    def get(self, word: str) -> dict:
        MAX_WORD_LEN = 34
        word = word.strip().lower()
        if len(word) > MAX_WORD_LEN:
            error_msg = f"Word must be <= {MAX_WORD_LEN}."
            api.abort(400, error_msg)
        elif not all(char in ALLOWED_CHARS for char in word):
            error_msg = f"Word must be contain English letters and punctuation only."
            api.abort(400, error_msg)
        else:
            word_etymology_pred_probs = EtymologyPrediction.predict(word)
            return {"etymology": word_etymology_pred_probs}
