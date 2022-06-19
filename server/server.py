from flask import Flask, send_from_directory

from flask_cors import CORS
from services.classification.classifier import predict

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return send_from_directory("../client/dist", "index.html")


@app.route("/<path:path>")
def serve_static_files(path):
    return send_from_directory("../client/dist", path)


@app.route("/etymology/<word>", methods=["GET"])
def predict_etymology_of(word: str):
    word_etymology_pred_probs = predict(word)
    return {"etymology": word_etymology_pred_probs}
