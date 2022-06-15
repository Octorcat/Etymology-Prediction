from flask import Flask
from services.classification.classifier import predict

app = Flask(__name__)


@app.route("/etymology/<word>", methods=["GET"])
def predict_etymology_of(word: str):
    word_etymology_pred_probs = predict(word)
    return {"etymology": word_etymology_pred_probs}
