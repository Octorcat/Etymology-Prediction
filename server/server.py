from flask import Flask

app = Flask(__name__)

''' TODO: Serve the etymology prediction of each word'''
@app.route("/etymology/<word>", methods=["GET"])
def predict_etymology_of(word: str) -> str:
    return word
