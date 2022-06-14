from flask import Flask
import torch
import math

app = Flask(__name__)

ALL_CHARS = [
    "y",
    "v",
    "r",
    "q",
    "x",
    "w",
    "k",
    "t",
    "a",
    "i",
    "p",
    "d",
    "z",
    "h",
    "n",
    "b",
    "s",
    ".",
    "m",
    "f",
    "g",
    "l",
    "'",
    "-",
    "e",
    "j",
    "c",
    "o",
    "u",
]
NB_CHARS = len(ALL_CHARS)
NB_HIDDEN = 34
NB_CATEGORIES = 2

import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.tanh = nn.Tanh()  # Added layer on original
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.tanh(hidden)
        output = self.i2o(combined)
        output = self.tanh(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


model = RNN(NB_CHARS, NB_HIDDEN, NB_CATEGORIES)
model.load_state_dict(torch.load("../model.pth"))
model.eval()


def word_to_tensor(word):
    tensor = torch.zeros(len(word), 1, NB_CHARS)
    for i, letter in enumerate(word):
        tensor[i][0][ALL_CHARS.index(letter)] = 1
    return tensor


def predict(word_tensor):
    hidden = model.initHidden()
    for i in range(word_tensor.size()[0]):
        output, hidden = model(word_tensor[i], hidden)
    return output


def tensor_to_etymology(out_tensor):
    etymology_pred_probs = []
    topv, topi = out_tensor.topk(NB_CATEGORIES, 1, True)
    for i in range(NB_CATEGORIES):
        category = ["Latin", "Germanic"][topi[0][i].item()]
        probability = math.exp(topv[0][i].item())
        etymology_pred_probs.append({category: probability})
    return etymology_pred_probs


@app.route("/etymology/<word>", methods=["GET"])
def predict_etymology_of(word: str):
    in_tensor = word_to_tensor(word)
    out_tensor = predict(in_tensor)
    word_etymology_pred_probs = tensor_to_etymology(out_tensor)
    return {"etymology": word_etymology_pred_probs}
