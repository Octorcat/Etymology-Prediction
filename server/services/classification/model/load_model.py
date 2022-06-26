from typing import List
import torch
import torch.nn as nn
import os


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


ALL_CHARS: List[str] = [
    "x",
    "u",
    "l",
    "p",
    "f",
    "n",
    "v",
    "-",
    "g",
    "b",
    "d",
    "m",
    "e",
    "h",
    "z",
    "k",
    "s",
    "c",
    "o",
    "'",
    "w",
    "y",
    "i",
    "j",
    "r",
    "t",
    "q",
    "a",
    ".",
]
NB_CHARS: int = len(ALL_CHARS)
NB_HIDDEN: int = 34 * 2
NB_CATEGORIES: int = 2
MODEL_PTH_PATH: str = f"{os.getcwd()}{'/server' if os.environ.get('FLASK_ENV') == 'production' else ''}/services/classification/model/model.pth"
model: RNN = RNN(NB_CHARS, NB_HIDDEN, NB_CATEGORIES)
model.load_state_dict(torch.load(MODEL_PTH_PATH))
model.eval()
