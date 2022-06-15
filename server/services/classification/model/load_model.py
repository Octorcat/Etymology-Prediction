import torch
import torch.nn as nn
import os


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


MODEL_PTH_PATH = f"{os.getcwd()}/services/classification/model/model.pth"
model = RNN(NB_CHARS, NB_HIDDEN, NB_CATEGORIES)
model.load_state_dict(torch.load(MODEL_PTH_PATH))
model.eval()
