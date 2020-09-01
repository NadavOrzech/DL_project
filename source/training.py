import torch
import torch.nn as nn

INPUT_SIZE = (100, 28000, 1)
HIDDEN_SIZE = (100, 400)


def get_model():
    model = nn.Sequential(
        nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bidirectional=True)
    )