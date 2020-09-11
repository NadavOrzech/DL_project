import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from data_preprocess import BEAT_SIZE
from training import LSTM

HIDDEN_SIZE = 400
BEATS = BEAT_SIZE*2
NUM_EPOCHS = 4


if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloader()
    model = LSTM(input_dim=BEATS, hidden_dim=HIDDEN_SIZE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train(optimizer, loss_fn, train_dataloader, max_epochs=NUM_EPOCHS)
    model.test(loss_fn, test_dataloader, max_epochs=NUM_EPOCHS)
