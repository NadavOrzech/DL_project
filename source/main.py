import torch.nn as nn
import torch.optim as optim
from data_loader import ECGDataLoader
# from data_preprocess import BEAT_SIZE
from training import BaselineModel

from config import Config

# HIDDEN_SIZE = 400
# BEATS = BEAT_SIZE*2
# NUM_EPOCHS = 4


if __name__ == '__main__':
    config = Config()
    dataloader = ECGDataLoader(config)
    train_dataloader, test_dataloader = dataloader.get_dataloader()
    
    model = BaselineModel(config)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    model.train(optimizer, loss_fn, train_dataloader, max_epochs=config.num_epochs)
    model.test(loss_fn, test_dataloader, max_epochs=config.num_epochs)


