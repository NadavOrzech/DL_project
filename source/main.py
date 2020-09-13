import torch.nn as nn
import torch.optim as optim
from data_loader import ECGDataLoader
from training import BaselineModel
from cs236781.plot import plot_fit
from config import Config


if __name__ == '__main__':
    config = Config()
    dataloader = ECGDataLoader(config)
    train_dataloader, test_dataloader = dataloader.get_dataloader()
    model = BaselineModel(config)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    fit_result = model.fit(train_dataloader, test_dataloader, optimizer, loss_fn, max_epochs=config.num_epochs,
                           early_stopping=config.early_stopping)
    fig, axes = plot_fit(fit_result)


