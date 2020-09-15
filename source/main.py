import torch.nn as nn
import torch.optim as optim
from data_loader import ECGDataLoader
from training import BaselineModel, AttentionModel
from cs236781.plot import plot_fit
from config import Config
import torch


if __name__ == '__main__':
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = ECGDataLoader(config)
    train_dataloader, test_dataloader = dataloader.get_dataloader()
    model = AttentionModel(config, device=device)
    model.to(device)
    print("device: {}".format(device))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    fit_result = model.fit(train_dataloader, test_dataloader, optimizer, loss_fn, max_epochs=config.num_epochs,
                           early_stopping=config.early_stopping)
    fig, axes = plot_fit(fit_result)


