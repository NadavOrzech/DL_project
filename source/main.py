import torch.nn as nn
import torch.optim as optim
from data_loader import WeightedDataLoader, BaseDataloader
from data_preprocess import DataProcessor
from models import BaselineModel, AttentionModel
from cs236781.plot import plot_fit,plot_attention_map
from config import Config
import torch
import os


if __name__ == '__main__':
    config = Config()
    checkpoint_dir = os.path.join('.', 'checkpoints')
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpointatt')
    if os.path.isfile(checkpoint_file):
        data = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        fit_result = data['fit_result']
        heat_map = data['heatmap']
        seq_file = os.path.join('dataset_checkpoints', 'seq_dataset_{}'.format(config.overlap))
        seq = torch.load(seq_file)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        proccesor = DataProcessor(config.files_dir, config.overlap, config.seq_size, config.beat_size)
        dataset, seq = proccesor.get_data()

        dataloader = WeightedDataLoader(config, dataset)
        train_dataloader = dataloader.get_train_dataloader()
        test_dataloader = dataloader.get_test_dataloader()

        model = AttentionModel(config, device=device, checkpoint_file='checkpoint1')
        model.to(device)

        print("device: {}".format(device))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        fit_result, heat_map = model.fit(train_dataloader, test_dataloader, optimizer, loss_fn, max_epochs=config.num_epochs,
                                         early_stopping=config.early_stopping)
    fig, axes = plot_fit(fit_result, 'first_graph', legend='total')

    plot_attention_map(heat_map, seq)
