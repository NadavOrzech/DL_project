import torch.nn as nn
import torch.optim as optim
from data_loader import WeightedDataLoader, BaseDataloader
from data_preprocess import DataProcessor
from models import BaselineModel, AttentionModel
from cs236781.plot import plot_fit,plot_attention_map
from config import Config
import torch


if __name__ == '__main__':
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proccesor = DataProcessor(config.files_dir, config.overlap, config.seq_size, config.beat_size)
    dataset, seq = proccesor.get_data()
    
    dataloader = WeightedDataLoader(config, dataset)
    train_dataloader = dataloader.get_train_dataloader()
    test_dataloader = dataloader.get_test_dataloader()
    
    # print(dataset[0])

    model = AttentionModel(config, device=device,checkpoint_file='checkpoint1')
    model.to(device)

    print("device: {}".format(device))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    fit_result, heat_map = model.fit(train_dataloader, test_dataloader, optimizer, loss_fn, max_epochs=config.num_epochs,
                           early_stopping=config.early_stopping,)
    # print(model.soft_attn_weights)

    fig, axes = plot_fit(fit_result, 'first_graph', legend='total')

    plot_attention_map(heat_map,seq)
