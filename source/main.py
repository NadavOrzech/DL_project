import torch.nn as nn
import torch.optim as optim
from data_loader import CustomDataLoader
from data_preprocess import DataProcessor
from models import BaselineModel, AttentionModel
from cs236781.plot import plot_fit, plot_attention_map, plot_both_models
from config import Config
import torch
import os


if __name__ == '__main__':
    config = Config()
    checkpoint_dir = os.path.join('.', 'checkpoints')
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_attention')
    base_checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_baseline')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.isfile(checkpoint_file):
        data = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        fit_result = data['fit_result']
        heat_map = data['heatmap']

        base_data = torch.load(base_checkpoint_file, map_location=torch.device('cpu'))
        base_fit_result = base_data['fit_result']

        seq_file = os.path.join('dataset_checkpoints', 'seq_dataset_{}_test'.format(config.overlap))
        seq = torch.load(seq_file)
    else:
        proccesor = DataProcessor(config.files_dir, config.overlap, config.seq_size, config.beat_size)
        train_dataset, train_seq = proccesor.get_data(start_file=0,end_file=20)
        test_dataset, test_seq = proccesor.get_data(start_file=20,end_file=23)

        dataloader = CustomDataLoader(config, dataset=train_dataset, test_dataset=test_dataset)
        train_dataloader = dataloader.get_train_dataloader()
        test_dataloader = dataloader.get_test_dataloader()

        model = AttentionModel(config, device=device, checkpoint_file='checkpoint_attention')
        model.to(device)

        baseline_model = BaselineModel(config, device=device, checkpoint_file='checkpoint_baseline')
        baseline_model.to(device)

        print("device: {}".format(device))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=config.lr)
        
        fit_result, heat_map = model.fit(train_dataloader, test_dataloader, optimizer, loss_fn, max_epochs=config.num_epochs,
                                         early_stopping=config.early_stopping)

        base_fit_result, _ = baseline_model.fit(train_dataloader, test_dataloader, baseline_optimizer, loss_fn, max_epochs=config.num_epochs,
                                         early_stopping=config.early_stopping)
    
    if device.type == 'cpu':
        fig, axes = plot_fit(fit_result, 'Attention_graph', legend='total')
        plot_attention_map(heat_map, seq)
        fig, axes = plot_fit(base_fit_result, 'Baseline_graph', legend='total')
        fig, axes = plot_both_models(fit_attention=fit_result, fit_base=base_fit_result, output_name='compare')
