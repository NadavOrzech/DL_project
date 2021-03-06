import torch
import torch.nn as nn
import sys
import tqdm
import os

from cs236781.train_results import EpochResult, FitResult, EpochHeatMap
from torch.utils.data import DataLoader


class BaselineModel(nn.Module):
    def __init__(self, config, device, output_dim=2, num_layers=1,checkpoint_file=None):
        """
        Custom Bidirectional LSTM module

        :param config: Config class containing all configurations
        :param device: Device to run the model training on 
        :param output_dim: The model output dimension. On the standard case is 2 - (Positive, Negative)
        :param num_layers: Number of layer for LSTM layer (default is 1)
        :param checkpoint_file: Name for checkpoint file to load from and save to
        """
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = config.dropout
        self.overlap = config.overlap
        self.batch_size = config.batch_size
        self.device = device
        self.checkpoint_file = None
        self.soft_attn_weights = torch.zeros(self.batch_size)  # initialized for generating Graphs for AttentionModel

        if checkpoint_file is not None:
            checkpoint_dir = os.path.join('.', 'checkpoints')
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            self.checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file)

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, bidirectional=True, dropout=config.lstm_dropout)
        self.max_pool = nn.MaxPool1d(kernel_size=config.seq_size)
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim*2, out_features=50),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=50, out_features=self.output_dim),
            nn.Sigmoid()
        )

    def save_checkpoint(self, best_acc, heatmap, fit_result):
        """
        Saves checkpoint file for the model training 
        :param best_acc: Best test accuracy 
        :param heatmap: A EpochHeatMap object holds the attention map for all sequences
        :param fit_result: A FitResult object containing train and test losses per epoch. 
        """
        
        lstm_params = self.lstm.state_dict()
        linear_params = self.linear.state_dict()
        data = dict(
            lstm_params=lstm_params,
            linear_params=linear_params,
            best_acc=best_acc,
            heatmap=heatmap,
            fit_result=fit_result,
        )
        torch.save(data, self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load the init checkpoint file
        :return: A tuple of:
                    Best test accuracy for last checkpoint
                    A EpochHeatMap object holds the attention map for all sequences for last checkpoint
                    A FitResult object containing train and test losses per epoch for last checkpoint
        """
        
        print(f'=== Loading checkpoint {self.checkpoint_file}, ', end='')
        data = torch.load(self.checkpoint_file, map_location=torch.device('cpu'))
        self.lstm.load_state_dict(data['lstm_params'])
        self.linear.load_state_dict(data['linear_params'])
        print(f'best_accuracy={data["best_acc"][0]:.2f}')
        return data['best_acc'], data['heatmap'], data['fit_result']

    def forward(self, input):
        """
        Forward Pass
        :param input: input batch tensor 
        :return: model prediction tensor for the input 
        """
        lstm_out, (h_n, c_n) = self.lstm(input)
        lstm_out = lstm_out.permute(1, 2, 0)
        max_pool_out = self.max_pool(lstm_out)
        max_pool_out = max_pool_out.squeeze()

        y_pred = self.linear(max_pool_out)
        return y_pred

    def fit(self, dl_train: DataLoader, dl_test: DataLoader, optimizer, loss_fn, max_epochs=4,
            early_stopping: int = None):
        """
        Trains the model for multiple epochs with a given training set,
        and calculates test loss over a given test set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param optimizer: The optimizer to train with.
        :param loss_fn: The loss function to evaluate with.
        :param max_epochs: Number of epochs to train for.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        train_pos_acc, train_neg_acc, test_pos_acc, test_neg_acc = [], [], [], []
        best_acc = None
        epochs_without_improvement = 0
        print(f"{'-'*20}Starting training with overlap {self.overlap}{'-'*20}")

        if self.checkpoint_file and os.path.isfile(self.checkpoint_file):
            best_acc, heat_map_test, fit_result = self.load_checkpoint()
            return fit_result, heat_map_test
            
        for epoch_idx in range(max_epochs):
            print(f'--- EPOCH {epoch_idx + 1}/{max_epochs} ---')
            res_train = self.train_epoch(optimizer, loss_fn, dl_train,)

            res_test, heat_map_test = self.test_epoch(loss_fn, dl_test, True)

            if early_stopping is not None:
                if best_acc is None:
                    best_acc = res_test[1]
                elif res_test[1] <= best_acc:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping:
                        break
                else:
                    epochs_without_improvement = 0
                    best_acc = res_test[1]

            train_loss.append(sum(res_train[0]) / len(res_train[0]))
            train_acc.append(res_train[1])
            test_loss.append(sum(res_test[0]) / len(res_test[0]))
            test_acc.append(res_test[1])

            train_pos_acc.append(res_train[2])
            train_neg_acc.append(res_train[3])
            test_pos_acc.append(res_test[2])
            test_neg_acc.append(res_test[3])

        fit_result = FitResult(max_epochs, train_loss, train_acc, test_loss, test_acc, train_pos_acc, train_neg_acc,
                               test_pos_acc, test_neg_acc)

        if self.checkpoint_file is not None:
            if best_acc is None: best_acc = res_test[1]
            self.save_checkpoint(best_acc, heat_map_test, fit_result)

        return fit_result, heat_map_test

    def train_epoch(self, optimizer, loss_fn, dataloader):
        """
        Train once over a training set (single epoch).
        :param optimizer: The optimizer to train with.
        :param loss_fn: The loss function to evaluate with.
        :param dataloader: DataLoader for the training set.
        :return: An EpochResult for the epoch.
        """
        train_loss, train_acc, losses = [], [], []
        pos_accuracy, neg_accuracy = [], []
        total_loss, num_correct = 0, 0
        tp_tot, fp_tot, tn_tot, fn_tot = 0, 0, 0, 0
        pbar_file = sys.stdout
        pbar_name = "train_batch"
        num_batches = len(dataloader.batch_sampler)
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            for batch_idx, (batch,indices) in enumerate(dataloader):
                X, y = batch[0], batch[1]

                # Forward pass
                X = torch.transpose(X, dim0=0, dim1=1)
                X = X.to(self.device)
                y = y.to(self.device)
                y_pred_log_proba = self.forward(X)
                y = torch.squeeze(y).long()         # should be of size (N,)

                # Backward pass
                optimizer.zero_grad()
                loss = loss_fn(y_pred_log_proba, y)
                loss.backward()

                # Weight updates
                optimizer.step()

                # Calculate accuracy
                total_loss += loss.item()
                y_pred = torch.argmax(y_pred_log_proba, dim=1)
                tp, fp, tn, fn = self.calculate_acc(y_pred, y)
                tp_tot += tp
                fp_tot += fp
                tn_tot += tn
                fn_tot += fn
                num_correct += torch.sum(y_pred == y).float().item()
                total_samp = tp_tot + tn_tot + fp_tot + fn_tot

                pbar.set_description(f'{pbar_name} ({loss.item():.3f})')
                pbar.update()
                losses.append(loss.item())
            accuracy = 100. * num_correct / total_samp
            train_loss.append(sum(losses) / len(losses))
            train_acc.append(accuracy)
            pos = 0 if tp+fn == 0 else 100. * tp/(tp+fn)
            neg = 0 if tn+fp == 0 else 100. * tn/(tn+fp)
            pos_accuracy.append(pos)
            neg_accuracy.append(neg)

        print(f"accuracy={num_correct / total_samp:.3f}, tp: {tp_tot}, fp: {fp_tot}, tn: {tn_tot}, fn: {fn_tot}")
        print(f"Pos acc: {tp_tot / (tp_tot + fn_tot):.3f},  Neg acc: {tn_tot / (tn_tot + fp_tot):.3f}")
        print('---')

        return EpochResult(train_loss, train_acc, pos_accuracy, neg_accuracy)

    def test_epoch(self, loss_fn, dataloader, get_heatmap=False):
        """
        Evaluate model once over a test set (single epoch).
        :param loss_fn: The loss function to evaluate with.
        :param dataloader: DataLoader for the test set.
        :return: An EpochResult for the epoch.
        """
        test_loss, test_acc, losses = [], [], []
        pos_accuracy, neg_accuracy = [], []
        y_vals, attention_map, indices_list = [],[],[]
        total_loss, num_correct = 0, 0
        tp_tot, fp_tot, tn_tot, fn_tot = 0, 0, 0, 0
        pbar_file = sys.stdout
        pbar_name = "test_batch"
        num_batches = len(dataloader.batch_sampler)
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            for batch_idx, (batch,indices) in enumerate(dataloader):
                X, y = batch[0], batch[1]

                # Forward pass    
                with torch.no_grad():
                    X = torch.transpose(X, dim0=0, dim1=1)
                    X = X.to(self.device)
                    y = y.to(self.device)
                    y_pred_log_proba = self.forward(X)
                    if get_heatmap:
                        for i in range(self.batch_size):
                            y_vals.append(int(y[i].item()))
                            attention_map.append(self.soft_attn_weights[i])
                            indices_list.append(indices[i].item())

                    y = torch.squeeze(y).long()

                    loss = loss_fn(y_pred_log_proba, y)

                    total_loss += loss.item()
                    y_pred = torch.argmax(y_pred_log_proba, dim=1)
                    tp, fp, tn, fn = self.calculate_acc(y_pred, y)
                    tp_tot += tp
                    fp_tot += fp
                    tn_tot += tn
                    fn_tot += fn
                    num_correct += torch.sum(y_pred == y).float().item()
                    total_samp = tp_tot + tn_tot + fp_tot + fn_tot

                    pbar.set_description(f'{pbar_name} ({loss.item():.3f})')
                    pbar.update()
                    losses.append(loss.item())

            accuracy = 100. * num_correct / total_samp
            test_loss.append(sum(losses) / len(losses))
            test_acc.append(accuracy)
            pos = 0 if tp+fn == 0 else 100. * tp/(tp+fn)
            neg = 0 if tn+fp == 0 else 100. * tn/(tn+fp)
            pos_accuracy.append(pos)
            neg_accuracy.append(neg)

        print(f"accuracy={num_correct / total_samp:.3f}, tp: {tp_tot}, fp: {fp_tot}, tn: {tn_tot}, fn: {fn_tot}")
        if tp_tot + fn_tot > 0:
            print(f"Pos acc: {tp_tot / (tp_tot + fn_tot):.3f},  Neg acc: {tn_tot / (tn_tot + fp_tot):.3f}")

        return EpochResult(test_loss, test_acc, pos_accuracy, neg_accuracy), EpochHeatMap(y_vals, attention_map, indices_list)

    @staticmethod
    def calculate_acc(y_pred, y):
        """
        Calculates the accuracy of predicted y vector
        """
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(y_pred.shape[0]):
            if y_pred[i] == 0:
                if y[i] == 0:
                    tn += 1
                else: fn += 1
            elif y[i] == 0:
                fp += 1
            else:
                tp += 1
        return tp, fp, tn, fn


class AttentionModel(BaselineModel):
    def __init__(self, config, device, output_dim=2, num_layers=1,checkpoint_file=None):
        """
        Inheritance class from Baseline model.
        Applies an attention layer to the model

        :param config:
        :param device:
        :param output_dim:
        :param num_layers:
        :param checkpoint_file:
        """
        super().__init__(config,device,output_dim,num_layers,checkpoint_file=checkpoint_file)
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim*2, out_features=50),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=50, out_features=self.output_dim),
            nn.Sigmoid()
        )

    def attention_net(self, lstm_output, final_state):
        """ 
        We will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM.
        We will be using torch.bmm for the batch matrix multiplication.
        
        :param lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        :param final_state : Final time-step hidden state (h_n) of the LSTM
        :returns new_hidden_state: It performs attention mechanism by first computing weights for each of the
        sequence present in lstm_output and and then finally computing the new hidden state.
                    
        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)
        """
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        self.soft_attn_weights = nn.functional.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), self.soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input):
        """
        Forward Pass
        :param input: input batch tensor 
        :return: model prediction tensor for the input 
        """
        lstm_out, (h_n, c_n) = self.lstm(input)
        lstm_out = lstm_out.permute(1, 0 ,2)
        h_n = torch.reshape(h_n, (1, self.batch_size, self.hidden_dim*2))
        att_output = self.attention_net(lstm_out, h_n)
        y_pred = self.linear(att_output)
        return y_pred
