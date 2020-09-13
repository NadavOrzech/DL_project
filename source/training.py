import torch
import torch.nn as nn
import time
from cs236781.train_results import BatchResult, EpochResult, FitResult
from torch.utils.data import DataLoader

import torch.optim as optim
# from data_loader import get_dataloader
# from data_preprocess import BEAT_SIZE, OVERLAP

# HIDDEN_SIZE = 400
# BATCH_SIZE = 4
# BEATS = BEAT_SIZE*2
# NUM_EPOCHS = 4
# DROPOUT = 0.5


class BaselineModel(nn.Module):
    def __init__(self, config, output_dim=2, num_layers=1):
        """
        Custom Bidirectional LSTM module

        :param config:
        :param output_dim:
        :param num_layers:
        """
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = config.dropout
        self.overlap = config.overlap
        self.batch_size = config.batch_size

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, bidirectional=True)
        self.linear = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Linear(in_features=400, out_features=50),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=50, out_features=self.output_dim),
            nn.Sigmoid()
        )
        # TODO: do we need to initialize the hidden dims?

    def forward(self, input):
        lstm_out, hidden_dims = self.lstm(input)
        y_pred = self.linear(lstm_out)
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
        best_acc = None
        epochs_without_improvement = 0
        print(f"{'-'*20}Starting training with overlap {self.overlap}{'-'*20}")
        for epoch_idx in range(max_epochs):
            print(f'--- EPOCH {epoch_idx + 1}/{max_epochs} ---')
            res_train = self.train_epoch(optimizer, loss_fn, dl_train,)
            res_test = self.test_epoch(loss_fn, dl_test)

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

        return FitResult(max_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, optimizer, loss_fn, dataloader):
        """
        Train once over a training set (single epoch).
        :param optimizer: The optimizer to train with.
        :param loss_fn: The loss function to evaluate with.
        :param dataloader: DataLoader for the training set.
        :return: An EpochResult for the epoch.
        """
        train_loss, train_acc, losses = [], [], []
        total_loss, num_correct = 0, 0
        start_time = time.time()
        tp_tot, fp_tot, tn_tot, fn_tot = 0, 0, 0, 0
        for batch_idx, batch in enumerate(dataloader):
            X, y = batch[0], batch[1]

            # Forward pass
            X = torch.transpose(X, dim0=0, dim1=1)
            y_pred_log_proba = self.forward(X)
            last_output = y_pred_log_proba[-1]  # should be of size (N,C) {C is num_classes}
            y = torch.squeeze(y).long()         # should be of size (N,)

            # Backward pass
            optimizer.zero_grad()
            loss = loss_fn(last_output, y)
            loss.backward()

            # Weight updates
            optimizer.step()

            # Calculate accuracy
            total_loss += loss.item()
            y_pred = torch.argmax(last_output, dim=1)
            tp, fp, tn, fn = self.calculate_acc(y_pred, y)
            tp_tot += tp
            fp_tot += fp
            tn_tot += tn
            fn_tot += fn
            num_correct += torch.sum(y_pred == y).float().item()
            total_samp = tp_tot + tn_tot + fp_tot + fn_tot

            losses.append(loss.item())
        accuracy = 100. * num_correct / total_samp
        train_loss.append(sum(losses) / len(losses))
        train_acc.append(accuracy)

        print(
            f"train_batch: loss={total_loss / (total_samp / self.batch_size):.3f}, accuracy={num_correct / total_samp:.3f}, elapsed={time.time() - start_time:.1f} sec")
        print(f"tp: {tp_tot}, fp: {fp_tot}, tn: {tn_tot}, fn: {fn_tot}")
        print(f"Pos acc: {tp_tot / (tp_tot + fn_tot):.3f},  Neg acc: {tn_tot / (tn_tot + fp_tot):.3f}")
        print('---')

        return EpochResult(train_loss, train_acc)

    def test_epoch(self, loss_fn, dataloader):
        """
        Evaluate model once over a test set (single epoch).
        :param loss_fn: The loss function to evaluate with.
        :param dataloader: DataLoader for the test set.
        :return: An EpochResult for the epoch.
        """
        test_loss, test_acc, losses = [], [], []
        total_loss, num_correct = 0, 0
        start_time = time.time()
        tp_tot, fp_tot, tn_tot, fn_tot = 0, 0, 0, 0
        for batch_idx, batch in enumerate(dataloader):
            X, y = batch[0], batch[1]

            # Forward pass
            with torch.no_grad():
                X = torch.transpose(X, dim0=0, dim1=1)
                y_pred_log_proba = self.forward(X)
                last_output = y_pred_log_proba[-1]
                y = torch.squeeze(y).long()

                loss = loss_fn(last_output, y)

                total_loss += loss.item()
                y_pred = torch.argmax(last_output, dim=1)
                tp, fp, tn, fn = self.calculate_acc(y_pred, y)
                tp_tot += tp
                fp_tot += fp
                tn_tot += tn
                fn_tot += fn
                num_correct += torch.sum(y_pred == y).float().item()
                total_samp = tp_tot + tn_tot + fp_tot + fn_tot

                losses.append(loss.item())

        accuracy = 100. * num_correct / total_samp
        test_loss.append(sum(losses) / len(losses))
        test_acc.append(accuracy)
        print(
            f"test_batch: loss={total_loss / (total_samp / self.batch_size):.3f}, accuracy={num_correct / total_samp:.3f}, elapsed={time.time() - start_time:.1f} sec")
        print(f"tp: {tp_tot}, fp: {fp_tot}, tn: {tn_tot}, fn: {fn_tot}")
        if tp_tot + fn_tot > 0:
            print(f"Pos acc: {tp_tot / (tp_tot + fn_tot):.3f},  Neg acc: {tn_tot / (tn_tot + fp_tot):.3f}")

        return EpochResult(test_loss, test_acc)

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


# if __name__ == '__main__':
#     train_dataloader, test_dataloader = get_dataloader()
#     model = LSTM(input_dim=BEATS, hidden_dim=HIDDEN_SIZE)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     model.train(optimizer, loss_fn, train_dataloader, max_epochs=NUM_EPOCHS)
#     model.test(loss_fn, test_dataloader, max_epochs=NUM_EPOCHS)
