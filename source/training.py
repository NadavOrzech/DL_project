import torch
import torch.nn as nn
import time
import torch.optim as optim
from data_loader import get_dataloader
from data_preprocess import BEAT_SIZE

INPUT_SIZE = (100, 4, BEAT_SIZE*2)
HIDDEN_SIZE = (100, BEAT_SIZE)
BATCH_SIZE = 4
BEATS = BEAT_SIZE*2
NUM_EPOCHS = 4


def train(model, optimizer, loss_fn, dataloader, max_epochs=4, max_batches=200):
    print("-----------------------------Starting training------------------------------")
    for epoch_idx in range(max_epochs):
        total_loss, num_correct = 0, 0
        start_time = time.time()
        tp_tot, fp_tot, tn_tot, fn_tot = 0, 0, 0, 0
        for batch_idx, batch in enumerate(dataloader):
            X, y = batch[0], batch[1]

            # Forward pass
            X = torch.transpose(X, dim0=0, dim1=1)
            y_pred_log_proba, hidden_dims = model(X)
            last_output = y_pred_log_proba[-1]
            y = torch.squeeze(y).long()
            # Backward pass
            optimizer.zero_grad()
            loss = loss_fn(last_output, y)
            loss.backward()

            # Weight updates
            optimizer.step()

            # Calculate accuracy
            total_loss += loss.item()
            y_pred = torch.argmax(last_output, dim=1)
            tp, fp, tn, fn = calculate_acc(y_pred, y)
            tp_tot += tp
            fp_tot += fp
            tn_tot += tn
            fn_tot += fn
            num_correct += torch.sum(y_pred == y).float().item()
            total_samp = tp_tot + tn_tot + fp_tot + fn_tot
            # if batch_idx == max_batches - 1:
            #     break

        print(
            f"Epoch #{epoch_idx}, loss={total_loss / (total_samp / BATCH_SIZE):.3f}, accuracy={num_correct / total_samp:.3f}, elapsed={time.time() - start_time:.1f} sec")
        print(f"tp: {tp_tot}, fp: {fp_tot}, tn: {tn_tot}, fn: {fn_tot}")
        print(f"Positive accuracy: {tp_tot / (tp_tot + fn_tot)}")
        print(f"Negative accuracy: {tn_tot / (tn_tot + fp_tot)}")


def test(model, loss_fn, dataloader, max_epochs=4, max_batches=200):
    print("-----------------------------Starting testing------------------------------")
    # for epoch_idx in range(max_epochs):
    total_loss, num_correct = 0, 0
    start_time = time.time()
    tp_tot, fp_tot, tn_tot, fn_tot = 0, 0, 0, 0
    for batch_idx, batch in enumerate(dataloader):
        X, y = batch[0], batch[1]

        # Forward pass
        with torch.no_grad():
            X = torch.transpose(X, dim0=0, dim1=1)
            y_pred_log_proba, hidden_dims = model(X)
            last_output = y_pred_log_proba[-1]
            y = torch.squeeze(y).long()

            loss = loss_fn(last_output, y)

            total_loss += loss.item()
            y_pred = torch.argmax(last_output, dim=1)
            tp, fp, tn, fn = calculate_acc(y_pred, y)
            tp_tot += tp
            fp_tot += fp
            tn_tot += tn
            fn_tot += fn
            num_correct += torch.sum(y_pred == y).float().item()
            total_samp = tp_tot + tn_tot + fp_tot + fn_tot

    print(
        f"loss={total_loss / (total_samp / BATCH_SIZE):.3f}, accuracy={num_correct / total_samp:.3f}, elapsed={time.time() - start_time:.1f} sec")
    print(f"tp: {tp_tot}, fp: {fp_tot}, tn: {tn_tot}, fn: {fn_tot}")
    if tp_tot + fn_tot > 0:
        print(f"Positive accuracy: {tp_tot / (tp_tot + fn_tot)}")
    print(f"Negative accuracy: {tn_tot / (tn_tot + fp_tot)}")


def get_model():
    model = nn.Sequential(
        nn.LSTM(BEATS, 2, 1)
    )
    return model


def calculate_acc(y_pred, y):
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


if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloader()
    model = get_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, optimizer, loss_fn, train_dataloader, max_epochs=NUM_EPOCHS)
    test(model, loss_fn, test_dataloader, max_epochs=NUM_EPOCHS)

