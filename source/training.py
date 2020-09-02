import torch
import torch.nn as nn
import time
import torch.optim as optim
from data_loader import get_dataloader
from data_preprocess import BEAT_SIZE

INPUT_SIZE = (100, 4, 1000)
HIDDEN_SIZE = (100, 400)
BATCH_SIZE = 4
BEATS = BEAT_SIZE*2


def train(model, optimizer, loss_fn, dataloader, max_epochs=4, max_batches=200):
    for epoch_idx in range(max_epochs):
        total_loss, num_correct = 0, 0
        start_time = time.time()

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
            num_correct += torch.sum(y_pred == y).float().item()

            if batch_idx == max_batches - 1:
                break

        print(
            f"Epoch #{epoch_idx}, loss={total_loss / (max_batches):.3f}, accuracy={num_correct / (max_batches * BATCH_SIZE):.3f}, elapsed={time.time() - start_time:.1f} sec")


def get_model():
    model = nn.Sequential(
        nn.LSTM(BEATS, 2, 1)
    )
    return model


if __name__ == '__main__':
    dataloader = get_dataloader()
    model = get_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, optimizer, loss_fn, dataloader)

