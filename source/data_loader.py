import torch.utils.data
import torch
from data_preprocess import get_data

BATCH_SIZE = 100


def get_dataloader():
    xx, yy = get_data()
    xx = torch.tensor(xx)
    yy = torch.tensor(yy)
    dataset = torch.utils.data.TensorDataset(xx, yy)
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)
    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader()
    print(f"num batches: {len(dataloader)}")
