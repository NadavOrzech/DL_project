import torch.utils.data
import torch
from data_preprocess import get_data

BATCH_SIZE = 4


def get_dataloader():
    xx, yy = get_data()
    xx = torch.tensor(xx, dtype=torch.float32)
    xx = torch.flatten(xx, start_dim=2) # TODO: maybe flatten differnetly - do first column then second column (and not first second first second..)
    yy = torch.tensor(yy, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(xx, yy)
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)
    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader()
    print(f"num batches: {len(dataloader)}")
