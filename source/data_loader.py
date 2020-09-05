import torch.utils.data
import torch
from data_preprocess import get_data, SEQ_SIZE

BATCH_SIZE = 4
TRAIN_TEST_RATIO = 0.9


def get_dataloader():
    xx, yy = get_data()
    xx = torch.tensor(xx, dtype=torch.float32)
    xx = torch.flatten(xx, start_dim=2) # TODO: maybe flatten differnetly - do first column then second column (and not first second first second..)
    yy = torch.tensor(yy, dtype=torch.float32)

    num_samples = xx.shape[0]
    num_train = int(TRAIN_TEST_RATIO * num_samples)
    train_dataset = torch.utils.data.TensorDataset(xx[:num_train], yy[:num_train])
    train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                                   shuffle=False, drop_last=True)

    test_dataset = torch.utils.data.TensorDataset(xx[num_train:], yy[num_train:])
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler,
                                                  shuffle=False, drop_last=True)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloader()
    print(f"num batches: {len(train_dataloader)}")
