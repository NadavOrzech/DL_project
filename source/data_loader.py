import torch.utils.data
import torch
from data_preprocess import get_data
import numpy as np

BATCH_SIZE = 4
TRAIN_TEST_RATIO = 0.9


def get_dataloader():
    dataset, num_samples, num_pos, num_neg, weight = get_data()
    # xx = torch.tensor(xx, dtype=torch.float32)
    # xx = torch.flatten(xx, start_dim=2) # TODO: maybe flatten differnetly - do first column then second column (and not first second first second..)
    # yy = torch.tensor(yy, dtype=torch.float32)
    #
    # num_samples = xx.shape[0]
    num_train = int(TRAIN_TEST_RATIO * num_samples)

    # num_pos = len([i for i in yy if i == 1])
    # num_neg = len([i for i in yy if i == 0])
    class_sample_count = np.array([num_neg, num_pos])
    # weight = []
    # for label in yy[:num_train]:
    #     if label == 1:
    #         weight.append(1. / num_pos)
    #     else:
    #         weight.append(1. / num_neg)
    # weight = 1. / class_sample_count
    # samples_weight = np.array([weight[t] for t in yy[0]])

    samples_weight = torch.tensor(weight, dtype=torch.float32)
    # samples_weight = samples_weight.double()
    split_lengths = [int(num_samples*TRAIN_TEST_RATIO), int(num_samples*0.1)]
    ds_train, ds_test = torch.utils.data.random_split(dataset, split_lengths)
    pos_indices = ds_train.where()
    # train_dataset = torch.utils.data.TensorDataset(xx[:num_train], yy[:num_train])
    weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_sampler = torch.utils.data.SequentialSampler(dataset)
    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=weighted_sampler,
                                                   shuffle=False, drop_last=True)

    # for i, (data, target) in enumerate(train_dataloader):
    #     print("batch index {}, 0/1: {}/{}".format(
    #         i,
    #         len(np.where(target.numpy() == 0)[0]),
    #         len(np.where(target.numpy() == 1)[0])))
    # test_dataset = torch.utils.data.TensorDataset(dataset)
    test_sampler = torch.utils.data.SequentialSampler(ds_test)
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, sampler=test_sampler,
                                                  shuffle=False, drop_last=True)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloader()
    print(f"num batches: {len(train_dataloader)}")
