import torch.utils.data
import torch


class WeightedDataLoader():
    def __init__(self, config, dataset, test_dataset=None):
        self.files_dir = config.files_dir
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.dataset_size = len(self.dataset)
        self.train_test_ratio = config.train_test_ratio
        self.batch_size = config.batch_size

        # if dataset is None we take the train/test datasets randomly from the entire dataset
        # else we produce the train/test datasets from different files
        if test_dataset is None:
            split_lengths = [int(self.dataset_size*self.train_test_ratio), self.dataset_size-int(self.dataset_size*self.train_test_ratio)]
            self.ds_train, self.ds_test = torch.utils.data.random_split(self.dataset, split_lengths)
        else:   
            self.ds_train, _ = torch.utils.data.random_split(dataset, [self.dataset_size, 0])
            self.ds_test, _ = torch.utils.data.random_split(test_dataset, [len(test_dataset), 0])

    def get_test_dataloader(self):
        subset_idx = self.ds_test.indices
        num_pos = len([i for i in subset_idx if self.test_dataset[i][0][1] == 1])
        num_neg = len(subset_idx) - num_pos
        print(f"Num pos in test: {num_pos}, Num neg in test: {num_neg}")
        #
        # class_sample_count = torch.tensor([num_neg, num_pos])
        # weight = 1. / class_sample_count.float()
        #
        # samples_weight = []
        # for i in subset_idx:
        #     t = int(self.test_dataset[i][0][1])
        #     samples_weight.append(weight[t])
        #
        # samples_weight = torch.tensor(samples_weight)
        # weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        #
        # test_dataloader = torch.utils.data.DataLoader(self.ds_test,
        #                                               batch_size=self.batch_size,
        #                                               sampler=weighted_sampler,
        #                                               shuffle=False, drop_last=True)

        test_sampler = torch.utils.data.SequentialSampler(self.ds_test)
        test_dataloader = torch.utils.data.DataLoader(self.ds_test, batch_size=self.batch_size, sampler=test_sampler,
                                                      shuffle=False, drop_last=True)

        return test_dataloader

    def get_train_dataloader(self):
        """
        Splits (by self.train_test_ration) data to training dataset and test dataset types torch.utils.data.Dataset
        For training Dataloader uses torch.utils.data.WeightedRandomSampler sampler (in order to balance the data)
        For test DataLoader uses torch.utils.data.SequentialSampler sampler
        :return train_dataloader: type torch.utils.data.Dataloader
                test_dataloader: type torch.utils.data.Dataloader
        """
        subset_idx = self.ds_train.indices
        num_pos = len([i for i in subset_idx if self.dataset[i][0][1] == 1])
        num_neg = len(subset_idx)-num_pos
        print(f"Num pos in train: {num_pos}, Num neg in train: {num_neg}")
        #
        # class_sample_count = torch.tensor([num_neg,num_pos])
        # weight = 1. / class_sample_count.float()
        #
        # samples_weight = []
        # for i in subset_idx:
        #     t = int(self.dataset[i][0][1])
        #     samples_weight.append(weight[t])
        #
        # samples_weight = torch.tensor(samples_weight)
        # weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        #
        # train_dataloader = torch.utils.data.DataLoader(self.ds_train, batch_size=self.batch_size, sampler=weighted_sampler,
        #                                                shuffle=False, drop_last=True)

        train_sampler = torch.utils.data.SequentialSampler(self.ds_train)
        train_dataloader = torch.utils.data.DataLoader(self.ds_train, batch_size=self.batch_size, sampler=train_sampler,
                                                      shuffle=False, drop_last=True)
        return train_dataloader

