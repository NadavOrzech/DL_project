import torch.utils.data
import torch
# from data_preprocess import SEQ_SIZE, OVERLAP
from data_preprocess import DataProcessor
# BATCH_SIZE = 4
# TRAIN_TEST_RATIO = 0.9


class ECGDataLoader():
    def __init__(self, config):
        self.files_dir = config.files_dir
        self.processor = DataProcessor(config.files_dir, config.overlap, config.seq_size, config.beat_size)
        self.dataset = self.processor.get_data()
        self.dataset_size = len(self.dataset)
        self.train_test_ratio = config.train_test_ratio
        self.batch_size = config.batch_size

    def get_dataloader(self):
        # files_dir = 'C:\\Users\\ronien\\PycharmProjects\\DL_Course\\mit-bih-af\\small_files'
        # # files_dir = 'C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project_data\\mit-bih\\files\\tmp'

        # processor = DataProcessor(files_dir, OVERLAP, SEQ_SIZE)
        # dataset = processor.get_data()
        
        # dataset_size = len(dataset)
        split_lengths = [int(self.dataset_size*self.train_test_ratio), self.dataset_size-int(self.dataset_size*self.train_test_ratio)]
        ds_train, ds_test = torch.utils.data.random_split(self.dataset, split_lengths)
        
        subset_idx = ds_train.indices
        num_pos = len([i for i in subset_idx if self.dataset[i][1]==1])
        num_neg = len(subset_idx)-num_pos

        class_sample_count = torch.tensor([num_neg,num_pos])
        weight = 1. / class_sample_count.float()

        samples_weight = []
        for i in subset_idx:
            t = int(self.dataset[i][1])
            samples_weight.append(weight[t])

        samples_weight = torch.tensor(samples_weight)
        # samples_weigth = samples_weight.double()
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        
        train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=self.batch_size, sampler=weighted_sampler,
                                                    shuffle=False, drop_last=True)

        test_sampler = torch.utils.data.SequentialSampler(ds_test)
        test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=self.batch_size, sampler=test_sampler,
                                                    shuffle=False, drop_last=True)
        return train_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloader()
    print(f"num batches: {len(train_dataloader)}")
