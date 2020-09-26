import wfdb
import numpy as np
import glob
import os
from typing import List
import math
import time
import torch.utils.data
import bisect


class Beat:
    """
    Class that represents a single Beat

    :param start_index: index for the first signal relevant to the current beat
    :param end_index: index for the last signal relevant to the current beat
    :param p_signal: p_signals from the file record
    :param index: index of beat in the file
    :param annotation: label of the beat
    """
    def __init__(self,
                 start_index: int,
                 end_index: int,
                 p_signal,
                 index: int,
                 annotation=None):
        self.start_idx = start_index
        self.end_idx = end_index
        self.p_signal = p_signal
        self.index = index
        self.annotation = annotation


class DataProcessor():
    def __init__(self, input_dir, overlap, seq_size, beat_size):
        """
        Class that does the data preprocessing from MIT BIH AF db.
        receives path to files from the db and returns the data processed into RR intervals and labeled

        :param input_dir: path to files from db
        :param overlap: overlap between adjacent sequences
        :param seq_size: number of beats in a sequence
        """
        self.input_dir = input_dir
        self.overlap = overlap
        self.seq_size = seq_size
        self.beat_size = beat_size

    def get_beat_list_from_ecg_data(self, datfile) -> List[Beat]:
        """
        Given a file name, splits the data to RR intervals and saves a Beat list of the data

        :param datfile: path to data file
        :return: list of type beat holding beats from file ordered chronology
        """
        recordpath = datfile.split(".dat")[0]
        record = wfdb.rdsamp(recordpath)
        annotation_atr = wfdb.rdann(recordpath, extension='atr', sampfrom=0, sampto=None)
        annotation_qrs = wfdb.rdann(recordpath, extension='qrs', sampfrom=0, sampto=None)
        Vctrecord = record.p_signals
        beats_list = self.get_RR_intervals(Vctrecord, annotation_qrs, annotation_atr)
        return beats_list

    @staticmethod
    def get_RR_intervals(p_signals, annotation_qrs, annotation_atr) -> List[Beat]:
        """
        Splits the data into RR intervals

        :param p_signals: p_signals from the file record
        :param annotation_qrs: data from qrs file - beat data
        :param annotation_atr: data from atr file - rhythm data
        :return beats_list: list of type beat holding beats from file ordered chronology
        """
        start = annotation_qrs.sample[0]
        atr_pointer = 1
        curr_note = 0 if annotation_atr.aux_note[0] == '(N' else 1
        beats_list = []
        for i, end in enumerate(annotation_qrs.sample[1:]):
            if atr_pointer == len(annotation_atr.sample) or end < annotation_atr.sample[atr_pointer]:
                beat = Beat(start, end, p_signals[start:end], i, curr_note)
            else:
                curr_note = 0 if annotation_atr.aux_note[atr_pointer] == '(N' else 1
                atr_pointer += 1
                beat = Beat(start, end, p_signals[start:end], i, curr_note)
            beats_list.append(beat)
            start = end
        return beats_list

    def split_to_beat(self, beats_list):
        """
        Creates a tensor holding the RR intervals of the beats

        :param beats_list: list[Beat] of the data
        :return xx: tensor of shape (N, S, B, 2) holding the RR intervals of the data
         N: number of sequences, S: seq_size, B: beat_size
        :return yy: tensor of shape (N, 1) holding the labeling of the data
        """
        dim_0_size = math.ceil(len(beats_list) / (self.seq_size - self.overlap))
        dim_0_counter = 0
        xx = np.zeros((dim_0_size, self.seq_size, self.beat_size, 2))
        yy = np.zeros((dim_0_size, 1))
        zz = np.zeros((dim_0_size, self.seq_size))
        for j in range(0, len(beats_list) - self.seq_size, self.seq_size - self.overlap):
            y = 0
            for i in range(self.seq_size):
                data = beats_list[j + i].p_signal
                min_input = min(self.beat_size, data.shape[0])
                xx[dim_0_counter, i, :min_input, :] = data[:min_input, :]
                zz[dim_0_counter, i] = int(beats_list[j + i].annotation)
                if beats_list[j + i].annotation == 1:
                    y = 1
            yy[dim_0_counter] = y
            dim_0_counter += 1

        return xx, yy, zz

    def get_data(self, start_file=0, end_file=0):
        """
        Processes the data from self.input_dir and returns a dataset

        :return dataset: dataset type torch.utils.data.ConcatDataset
        """
        if not os.path.isdir(os.path.join('.', 'dataset_checkpoints')):
            os.mkdir(os.path.join('.', 'dataset_checkpoints'))

        suffix=''
        if start_file == 0 and end_file != 0:
            suffix = '_train'
        elif start_file != 0:
            suffix = '_test'

        seq_file = os.path.join('dataset_checkpoints', 'seq_dataset_{}{}'.format(self.overlap,suffix))

        datfiles = glob.glob(os.path.join(self.input_dir, "*.dat"))
        start_time = time.time()
        datasets, weight = [], []
        seq_datasets = []
        num_samples, num_pos, num_neg = 0, 0, 0
        
        if end_file==0:
            end_file = len(datfiles)
        
        for i, datfile in enumerate(datfiles[start_file:end_file]):
            print("Starting file num: {}/{}".format(i+1, end_file-start_file))
            qf = os.path.splitext(datfile)[0] + '.atr'
            if os.path.isfile(qf):
                beats_list = self.get_beat_list_from_ecg_data(datfile)
                x, y, z = self.split_to_beat(beats_list)
                x = torch.tensor(x, dtype=torch.float32)
                x = torch.flatten(x, start_dim=2)
                y = torch.tensor(y, dtype=torch.float32)
                num_samples += x.shape[0]
                print(f"number of sequences: {x.shape[0]}")
                datasets.append(torch.utils.data.TensorDataset(x, y))
                seq_datasets.append((z))

        dataset = IndicesDataset(datasets)
        seq_dataset = torch.utils.data.ConcatDataset(seq_datasets)

        print(f"elapsed time for preprocess = {time.time() - start_time: .1f} sec")
        print(f"total number of sequences: {num_samples}")
        torch.save(seq_dataset, seq_file)
        return dataset,seq_dataset


class IndicesDataset(torch.utils.data.ConcatDataset):
    """
    Class that inherite from torch.utils.data.ConcatDataset and behave the same,
        the __getitem__ function returns the tuple (sample data, sample index) as opposed to ConcatDataset
        which return the sample data alone
    """
    def __init__(self, datasets):
        super().__init__(datasets=datasets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx], idx
