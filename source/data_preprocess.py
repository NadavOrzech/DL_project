import wfdb
import numpy as np
import glob
import os
from typing import List
import math
import time
import torch.utils.data

# from config import Config

# BEAT_SIZE = 250
# SEQ_SIZE = 100
# OVERLAP = 0


class Beat():
    """
    Class that represents a single Beat

    :param start_index:
    :param end_index:
    :param p_signal: p_signals from the file record
    :param symbol: label of the beat. i.e "N" , "AFib" ...
    :param index: index of beat in the file
    :param annotation: label of the beat
    """
    def __init__(self,
                 start_index: int,
                 end_index: int,
                 p_signal,
                 symbol: str,
                 index: int,
                 annotation=None):
        self.start_idx = start_index
        self.end_idx = end_index
        self.p_signal = p_signal
        self.symbol = symbol
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
        # wfdb.plotrec(record, annotation=annotation_qrs, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits='seconds')

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
        # TODO : need to decide if we eliminate "bad" signals (annotation_qrs.symbol not N)
        start = 0
        atr_pointer = 0
        beats_list = []
        for i, end in enumerate(annotation_qrs.sample):
            annotation = 0
            if atr_pointer < len(annotation_atr.sample):
                if start <= annotation_atr.sample[atr_pointer] <= end:
                    if annotation_atr.aux_note[atr_pointer] == '(AFIB':
                        annotation = 1
                    atr_pointer += 1
            beat = Beat(start, end, p_signals[start:end], annotation_qrs.symbol[i], i, annotation)
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
        dim_0_size = math.ceil((len(beats_list) - 5000) / (self.seq_size - self.overlap))
        dim_0_counter = 0
        xx = np.zeros((dim_0_size, self.seq_size, self.beat_size, 2))
        yy = np.zeros((dim_0_size, 1))
        for j in range(0, len(beats_list) - 5000, self.seq_size - self.overlap):
            y = 0
            for i in range(self.seq_size):
                data = beats_list[j + i].p_signal
                # TODO : deal with big beats.. what do we do now??
                min_input = min(self.beat_size, data.shape[0])
                xx[dim_0_counter, i, :min_input, :] = data[:min_input, :]
                if beats_list[j + i].annotation == 1:
                    y = 1
            yy[dim_0_counter] = y
            dim_0_counter += 1

        return xx, yy

    def get_data(self):
        """
        Processes the data from self.input_dir and returns a dataset

        :return dataset: dataset type torch.utils.data.ConcatDataset
        """
        datfiles = glob.glob(os.path.join(self.input_dir, "*.dat"))
        start_time = time.time()
        datasets, weight = [], []
        num_samples, num_pos, num_neg = 0, 0, 0
        for i, datfile in enumerate(datfiles):
            print("Starting file num: {}/{}".format(i+1, len(datfiles)))
            qf = os.path.splitext(datfile)[0] + '.atr'
            if os.path.isfile(qf):
                beats_list = self.get_beat_list_from_ecg_data(datfile)
                x, y = self.split_to_beat(beats_list)
                x = torch.tensor(x, dtype=torch.float32)
                # TODO: maybe flatten differently - do first col then second col (and not first second first second..)
                x = torch.flatten(x, start_dim=2)
                y = torch.tensor(y, dtype=torch.float32)
                num_samples += x.shape[0]
                # TODO : consider normalization of x
                datasets.append(torch.utils.data.TensorDataset(x, y))

        dataset = torch.utils.data.ConcatDataset(datasets)
        print(f"elapsed time for preprocess = {time.time() - start_time: .1f} sec")
        return dataset


if __name__ == '__main__':
    files_dir = 'C:\\Users\\ronien\\PycharmProjects\\DL_Course\\mit-bih-af\\files'
    files_dir = 'C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project_data\\mit-bih\\files\\tmp'

    processor = DataProcessor(files_dir, OVERLAP, SEQ_SIZE)
    dataset = processor.get_data()
