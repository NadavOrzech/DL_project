import wfdb
import numpy as np
from sklearn.preprocessing import scale
import glob, os
from typing import List
import math
import time
import torch.utils.data

BEAT_SIZE = 250
SEQ_SIZE = 100
OVERLAP = 95


class Beat:
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
    def __init__(self, input_dir, processed_data_dir, overlap):
        self.input_dir = input_dir
        if not os.path.isdir(processed_data_dir):
            os.mkdir(processed_data_dir)
        
        self.overlap = overlap

        self.processed_data_dir = os.path.join(processed_data_dir,"overlap_{}".format(overlap))
        if not os.path.isdir(self.processed_data_dir):
            os.mkdir(self.processed_data_dir)

    def get_beat_list_from_ecg_data(self, datfile):
        # from given datfile name, splits to RR intervals and save the data on Beat list
        recordpath = datfile.split(".dat")[0]
        record = wfdb.rdsamp(recordpath)
        annotation_atr = wfdb.rdann(recordpath, extension='atr', sampfrom=0, sampto=None)
        annotation_qrs = wfdb.rdann(recordpath, extension='qrs', sampfrom=0, sampto=None)
        Vctrecord = record.p_signals
        # wfdb.plotrec(record, annotation=annotation_qrs, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits='seconds')

        beats_list = self.get_RR_intervals(Vctrecord, annotation_qrs, annotation_atr)
        return beats_list


    def get_RR_intervals(self, p_signals, annotation_qrs, annotation_atr) -> List[Beat]:
        # split the file data into RR intervals
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
            # if annotation_qrs.sample[i] == 8775761:
            #     a=0
            beats_list.append(beat)
            start = end
        return beats_list

    def split_to_beat(self, beats_list, seq_size, overlap):
        # create a single beat
        dim_0_size = math.ceil((len(beats_list) - 5000) / (seq_size - overlap))
        dim_0_counter = 0
        xx = np.zeros((dim_0_size, seq_size, BEAT_SIZE, 2))
        yy = np.zeros((dim_0_size, 1))
        for j in range(0, len(beats_list) - 5000, seq_size - overlap):
            y = 0
            for i in range(seq_size):
                data = beats_list[j + i].p_signal
                # TODO : deal with big beats.. what do we do now??
                min_input = min(BEAT_SIZE, data.shape[0])
                xx[dim_0_counter, i, :min_input, :] = data[:min_input, :]
                if beats_list[j + i].annotation == 1:
                    y = 1
            yy[dim_0_counter] = y
            dim_0_counter += 1

            # try:
            # xx = np.vstack((xx, [padded]))
            # yy = np.vstack((yy, [y]))
            # except UnboundLocalError:  ## on init
            # xx = [padded]
            # yy = [y]

        return xx, yy


    def get_data(self):
        # qtdbpath = "C:\\Users\\ronien\\PycharmProjects\\DL_Course\\mit-bih-af\\small_files"
        datfiles = glob.glob(os.path.join(self.input_dir, "*.dat"))
        start_time = time.time()
        datasets, weight = [], []
        num_samples, num_pos, num_neg = 0, 0, 0
        for i, datfile in enumerate(datfiles):
            print("Starting file num: {}/{}".format(datfile, len(datfiles)))
            qf = os.path.splitext(datfile)[0] + '.atr'
            if os.path.isfile(qf):
                beats_list = self.get_beat_list_from_ecg_data(datfile)
                x, y = self.split_to_beat(beats_list, SEQ_SIZE, OVERLAP)
                yy = y
                x = torch.tensor(x, dtype=torch.float32)
                x = torch.flatten(x, start_dim=2)  # TODO: maybe flatten differnetly - do first column then second column (and not first second first second..)
                y = torch.tensor(y, dtype=torch.float32)
                num_samples += x.shape[0]
                num_pos += len([i for i in y if i == 1])
                num_neg += len([i for i in y if i == 0])
                # TODO : consider normalization of x
                datasets.append(torch.utils.data.TensorDataset(x, y))
                try:  # concat
                    labels = np.vstack((labels, yy))
                except UnboundLocalError:  # if xx does not exist yet (on init)
                    labels = yy
        for label in labels:
            if label == 1:
                weight.append(1. / num_pos)
            else:
                weight.append(1. / num_neg)
        dataset = torch.utils.data.ConcatDataset(datasets)
        print(f"elapsed time for preprocess = {time.time() - start_time: .1f} sec")
        return dataset#, num_samples, num_pos, num_neg, weight




# def split_to_seq(self, beats_list, seq_size, overlap):
    #     num_big_data = 0
    #     for j in range(0, len(beats_list) - 5000, seq_size-overlap):
    #         padded = np.zeros((BEAT_SIZE * seq_size, 2))
    #         last_idx = 0
    #         y = 0
    #         for i in range(j, j+seq_size):
    #             if beats_list[i].annotation == 1:
    #                 y = 1
    #             data = beats_list[i].p_signal
    #             # this is to check that we made enough space for all the data
    #             if last_idx + data.shape[0] > BEAT_SIZE * seq_size:
    #                 a=0
    #             if data.shape[0] > 250:
    #                 num_big_data += 1
    #             padded[last_idx:last_idx+data.shape[0], :] = data
    #             last_idx = data.shape[0] + last_idx
    #             assert last_idx < BEAT_SIZE * seq_size

    #             # TODO : decide if we need this line:
    #             # padded = np.expand_dims(padded, 0)  ## add one dimension; so that you get shape (samples,timesteps,features)

    #         try:
    #             xx = np.vstack((xx, [padded]))
    #             yy = np.vstack((yy, [y]))
    #         except UnboundLocalError:  ## on init
    #             xx = [padded]
    #             yy = [y]

    #     print("output: ", xx.shape)
    #     return xx, yy

if __name__ == '__main__':
    xx, yy = get_data()

