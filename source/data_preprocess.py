import wfdb
import numpy as np
from sklearn.preprocessing import scale
import glob
import os
from typing import List

import torch

BEAT_SIZE = 250
SEQ_SIZE = 100
OVERLAP = 99


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

        self.processed_data_dir = os.path.join(processed_data_dir, "overlap_{}".format(overlap))
        if not os.path.isdir(self.processed_data_dir):
            os.mkdir(self.processed_data_dir)

    def get_beat_list_from_ecg_data(self, datfile):
        recordpath = datfile.split(".dat")[0]
        record = wfdb.rdsamp(recordpath)
        annotation_atr = wfdb.rdann(recordpath, extension='atr', sampfrom=0, sampto=None)
        annotation_qrs = wfdb.rdann(recordpath, extension='qrs', sampfrom=0, sampto=None)
        Vctrecord = record.p_signals
        # wfdb.plotrec(record, annotation=annotation_qrs, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits='seconds')

        beats_list = self.split_to_beats_list(Vctrecord, annotation_qrs, annotation_atr)
        return beats_list

    def split_to_beats_list(self, p_signals, annotation_qrs, annotation_atr) -> List[Beat]:
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

    def split_to_seq(self, beats_list, seq_size, overlap):
        num_big_data = 0
        for j in range(0, len(beats_list) - 5000, seq_size - overlap):
            padded = np.zeros((BEAT_SIZE * seq_size, 2))
            last_idx = 0
            y = 0
            for i in range(j, j + seq_size):
                if beats_list[i].annotation == 1:
                    y = 1
                data = beats_list[i].p_signal
                # this is to check that we made enough space for all the data
                if last_idx + data.shape[0] > BEAT_SIZE * seq_size:
                    a = 0
                if data.shape[0] > 250:
                    num_big_data += 1
                padded[last_idx:last_idx + data.shape[0], :] = data
                last_idx = data.shape[0] + last_idx
                assert last_idx < BEAT_SIZE * seq_size

                # TODO : decide if we need this line:
                # padded = np.expand_dims(padded, 0)  ## add one dimension; so that you get shape (samples,timesteps,features)

            try:
                xx = np.vstack((xx, [padded]))
                yy = np.vstack((yy, [y]))
            except UnboundLocalError:  ## on init
                xx = [padded]
                yy = [y]

        print("output: ", xx.shape)
        return xx, yy

    def split_to_beat_sequence(self, beats_list, seq_size, overlap):
        for j in range(0, len(beats_list) - 5000, seq_size - overlap):
            y = 0
            padded = np.zeros((seq_size, BEAT_SIZE, 2))
            for i in range(seq_size):
                data = beats_list[j + i].p_signal
                # TODO : deal with big beats.. what do we do now??
                min_input = min(BEAT_SIZE, data.shape[0])
                padded[i, :min_input, :] = data[:min_input, :]
                if beats_list[j + i].annotation == 1:
                    y = 1
                # try:
                #     seq_x = np.vstack((seq_x, padded))
                # except UnboundLocalError:
                #     seq_x = padded
            try:
                xx = np.vstack((xx, [padded]))
                yy = np.vstack((yy, [y]))
            except UnboundLocalError:  ## on init
                xx = [padded]
                yy = [y]

        return xx, yy

    def get_data(self):
        # qtdbpath = "C:\\Users\\ronien\\PycharmProjects\\DL_Course\\mit-bih-af\\small_files"
        datfiles = glob.glob(os.path.join(self.input_dir, "*.dat"))

        for i, datfile in enumerate(datfiles):
            print("Starting file num: {}/{}".format(i + 1, len(datfiles)))
            # if i == 10:
            #     break
            file_name = os.path.basename(datfile).split('.')[0]
            x_path = os.path.join(self.processed_data_dir, "{}_x.pt".format(file_name))
            y_path = os.path.join(self.processed_data_dir, "{}_y.pt".format(file_name))
            if os.path.isfile(x_path) and os.path.isfile(y_path):
                x = torch.load(x_path)
                y = torch.load(y_path)
            else:
                qf = os.path.splitext(datfile)[0] + '.atr'
                if os.path.isfile(qf):
                    beats_list = self.get_beat_list_from_ecg_data(datfile)
                    x, y = self.split_to_beat_sequence(beats_list, SEQ_SIZE, self.overlap)
                # TODO : consider normalization of x
                torch.save(x, x_path)
                torch.save(y, y_path)
            try:  # concat
                xx = np.vstack((xx, x))
                yy = np.vstack((yy, y))
            except NameError:  # if xx does not exist yet (on init)
                xx = x
                yy = y

        return xx, yy


'''
    def get_beat_list_from_ecg_data(datfile):
        recordpath = datfile.split(".dat")[0]
        record = wfdb.rdsamp(recordpath)
        annotation_atr = wfdb.rdann(recordpath, extension='atr', sampfrom=0, sampto=None)
        annotation_qrs = wfdb.rdann(recordpath, extension='qrs', sampfrom=0, sampto=None)
        Vctrecord = record.p_signals
        wfdb.plotrec(record, annotation=annotation_qrs, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits='seconds')

        beats_list = split_to_beats(Vctrecord, annotation_qrs, annotation_atr)
        return beats_list


    def split_to_beats(p_signals, annotation_qrs, annotation_atr) -> List[Beat]:
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


    def split_to_seq(beats_list, seq_size, overlap):
        num_big_data = 0
        for j in range(0, len(beats_list) - 5000, seq_size-overlap):
            padded = np.zeros((BEAT_SIZE * seq_size, 2))
            last_idx = 0
            y = 0
            for i in range(j, j+seq_size):
                if beats_list[i].annotation == 1:
                    y = 1
                data = beats_list[i].p_signal
                # this is to check that we made enough space for all the data
                if last_idx + data.shape[0] > BEAT_SIZE * seq_size:
                    a=0
                if data.shape[0] > 250:
                    num_big_data += 1
                padded[last_idx:last_idx+data.shape[0], :] = data
                last_idx = data.shape[0] + last_idx
                assert last_idx < BEAT_SIZE * seq_size

                # TODO : decide if we need this line:
                # padded = np.expand_dims(padded, 0)  ## add one dimension; so that you get shape (samples,timesteps,features)

            try:
                xx = np.vstack((xx, [padded]))
                yy = np.vstack((yy, [y]))
            except UnboundLocalError:  ## on init
                xx = [padded]
                yy = [y]

        print("output: ", xx.shape)
        return xx, yy


    def split_to_beat(beats_list, seq_size, overlap):
        for j in range(0, len(beats_list) - 5000, seq_size - overlap):
            y = 0
            padded = np.zeros((seq_size, BEAT_SIZE, 2))
            for i in range(seq_size):
                data = beats_list[j+i].p_signal
                # TODO : deal with big beats.. what do we do now??
                min_input = min(BEAT_SIZE, data.shape[0])
                padded[i, :min_input, :] = data[:min_input, :]
                if beats_list[j+i].annotation == 1:
                    y = 1
                # try:
                #     seq_x = np.vstack((seq_x, padded))
                # except UnboundLocalError:
                #     seq_x = padded
            try:
                xx = np.vstack((xx, [padded]))
                yy = np.vstack((yy, [y]))
            except UnboundLocalError:  ## on init
                xx = [padded]
                yy = [y]

        return xx, yy


    def get_data(files_dir):
        # qtdbpath = "C:\\Users\\ronien\\PycharmProjects\\DL_Course\\mit-bih-af\\small_files"
        datfiles = glob.glob(os.path.join(files_dir, "*.dat"))

        for datfile in datfiles:
            qf = os.path.splitext(datfile)[0] + '.atr'
            if os.path.isfile(qf):
                beats_list = get_beat_list_from_ecg_data(datfile)
                x, y = split_to_beat(beats_list, SEQ_SIZE, OVERLAP)
                # TODO : consider normalization of x
                try:  # concat
                    xx = np.vstack((xx, x))
                    yy = np.vstack((yy, y))
                except NameError:  # if xx does not exist yet (on init)
                    xx = x
                    yy = y
        return xx, yy
'''

if __name__ == '__main__':
    files_dir = 'C:\\Users\\ronien\\PycharmProjects\\DL_Course\\mit-bih-af\\small_files'
    processed_data_dir = 'C:\\Users\\ronien\\PycharmProjects\\DL_Course\\mit-bih-af\\processed_data_250'
    processor = DataProcessor(files_dir, processed_data_dir, OVERLAP)
    xx, yy = processor.get_data()

