import wfdb
import numpy as np
from sklearn.preprocessing import scale
import glob, os
from typing import List

BEAT_SIZE = 400
SEQ_SIZE = 100
OVERLAP = 0


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


def get_beat_list_from_ecg_data(datfile):
    recordpath = datfile.split(".dat")[0]
    record = wfdb.rdsamp(recordpath)
    annotation_atr = wfdb.rdann(recordpath, extension='atr', sampfrom=0, sampto=None)
    annotation_qrs = wfdb.rdann(recordpath, extension='qrs', sampfrom=0, sampto=None)
    Vctrecord = record.p_signals
    # wfdb.plotrec(record, annotation=annotation_qrs, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits='seconds')

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
        if annotation_qrs.sample[i] == 8775761:
            a=0
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


def get_data():
    qtdbpath = "C:\\Users\\ronien\\PycharmProjects\\DL_Course\\mit-bih-af\\small_files"
    datfiles = glob.glob(os.path.join(qtdbpath, "*.dat"))

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


if __name__ == '__main__':
    xx, yy = get_data()

