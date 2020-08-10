from IPython.display import display

import wfdb


class Record:
    def __init__(self):
        self.file_name = "C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project_data\\mit-bih\\files\\04043"
        self.record = wfdb.rdrecord(self.file_name)

    def visualization(self):
        # wfdb.plot_wfdb(record=self.record, title='test1')
        # display(self.record.__dict__)
        wfdb.plot_all_records("C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project\\mit-bih\\files")


if __name__ == '__main__':
    r = Record()
    r.visualization()