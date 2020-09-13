
class Config():
    def __init__(self):
        self.beat_size = 200
        self.seq_size = 100
        self.overlap = 99
        self.train_test_ratio = 0.85

        self.hidden_dim = 400
        self.input_dim = self.beat_size*2
        self.dropout = 0.1
        self.lr = 1e-3

        self.batch_size = 4
        self.num_epochs = 10
        self.early_stopping = 2

        self.files_dir = '/Users/roenglen/Documents/files'
        # files_dir = 'C:\\Users\\ronien\\PycharmProjects\\DL_Course\\mit-bih-af\\small_files'
        # files_dir = 'C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project_data\\mit-bih\\files\\tmp'