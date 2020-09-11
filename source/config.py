
class Config():
    def __init__(self):
        self.beat_size = 250
        self.seq_size = 100
        self.overlap = 0
        self.train_test_ratio = 0.9

        self.hidden_dim = 400
        self.input_dim = self.beat_size*2
        self.dropout = 0.1
        self.lr = 1e-3

        self.batch_size = 4
        self.num_epochs = 4

        self.files_dir = 'C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project_data\\mit-bih\\files\\tmp'