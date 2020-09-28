
class Config():
    def __init__(self):
        self.beat_size = 200            # size of beat object (p_signals)
        self.seq_size = 100             # number of beats per sequence
        self.overlap = 0               # size of overlap between sequences
        self.train_test_ratio = 0.85    # when training and testing on the same files, ratio of sequences between train to test

        self.hidden_dim = 200               # hidden dimention for LSTM layer
        self.input_dim = self.beat_size*2   # input dimention for LSTM layer
        self.lstm_dropout = 0.1             # dropout value for LSTM layer
        self.dropout = 0.1                  # dropout value for FC layer
        self.lr = 5e-4                      # optimizer learning rate

        self.batch_size = 1024              # training batch size
        self.num_epochs = 40                # max number of epochs
        self.early_stopping = 5             

        # path for directory which holds MIT-BIH Atrial Fibrillation files
        self.files_dir = 'C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project_data\\mit-bih\\files' 