from pytorch_pretrained_bert import BertTokenizer
import torch


class Config:
    def __init__(self):
        self.bert_path = '//data/IE/pytorch_bert_base_chinese'
        # self.bert_path = 'bert-pretrained'
        self.tokenizer = BertTokenizer(self.bert_path + '/vocab.txt')
        self.train_data = 'data/train.csv'
        self.valid_data = 'data/dev.csv'
        self.test_data = 'data/test.csv'
        self.max_length = 30
        self.batch_size = 50
        self.embed_size = 768
        self.num_class = 2
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.EPOCH = 5
        self.learning_rate = 1e-4
        self.lr_decay_rate = 0.96
        self.lr_decay_step = 1000
        self.clip_max_norm = 1


config = Config()

