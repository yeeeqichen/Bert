from Config import config
import csv
import numpy
import torch
PAD, CLS = '[PAD]', '[CLS]'
csv.field_size_limit(1000000)


def load_dataset():
    def helper(path):
        dataset = []
        with open(path) as f:
            reader = csv.reader(f)
            for line in reader:
                _, label, text = line[0].split('\t')
                if text == 'text':
                    continue
                tokens = [CLS] + config.tokenizer.tokenize(text)
                seq_len = len(tokens)
                ids = config.tokenizer.convert_tokens_to_ids(tokens)
                if seq_len < config.max_length:
                    mask = [1] * seq_len + [0] * (config.max_length - seq_len)
                    ids += ([0] * (config.max_length - seq_len))
                else:
                    mask = [1] * config.max_length
                    ids = ids[:config.max_length]
                    seq_len = config.max_length
                dataset.append((ids, int(label), seq_len, mask))
        return dataset

    train_dataset = helper(config.train_data)
    valid_dataset = helper(config.valid_data)
    test_dataset = helper(config.test_data)
    return train_dataset, valid_dataset, test_dataset


class DataGenerator:
    def __init__(self):
        self.train_dataset, self.valid_dataset, self.test_dataset = load_dataset()

    def run(self, mode):
        def _to_tensor():
            nonlocal dataset
            nonlocal begin
            nonlocal end
            data = dataset[begin:end]
            x = torch.LongTensor([_[0] for _ in data]).to(config.device)
            y = torch.LongTensor([_[1] for _ in data]).to(config.device)
            seq_len = torch.LongTensor([_[2] for _ in data]).to(config.device)
            mask = torch.LongTensor([_[3] for _ in data]).to(config.device)
            return (x, seq_len, mask), y

        if mode == 'train':
            dataset = self.train_dataset
        elif mode == 'valid':
            dataset = self.valid_dataset
        elif mode == 'test':
            dataset = self.test_dataset
            begin = 0
            end = len(dataset)
            return _to_tensor()
        else:
            raise Exception("please clarify mode!")
        for i in range(len(dataset) // config.batch_size):
            begin = i * config.batch_size
            end = (i + 1) * config.batch_size
            yield _to_tensor()


# loader = DataGenerator()
# for i, j in loader.run('train'):
#     print(i)
#     print(j)
#     break
