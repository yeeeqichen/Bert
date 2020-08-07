from pytorch_pretrained_bert import BertModel, BertAdam
import torch
from Config import config


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear = torch.nn.Linear(config.embed_size, config.num_class)

    def forward(self, batch):
        text = batch[0]
        mask = batch[2]
        _, pooled = self.bert(text, attention_mask=mask, output_all_encoded_layers=False)
        out = self.linear(pooled)
        return out


