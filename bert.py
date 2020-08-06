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


if __name__ == '__main__':
    """
    记录了模型的使用方法
    一定要使用BertAdam！！！
    """
    model = Model()
    import DataLoader
    loader = DataLoader.DataGenerator()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    t_total = (len(loader.train_dataset) // config.batch_size) * config.EPOCH
    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=config.learning_rate,
                             warmup=0.05,
                             t_total=t_total)


