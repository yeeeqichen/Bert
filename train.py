import DataLoader
from bert import Model
from Config import config
from pytorch_pretrained_bert import BertAdam
from sys import argv
import torch
import time


def train():
    model = Model()
    model.to(config.device)
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
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay_rate)
    for epoch in range(config.EPOCH):
        print("current epoch: {}".format(epoch))
        print(time.ctime(time.time()))
        step = 0
        for x, y in loader.run('train'):
            output = model(x)
            loss = loss_fn(output, y)
            model.zero_grad()
            loss.backward()
            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_max_norm, norm_type=2)
            optimizer.step()
            # print(loss)
            step += 1
            if step % 10 == 0:
                for valid_x, valid_y in loader.run('valid'):
                    logits = model(valid_x)
                    loss = loss_fn(logits, valid_y)
                    hit = 0
                    print("valid set loss: ", loss)
                    labels = valid_y.cpu().numpy()
                    predict = torch.argmax(torch.nn.functional.softmax(logits), dim=1).cpu().numpy()
                    for i, j in zip(labels, predict):
                        if i == j:
                            hit += 1
                    print("valid set precision: ", hit / config.batch_size)


train()

