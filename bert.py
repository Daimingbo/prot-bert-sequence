# -*- coding: utf-8 -*-
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

class MyDataset(Dataset):
    def __init__(self, sentences,targets):
        self._X = sentences
        self._y = targets

    def __getitem__(self, index):
        return self._X[index], self._y[index]

    def __len__(self):
        return len(self._X)


class BertClassification(nn.Module):
    def __init__(self, num_labels):
        super(BertClassification, self).__init__()
        self.model_name = './bert-model'
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.fc = nn.Linear(3, num_labels)

    def forward(self, x):
        batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
                                                           max_length=512, padding='max_length',
                                                           truncation='longest_first')
        input_ids = torch.tensor(batch_tokenized['input_ids']).to(device)
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(device)
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        output = self.fc(logits)
        return output
def evaluate_accuracy(inputs, labels):
    inputs = inputs
    labels = labels.long().tolist() #标签
    predictions = model(inputs).argmax(dim=1).tolist() #模型出来的结果 预测
    acc = accuracy_score(labels, predictions)
    pre = precision_score(labels, predictions, average='macro')
    rec = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    return acc, pre, rec, f1
if __name__ == '__main__':
    # #读取数据
    comments1 = pd.read_csv('./data/sequence_2400_train_val.csv', encoding='utf-8')['Sequence'].values.tolist()
    comments1 = [' '.join(sequence) for sequence in comments1]
    labels1 = pd.read_csv('./data/sequence_2400_train_val.csv', encoding='utf-8')['label'].values
    # 替换操作
    labels1 = [2 if label == 'GO:0006393' else label for label in labels1]
    labels1 = [1 if label == 'GO:0006391' else label for label in labels1]
    labels1 = [0 if label == 'GO:0006390' else label for label in labels1]
    comments0 = pd.read_csv('./data/sequence_600_test.csv', encoding='utf-8')['Sequence'].values.tolist()
    comments0 = [' '.join(sequence) for sequence in comments0]
    labels0 = pd.read_csv('./data/sequence_600_test.csv', encoding='utf-8')['label'].values
    # 替换操作
    labels0 = [2 if label == 'GO:0006393' else label for label in labels0]
    labels0 = [1 if label == 'GO:0006391' else label for label in labels0]
    labels0 = [0 if label == 'GO:0006390' else label for label in labels0]
    # 打印替换后的结果
    comments = comments1+comments0
    # # print(len(comments))
    labels = labels1+labels0
    # # print(len(labels))
    # # 构建数据集
    # # 划分数据集 8：1：1 训练集 测试集 验证集
    train_features, test_features, train_targets, test_targets = train_test_split(comments, labels, test_size=0.2,
                                                                         shuffle=True, random_state=45)
    test_features, val_features, test_targets, val_targets = train_test_split(test_features, test_targets,
                                                                              test_size=0.5,
                                                                              shuffle=True, random_state=45)
    # 封装数据
    train_dataset = MyDataset(comments1, labels1)
    train_dataload = DataLoader(train_dataset, shuffle=True, batch_size=2)
    val_dataset = MyDataset(comments0, labels0)
    val_dataload = DataLoader(val_dataset, shuffle=True, batch_size=2)
    test_dataset = MyDataset(test_features, test_targets)
    test_dataload = DataLoader(test_dataset, shuffle=True, batch_size=2)
    # # 定义超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    num_epochs = 5  # 训练轮数
    model = BertClassification(num_labels=3).to(device)
    loss_function = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 定义优化器
    los = 0.0  # 损失，写在这里，因为每一轮训练的损失都好应该重新开始计数
    f1_max = 0.0  # 最好的f1值
    print(len(train_dataload))
    for _ in range(num_epochs):
        for batch_idx, (inputs, labels) in enumerate(train_dataload):  # 刚才说了batch_count的意思有多少块（段），每段有16句话
            inputs = inputs
            targets = labels.long().to(device)
            optimizer.zero_grad()  # 1.梯度置零
            outputs = model(inputs)  # 2.模型获得结果
            loss = loss_function(outputs, targets)  # 3.计算损失
            loss.backward()  # 4.反向传播
            optimizer.step()  # 5.修改参数
            los += loss.item()  # item()返回loss的值
            # 下面每处理100个段，我们看一下当前损失是多少
            if batch_idx % 100 == 0:
                acc_val = 0.0
                pre_val = 0.0
                rec_val = 0.0
                f1_val = 0.0
                # 整个验证集的数据
                for batch_idx, (inputs, labels) in enumerate(val_dataload):
                    acc, pre, rec, f1 = evaluate_accuracy(inputs, labels)
                    acc_val += acc
                    pre_val += pre
                    rec_val += rec
                    f1_val += f1
                acc_val = acc_val / len(val_dataload)
                pre_val = pre_val / len(val_dataload)
                rec_val = rec_val / len(val_dataload)
                f1_val = f1_val / len(val_dataload)
                print(
                    f'loss:{los / 100:.5f},acc_val:{acc_val:.5f},pre_val:{pre_val:.5f},rec_val:{rec_val:.5f},f1_val:{f1_val:.5f}')
                # 当超过最大f1值时，重新保存最佳的模型
                if f1_val >= f1_max:
                    f1_max = f1_val
                    torch.save(model, 'models/bert_model.pkl') #保存模型
                los = 0.0
    # # 测试集
    # # 调用保存好的效果最佳模型
    model1 = torch.load('models/bert_model.pkl')
    acc_test = 0.0
    pre_test = 0.0
    rec_test = 0.0
    f1_test = 0.0
    for batch_idx, (inputs, labels) in enumerate(test_dataload):
        predictions = model1(inputs).argmax(dim=1).tolist()  # 模型出来的结果 预测
        acc = accuracy_score(labels, predictions)
        pre = precision_score(labels, predictions, average='macro')
        rec = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')
        acc_test += acc
        pre_test += pre
        rec_test += rec
        f1_test += f1
    acc_test = acc_test / len(test_dataload)
    pre_test = pre_test / len(test_dataload)
    rec_test = rec_test / len(test_dataload)
    f1_test = f1_test / len(test_dataload)
    print(f'acc_test:{acc_test:.5f},pre_test:{pre_test:.5f},rec_test:{rec_test:.5f},f1_test:{f1_test:.5f}')







