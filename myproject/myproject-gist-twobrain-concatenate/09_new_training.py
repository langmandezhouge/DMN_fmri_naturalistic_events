import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
from torch.utils.data import random_split
import numpy as np
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils
import matplotlib.pyplot as plt
from transformers.optimization import get_linear_schedule_with_warmup
import argparse
import pickle
import os
from scipy import stats
import pandas as pd

class subDataset(Dataset.Dataset):
    # 初始化定义数据内容和标签
    def __init__(self, data, label):
        self.data = data
        self.label = label

    # 返回数据集大小
    def __len__(self):
        return len(self.data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        label = torch.Tensor(self.label[index])
        return data, label


class Model(nn.Module):
    def __init__(self, input_size, 
                       hidden1_size, hidden2_size, hidden3_size, hidden4_size, 
                       output_size, 
                       prob1, prob2, prob3, prob4, prob5):
        super(Model, self).__init__()
        self.drop_1 = nn.Dropout(p=prob1)
        self.drop_2 = nn.Dropout(p=prob2)
        self.drop_3 = nn.Dropout(p=prob3)
        self.drop_4 = nn.Dropout(p=prob4)
        self.drop_5 = nn.Dropout(p=prob5)
        self.layer1 = nn.LayerNorm([input_size])
        self.layer2 = nn.LayerNorm([hidden1_size])
        self.layer3 = nn.LayerNorm([hidden2_size])
        self.layer4 = nn.LayerNorm([hidden3_size])
        self.layer5 = nn.LayerNorm([hidden4_size])
        self.lstm_1 = nn.LSTM(input_size, hidden1_size, 1, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden1_size, hidden2_size, 1, batch_first=True)
        self.lstm_3 = nn.LSTM(hidden2_size, hidden3_size, 1, batch_first=True)
        self.lstm_4 = nn.LSTM(hidden3_size, hidden4_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden4_size, output_size)
        """
        bias = ["bias_hh_l0", "bias_ih_l0"]
        for param in self.lstm_1.named_parameters():
            if len(param[1].shape) > 1:
                print(2)
                nn.init.orthogonal_(param[1])
            if param[0] in bias:
                print(1)
                with torch.no_grad():
                    param[1][hidden1_size*1:hidden1_size*2] = 5
        for param in self.lstm_2.named_parameters():
            if len(param[1].shape) > 1:
                print(2)
                nn.init.orthogonal_(param[1])
            if param[0] in bias:
                print(1)
                with torch.no_grad():
                    param[1][hidden2_size*1:hidden2_size*2] = 5
        """
    
    def forward(self, batch):
        padseq, length = pad_packed_sequence(batch, batch_first=True)
        padseq = self.drop_1(padseq)
        batch = pack_padded_sequence(padseq, length, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm_1(batch)

        padseq, length = pad_packed_sequence(output, batch_first=True)
        padseq = self.drop_2(padseq)
        output = pack_padded_sequence(padseq, length, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm_2(output)

        padseq, length = pad_packed_sequence(output, batch_first=True)
        padseq = self.drop_3(padseq)
        output = pack_padded_sequence(padseq, length, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm_3(output)

        padseq, length = pad_packed_sequence(output, batch_first=True)
        padseq = self.drop_4(padseq)
        output = pack_padded_sequence(padseq, length, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm_4(output)

        hn = self.drop_5(hn)
        return self.linear(hn[0])


def collate_fn(dataset):
    data_batch = []
    label_batch = []
    for data,label in dataset:
        data_batch.append(data)
        label_batch.append(label)
    seq_len = [s.size(0) for s in data_batch]
    data_pad = pad_sequence(data_batch, batch_first=True)
    label_batch = torch.stack(label_batch)
    data_batch = pack_padded_sequence(data_pad, seq_len, batch_first=True, enforce_sorted=False)
    return {'data': data_batch,'label': label_batch}


# 下面部分修改default的值就可以
parser = argparse.ArgumentParser()
parser.add_argument('--h1', type=int, default=54) # LSTM第一层隐藏层维度
parser.add_argument('--h2', type=int, default=54) # LSTM第二层隐藏层维度
parser.add_argument('--h3', type=int, default=54) # LSTM第三层隐藏层维度
parser.add_argument('--h4', type=int, default=54) # LSTM第四层隐藏层维度
parser.add_argument('--p1', type=float, default=0.0) # 第一层dropout(用于抑制过拟合)的dropout rate(0~1,越高效果越明显)
parser.add_argument('--p2', type=float, default=0.0) # 第二层dropout(用于抑制过拟合)的dropout rate(0~1,越高效果越明显)
parser.add_argument('--p3', type=float, default=0.0) # 第三层dropout(用于抑制过拟合)的dropout rate(0~1,越高效果越明显)
parser.add_argument('--p4', type=float, default=0.0) # 第四层dropout(用于抑制过拟合)的dropout rate(0~1,越高效果越明显)
parser.add_argument('--p5', type=float, default=0.0) # 第五层dropout(用于抑制过拟合)的dropout rate(0~1,越高效果越明显)
parser.add_argument("--filename", type=str, default="result.pkl")
args = parser.parse_args()


path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/10_labels/region-022_labels.npy'
data_file = np.load(path,allow_pickle=True)
datas = np.transpose(data_file)
data  = datas[0]
label = datas[1]

#构建数据集
batch_size = 16 # batch
learning_rate = 1e-3 # 学习率大小
input_size = 60 # 输入维度
hidden1_size = args.h1
hidden2_size = args.h2
hidden3_size = args.h3
hidden4_size = args.h4
output_size = 54 # 输出维度
prob1 = args.p1
prob2 = args.p2
prob3 = args.p3
prob4 = args.p4
prob5 = args.p5
max_epochs = 100 # epoch
test_interval = 5
warmup_rate = 0.2
weight_decay = 0.1 # weight decay
torch.manual_seed(1029)
dataset = subDataset(data, label)
train_size = int(len(dataset) * 0.8)
eval_size = int(len(dataset) * 0.1)
test_size = len(dataset) - eval_size - train_size
train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size, test_size])
train_loader = DataLoader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn)
validate_loader = DataLoader.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn)
test_loader = DataLoader.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn)

# 模型和优化器定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm = Model(input_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, output_size, prob1, prob2, prob3, prob4, prob5).to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=weight_decay)
trains_step = max_epochs * len(train_loader)
warmup_step = round(trains_step * warmup_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, trains_step)
loss_function = nn.CosineSimilarity(dim=1, eps=1e-6)
lstm.train()

# 模型训练
loss_list = []
eval_list = []
test_list = []
loss_index = []
eval_index = []
test_index = []
for epoch in range(max_epochs):
    for i, item in enumerate(train_loader):
        mids = lstm(item["data"].to(device))
        loss = 1 - torch.mean(loss_function(mids, item["label"].to(device)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_list.append(loss.cpu().detach().item())
        loss_index.append(len(train_loader)*epoch+i+1)
        print('Epoch: [{}/{}], Step: [{}/{}], Loss:{}'.format(epoch+1, max_epochs, i+1, len(train_loader), loss_list[-1]))
        if i%test_interval == 0:
            lstm.eval()
            eval_loss = []
            for j, jtem in enumerate(validate_loader):
                mids = lstm(jtem["data"].to(device))
                loss = loss_function(mids, jtem["label"].to(device))
                eval_loss.append(loss)
            test_loss = []
            for j, jtem in enumerate(test_loader):
                mids = lstm(jtem["data"].to(device))
                loss = loss_function(mids, jtem["label"].to(device))
                test_loss.append(loss)
            eval_loss = 1 - torch.mean(torch.cat(eval_loss, 0)).cpu().detach().item()
            test_loss = 1 - torch.mean(torch.cat(test_loss, 0)).cpu().detach().item()
            eval_list.append(eval_loss)
            test_list.append(test_loss)
            eval_index.append(loss_index[-1])
            test_index.append(loss_index[-1])
            print('----Epoch: [{}/{}], Step: [{}/{}], evalLoss:{}----'.format(epoch+1, max_epochs, i+1, len(train_loader), eval_loss))
            print('----Epoch: [{}/{}], Step: [{}/{}], testLoss:{}----'.format(epoch+1, max_epochs, i+1, len(train_loader), test_loss))
            lstm.train()

    similarity = []
    if epoch+1 == max_epochs:
        for j, jtem in enumerate(test_loader):
            mids = lstm(jtem["data"].to(device))
            loss = loss_function(mids, jtem["label"].to(device))
            similarity.append(loss)
        test_similarity = torch.cat(similarity, 0).cpu().detach().numpy()
        test_similarity_mean = torch.mean(torch.cat(similarity, 0)).cpu().detach().numpy()
        r,p_value = stats.ttest_1samp(test_similarity, 0, axis=0)

        out = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/lstm-result/region-022/'
        if not os.path.exists(out):
           os.makedirs(out)
        np.save(os.path.join(out, "test_similarity"), test_similarity)
        np.save(os.path.join(out, "test_similarity_mean"), test_similarity_mean)
        np.save(os.path.join(out, "p"), p_value)


with open(args.filename, "wb") as file:
    pickle.dump([loss_index, loss_list, eval_index, eval_list, test_index, test_list], file)
print("final:[best_eval:{} best_test:{}]".format(min(eval_list), test_list[eval_list.index(min(eval_list))]))

plt.xlabel("step")
plt.ylabel("loss")
plt.plot(loss_index, loss_list, "r")
plt.plot(eval_index, eval_list, "g")
plt.plot(test_index, test_list, "b")
plt.title('region-022')
plt.savefig('/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/lstm-result/region-022/region-022.png')
plt.show()
