import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, classification_report
import torch
from torch import nn
from torch.optim import Adam, AdamW, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import time


def save_metrics(outputs, labels, file_name):
    # 计算预测值
    _, predicted = torch.max(outputs.data, 1)

    # 计算混淆矩阵
    cm = confusion_matrix(labels.cpu(), predicted.cpu())

    # 计算评估指标
    accuracy = accuracy_score(labels.cpu(), predicted.cpu())
    recall = recall_score(labels.cpu(), predicted.cpu(), average='macro')
    precision = precision_score(labels.cpu(), predicted.cpu(), average='macro')
    f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro')  # 使用宏平均来计算F1分数

    # 将指标和混淆矩阵保存到文件
    with open(file_name, 'w') as file:
        file.write(f'Accuracy: {accuracy}\n')
        file.write(f'Recall (macro-avg): {recall}\n')
        file.write(f'Precision (macro-avg): {precision}\n')
        file.write(f'F1 Score (macro-avg): {f1}\n')  # 添加宏平均F1分数
        file.write(f'Confusion Matrix: \n{cm}')

        # 可以选择添加分类报告，它提供了每个类的性能细节
        class_report = classification_report(labels.cpu(), predicted.cpu())
        file.write(f'\nClassification Report: \n{class_report}')
    return accuracy


# 读取ECG数据和标签
def read_ecg_data(lead_files, label_file, min_length=325):
    # 初始化一个空的list来收集各个导联的数据
    lead_data_list = []

    # 首先处理第一个导联文件，记录长度正确的行的索引
    with open(lead_files[0], 'r') as file:
        valid_indices = [index for index, line in enumerate(file) if len(line.split()) == min_length]

    # 读取第一个导联文件的数据
    with open(lead_files[0], 'r') as file:
        lead_data = np.array(
            [np.fromstring(line, sep=' ') for index, line in enumerate(file) if index in valid_indices])
        print(lead_data.shape)
        lead_data_list.append(lead_data)

    # 处理剩余的导联文件，只保留有效索引处的数据
    for lead_file in lead_files[1:]:
        with open(lead_file, 'r') as file:
            lead_data = np.array(
                [np.fromstring(line, sep=' ') for index, line in enumerate(file) if index in valid_indices])
            print(lead_data.shape)
            lead_data_list.append(lead_data)

    # 将各个导联的数据堆叠起来，形成一个新的维度
    data = np.stack(lead_data_list, axis=-1)

    # 读取标签，只保留有效索引处的标签
    all_labels = np.loadtxt(label_file)
    labels = all_labels[valid_indices]
    print(labels.shape)

    return data, labels


# 数据标准化
def standardize_data_per_lead(data):
    # data的形状应为(num_samples, num_leads, sequence_length)
    # 初始化一个相同形状的数组来存储标准化后的数据
    standardized_data = np.zeros_like(data)

    # 对每个导联单独进行标准化
    for i in range(data.shape[1]):  # 遍历所有导联
        lead_data = data[:, i, :]  # 获取单个导联的数据
        mean_val = np.mean(lead_data)
        std_val = np.std(lead_data)
        # 对该导联的数据进行标准化
        standardized_data[:, i, :] = (lead_data - mean_val) / (std_val + 1e-10)

    return standardized_data


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_ds():
    leadlist = [6, 7, 8, 9, 12]
    lead_files = [f'../lead{i}.txt' for i in leadlist]
    label_file = '../labels.txt'
    # 读取数据
    data, labels = read_ecg_data(lead_files, label_file)
    print(f'data.shape = {data.shape}, labels.shape = {labels.shape}')

    # 应用数据标准化
    data = standardize_data_per_lead(data)
    print(f'data.shape = {data.shape}, labels.shape = {labels.shape}')

    train = data.transpose(0, 2, 1)
    print(f'train.shape = {train.shape}, labels.shape = {labels.shape}')

    input_size = train.shape[2]
    sequence_length = train.shape[1]

    print(f'input_size = {input_size}, sequence_length = {sequence_length}')

    data_train, data_valid, labels_train, labels_valid = train_test_split(train, labels, test_size=0.2, random_state=42)
    print(f'data_train.shape = {data_train.shape}, data_valid.shape = {data_valid.shape}')
    print(f'labels_train.shape = {labels_train.shape}, labels_valid.shape = {labels_valid.shape}')

    print(f'labels_train.shape = {labels_train.shape}, labels_valid.shape = {labels_valid.shape}')

    train_X_tensor = torch.tensor(data_train).float()
    valid_X_tensor = torch.tensor(data_valid).float()

    train_y_tensor = torch.from_numpy(labels_train).long()
    valid_y_tensor = torch.from_numpy(labels_valid).long()

    print(f'train_X_tensor.shape = {train_X_tensor.shape}, train_y_tensor.shape = {train_y_tensor.shape}')

    train_tensor = TensorDataset(train_X_tensor, train_y_tensor)
    valid_tensor = TensorDataset(valid_X_tensor, valid_y_tensor)

    # Defining the dataloaders
    dataloaders = dict()
    dataloaders['train'] = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    dataloaders['valid'] = DataLoader(valid_tensor, batch_size=batch_size, shuffle=False)
    return dataloaders, input_size, sequence_length


def train_epochs(dataloaders, epochs, optimizer, scheduler, loss_func):
    scaler = GradScaler()

    for epoch in range(epochs):
        train_loss = AverageMeter()
        model.train()
        for batch_x, labels in tqdm(dataloaders['train']):
            batch_x, labels = batch_x.to(device), labels.to(device)
            labels = labels
            with autocast():
                output = model.forward(batch_x)
                # print(f'output.shape = {output.shape}, labels.shape = {labels.shape}')
                loss = loss_func(output, labels)
            scaler.scale(loss).backward()
            train_loss.update(loss.item(), output.size(0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()


if __name__ == "__main__":
    import models

    # 设置随机数
    seed = 724
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置网络参数
    batch_size = 256
    lr = 0.001
    epochs = 30
    hidden_sizes = [288, 192, 144, 96, 32]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # 获取数据集dataloaders
    dataloaders, input_size, sequence_length = get_ds()

    model = models.ResBiTimeNet(input_size, hidden_sizes, sequence_length)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(dataloaders['train']),
                                        pct_start=0.2, anneal_strategy='cos')

    loss_func = nn.CrossEntropyLoss()
    start_time = time.time()
    torch.cuda.reset_max_memory_allocated()
    train_epochs(dataloaders=dataloaders, epochs=30, optimizer=optimizer, scheduler=scheduler, loss_func=loss_func)
    end_time = time.time()
    print(f'time: {end_time - start_time}')
    with open('../5leads_results/times.txt', 'a') as file:
        file.write(f'time: {end_time - start_time}\n')

    max_memory = torch.cuda.max_memory_allocated()
    with open('../5leads_results/max_memory.txt', 'a') as file:
        file.write(f'max memory: {max_memory}\n')
    # 计算验证集上的指标
    all_outputs = []
    all_labels = []
    valid_loss = AverageMeter()
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    with open('../5leads_results/numberofpara.txt', 'a') as file:
        file.write(f'Total number of parameters: {total_params}\n')
    preds_all = torch.LongTensor()
    labels_all = torch.LongTensor()
    for batch_x, labels in tqdm(dataloaders['valid']):
        labels_all = torch.cat((labels_all, labels), dim=0)
        batch_x, labels = batch_x.to(device), labels.to(device)
        labels = labels
        with torch.no_grad():
            output = model.forward(batch_x)
            loss = loss_func(output, labels)
        preds_all = torch.cat((preds_all, torch.sigmoid(output).to('cpu')), dim=0)
        valid_loss.update(loss.item(), output.size(0))
        all_outputs.append(output)
        all_labels.append(labels)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    acc = save_metrics(all_outputs, all_labels, f'../5leads_results/results.txt')

