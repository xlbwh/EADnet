import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


# 文件名列表
lead_files = [f'9classPreprocess/lead{i + 1}.txt' for i in range(12)]
label_file = '9classPreprocess/labels.txt'

# 读取数据
data, labels = read_ecg_data(lead_files, label_file)


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


# 应用数据标准化
data = standardize_data_per_lead(data)

# 将数据划分为训练集和测试集
data_train, data_test, labels_train, labels_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)  # 10%的数据用于测试集

# 将数据转换为PyTorch张量并重新排列维度
data_train_tensor = torch.from_numpy(data_train).float().transpose(1, 2)
labels_train_tensor = torch.from_numpy(labels_train).long()
data_test_tensor = torch.from_numpy(data_test).float().transpose(1, 2)
labels_test_tensor = torch.from_numpy(labels_test).long()

# 创建数据集和数据加载器
train_dataset = TensorDataset(data_train_tensor, labels_train_tensor)
test_dataset = TensorDataset(data_test_tensor, labels_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义模型
class LightSeizureNet(nn.Module):
    def __init__(self, num_channels=21):
        super(LightSeizureNet, self).__init__()

        # 根据模型架构图和参数表定义分支卷积层
        self.branches = nn.ModuleList()
        dilation_rates = [1, 2, 4, 8, 16, 32]  # 定义扩张率
        for i in range(num_channels):
            self.branches.append(nn.Sequential(
                nn.Conv1d(1, 12, kernel_size=2, stride=1),
                nn.ReLU(),
                nn.Conv1d(12, 12, kernel_size=1, stride=1, groups=12),
                nn.ReLU(),
                nn.Conv1d(12, 24, kernel_size=24, stride=2),
                nn.MaxPool1d(kernel_size=3, stride=1),
                nn.Conv1d(24, 24, kernel_size=1, stride=1, groups=24),
                nn.ReLU(),
                nn.Conv1d(24, 32, kernel_size=32, stride=1),
                nn.MaxPool1d(kernel_size=3, stride=1),
                nn.Conv1d(32, 32, kernel_size=1, stride=1, groups=32),
                nn.ReLU(),
                nn.Conv1d(32, 48, kernel_size=32, stride=1),
                nn.MaxPool1d(kernel_size=3, stride=1),
                nn.Conv1d(48, 48, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv1d(48, 60, kernel_size=48, stride=1),
                nn.MaxPool1d(kernel_size=3, stride=1),
                nn.AdaptiveAvgPool1d(1)
            ))

        # 定义全连接层
        self.fc1 = nn.Linear(60 * num_channels, 9)  # 21个通道，每个通道48个特征

    def forward(self, x):
        # 分支处理，每个分支处理一个通道
        branch_outputs = [branch(x[:, i:i + 1, :]) for i, branch in enumerate(self.branches)]

        # 拼接所有分支的输出
        x = torch.cat(branch_outputs, dim=1)

        # 展平处理
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


# 实例化模型
model = LightSeizureNet(num_channels=12).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0005)

# 训练模型
num_epochs = 60  # 训练轮数
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # 将数据和标签移到 GPU
        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度归零
        optimizer.zero_grad()

        # 正向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 100 == 99:  # 每100个批次打印一次
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('Finished Training')

torch.save(model.state_dict(), 'modelfor9class.pt')
print('Model Saved')

# 评估模型性能
with torch.no_grad():
    model.eval()
    all_outputs = []
    all_labels = []
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.cpu()
        all_outputs.append(outputs)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    # 保存指标到文件
    save_metrics(all_outputs, all_labels, 'resultsfor9clss_2.txt')

print('Finished Training and Saved Metrics')
