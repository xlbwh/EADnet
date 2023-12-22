import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        self.fc1 = nn.Linear(60 * num_channels, 9)  # 12个通道，每个通道60个特征

    def forward(self, x):
        # 分支处理，每个分支处理一个通道
        branch_outputs = [branch(x[:, i:i + 1, :]) for i, branch in enumerate(self.branches)]

        # 拼接所有分支的输出
        x = torch.cat(branch_outputs, dim=1)

        # 展平处理
        x = x.view(x.size(0), -1)
        # 保存卷积层输出
        self.conv_output = x
        # 全连接层
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

    def get_CAM(self, target_class):
        # 获取全连接层权重
        weight_fc = self.fc1.weight.data

        # 获取目标类别的权重
        class_weights = weight_fc[target_class]

        # 生成CAM
        cam = class_weights * self.conv_output

        cam = cam.cpu().detach().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


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


# 加载模型
model = LightSeizureNet(num_channels=12)
model.load_state_dict(torch.load('modelfor9class.pt', map_location = device))
model.to(device)
model.eval()
lead_files = [f'9classPreprocess/lead{i}.txt' for i in range(1, 13)]
label_file = '9classPreprocess/labels.txt'
data, labels = read_ecg_data(lead_files, label_file)
print(data.shape)
print(labels.shape)
data = standardize_data_per_lead(data)
data = torch.from_numpy(data).float().to(device)
labels = torch.from_numpy(labels).long().to(device)
data = data.transpose(1, 2)
print(data.shape)

cams = np.zeros((9, 12))
for i in range(data.shape[0]):
    print(f'Processing {i + 1} sample', end='\r')
    sample = data[i].unsqueeze(0)
    print(sample.shape)
    output = model(sample)
    cam = model.get_CAM(labels[i])
    result = np.sum(cam.reshape((12, 60)), axis=1)
    result = np.divide(result - np.min(result), np.max(result) - np.min(result))
    cams[labels[i]] += result
    print(f'Processed {i + 1} samples', end='\r')

for i in range(9):
    cams[i] = np.divide(cams[i] - np.min(cams[i]), np.max(cams[i]) - np.min(cams[i]))

# 绘制CAM
print(cams.shape)
#cams = np.divide(cams - np.min(cams), np.max(cams) - np.min(cams))
# 创建热力图
plt.imshow(cams, cmap='viridis', aspect='auto')
plt.xticks(np.arange(12),['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'])
plt.yticks(np.arange(9),['N','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE'])
# 添加颜色条
plt.colorbar()
plt.savefig('heatmap_average——3.png')

