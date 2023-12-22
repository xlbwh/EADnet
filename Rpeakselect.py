import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import scipy.io

def print_mat_structure(mat_file):
    # 加载.mat文件
    mat_data = scipy.io.loadmat(mat_file)

    # 打印文件结构
    for var_name in mat_data:
        # 忽略MATLAB的内置变量
        if var_name[0] != '_':
            print(f"Variable Name: {var_name}")
            print(f"Type: {type(mat_data[var_name])}")
            # 你可以根据需要添加更多信息，例如数组的形状、数据类型等
            if isinstance(mat_data[var_name], np.ndarray):
                print(f"Array Shape: {mat_data[var_name].shape}")
                print(f"Data Type: {mat_data[var_name].dtype}")
            print("-" * 40)


# Pan-Tompkins算法实现
def pan_tompkins_detector(ecg_signal, sampling_rate):
    # 低通滤波器
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    # 高通滤波器
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    # 信号滤波
    order = 1
    fs = sampling_rate       # 采样率
    lowcut = 0.5
    highcut = 45
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = filtfilt(b, a, ecg_signal)
    b, a = butter_highpass(highcut, fs, order=order)
    y = filtfilt(b, a, y)

    # 微分
    diff = np.diff(y)

    # 平方
    squared = diff ** 2

    # 移动窗口积分
    N = int(0.12 * sampling_rate)
    integrate = np.convolve(squared, np.ones(N)/N, mode='same')

    # 寻找R波峰值
    peak_indices = find_peaks(integrate, distance=0.65 * sampling_rate)[0]

    return peak_indices


# 加载标签文件
df = pd.read_csv('datafile/REFERENCE.csv')

# 筛选正常样本
normal_df = df[(df['First_label'] == 1)]
normal_list = normal_df['Recording'].tolist()
# 筛选AF样本
AF_df = df[(df['First_label'] == 2) & (df['Second_label'].isna())]
AF_list = AF_df['Recording'].tolist()
# 筛选I-AVB样本
I_AVB_df = df[(df['First_label'] == 3) & (df['Second_label'].isna())]
I_AVB_list = I_AVB_df['Recording'].tolist()
# 筛选LBBB样本
LBBB_df = df[(df['First_label'] == 4) & (df['Second_label'].isna())]
LBBB_list = LBBB_df['Recording'].tolist()
# 筛选PAC样本
PAC_df = df[(df['First_label'] == 5) & (df['Second_label'].isna())]
PAC_list = PAC_df['Recording'].tolist()
# 筛选PVC样本
PVC_df = df[(df['First_label'] == 6) & (df['Second_label'].isna())]
PVC_list = PVC_df['Recording'].tolist()
# 筛选RBBB样本
RBBB_df = df[(df['First_label'] == 7) & (df['Second_label'].isna())]
RBBB_list = RBBB_df['Recording'].tolist()
# 筛选STD样本
STD_df = df[(df['First_label'] == 8) & (df['Second_label'].isna())]
STD_list = STD_df['Recording'].tolist()
# 筛选STE样本
STE_df = df[(df['First_label'] == 9) & (df['Second_label'].isna())]
STE_list = STE_df['Recording'].tolist()

# 分通道读取数据
list_1 =[]
list_2 =[]
list_3 =[]
list_4 =[]
list_5 =[]
list_6 =[]
list_7 =[]
list_8 =[]
list_9 =[]
list_10 =[]
list_11 =[]
list_12 =[]
labels = []

def readlead(samlist, number):
    for samples in samlist:
        # 加载.mat文件
        mat_file_path = 'datafile/' + samples + '.mat'
        mat_data = scipy.io.loadmat(mat_file_path)
        ecg_data = mat_data['ECG']
        data_field = ecg_data['data'][0, 0]
        ecg_leads = np.array(data_field)
        n_samples = ecg_leads.shape[1]
        # 选择一个导联进行R波检测，例如导联II
        lead_II = ecg_leads[1]  # 假设第2行是导联II
        r_peaks = pan_tompkins_detector(lead_II, 500)
        # 分割心拍
        pre_points = 125  # 在R波之前取的样本点数
        post_points = 200  # 在R波之后取的样本点数

        heartbeats = {lead: [] for lead in range(12)}  # 为每个导联创建一个列表

        for peak in r_peaks:
            start = max(0, peak - pre_points)
            end = min(n_samples, peak + post_points)
            labels.append(number)
            for lead in range(12):
                heartbeat = ecg_leads[lead, start:end]
                heartbeats[lead].append(heartbeat)

        list_1.extend(heartbeats[0])
        list_2.extend(heartbeats[1])
        list_3.extend(heartbeats[2])
        list_4.extend(heartbeats[3])
        list_5.extend(heartbeats[4])
        list_6.extend(heartbeats[5])
        list_7.extend(heartbeats[6])
        list_8.extend(heartbeats[7])
        list_9.extend(heartbeats[8])
        list_10.extend(heartbeats[9])
        list_11.extend(heartbeats[10])
        list_12.extend(heartbeats[11])
        print(samples + '处理完成！')


readlead(normal_list, 0)
readlead(AF_list, 1)
readlead(I_AVB_list, 2)
readlead(LBBB_list, 3)
readlead(PAC_list, 4)
readlead(PVC_list, 5)
readlead(RBBB_list, 6)
readlead(STD_list, 7)
readlead(STE_list, 8)

def save_data(my_list, filename):
    # 打开一个文件用于写入
    with open(filename, 'w') as file:
        # 遍历大列表中的每个子列表
        for sublist in my_list:
            # 将子列表的每个元素转换为字符串并用空格连接
            line = ' '.join(map(str, sublist))
            # 将字符串写入文件，每个子列表占一行
            file.write(line + '\n')
        print(filename+'保存完成！')

save_data(list_1, '9classPreprocess/lead1.txt')
save_data(list_2, '9classPreprocess/lead2.txt')
save_data(list_3, '9classPreprocess/lead3.txt')
save_data(list_4, '9classPreprocess/lead4.txt')
save_data(list_5, '9classPreprocess/lead5.txt')
save_data(list_6, '9classPreprocess/lead6.txt')
save_data(list_7, '9classPreprocess/lead7.txt')
save_data(list_8, '9classPreprocess/lead8.txt')
save_data(list_9, '9classPreprocess/lead9.txt')
save_data(list_10, '9classPreprocess/lead10.txt')
save_data(list_11, '9classPreprocess/lead11.txt')
save_data(list_12, '9classPreprocess/lead12.txt')

with open('9classPreprocess/labels.txt', 'w') as file:
    # 遍历列表中的每个元素
    for item in labels:
        # 将每个元素写入文件，每个元素后跟一个换行符
        file.write(str(item) + '\n')


print('9类数据预处理完成！')
