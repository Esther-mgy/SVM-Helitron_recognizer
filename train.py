import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib  # 用于保存模型
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置参数（基于论文设置）=====================
# FCGS2参数：2-mer核苷酸组合
DINUCLEOTIDES = ['AA', 'AT', 'AC', 'AG',
                 'TA', 'TT', 'TC', 'TG',
                 'CA', 'CT', 'CC', 'CG',
                 'GA', 'GT', 'GC', 'GG']
# CWT参数（论文设置：64个尺度，omega0=5.4285）
CWT_SCALES = 64
OMEGA0 = 5.4285
# SVM参数（论文最优：RBF核，c=60，sigma=0.0000000015625）
SVM_C = 60
SVM_SIGMA = 0.0000000015625
RBF_GAMMA = 1 / (2 * SVM_SIGMA ** 2)  # SVM库中gamma与sigma的转换关系
# 数据路径
DATA_DIR = "/public/home/h2024319020/mgy/data/clear_data/plant/split_LINE_model/"
# 模型保存路径（用户指定）
MODEL_SAVE_PATH = "/public/home/h2024319020/mgy/SVM-Helitron recognizer/models2/LINE/"
# 输出结果保存路径
OUTPUT_DIR = "/public/home/h2024319020/mgy/SVM-Helitron recognizer/results/"

# 创建必要的文件夹
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 工具函数 =====================
def load_and_split_data(data_dir):
    """
    加载数据并按要求划分：每个txt文件前90%为训练集，后10%为测试集（按顺序）
    返回：train_sequences, train_labels, test_sequences, test_labels
    """
    train_seqs = []
    train_labs = []
    test_seqs = []
    test_labs = []
    
    # 遍历所有txt文件
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            print(f"正在处理文件：{filename}")
            
            # 读取文件（每行：标签,序列）
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # 按顺序划分：前90%训练，后10%测试（向下取整）
            split_idx = int(len(lines) * 0.9)
            train_lines = lines[:split_idx]
            test_lines = lines[split_idx:]
            
            # 解析训练集
            for line in train_lines:
                if ',' not in line:
                    continue  # 跳过格式错误的行
                label, seq = line.split(',', 1)
                seq = seq.upper()  # 转为大写
                if all(c in ['A', 'T', 'C', 'G'] for c in seq):  # 过滤非法字符
                    train_seqs.append(seq)
                    train_labs.append(label)
            
            # 解析测试集
            for line in test_lines:
                if ',' not in line:
                    continue
                label, seq = line.split(',', 1)
                seq = seq.upper()
                if all(c in ['A', 'T', 'C', 'G'] for c in seq):
                    test_seqs.append(seq)
                    test_labs.append(label)
    
    print(f"\n数据加载完成：")
    print(f"训练集：{len(train_seqs)}条序列")
    print(f"测试集：{len(test_seqs)}条序列")
    print(f"类别数量：{len(set(train_labs))}")
    print(f"类别列表：{sorted(set(train_labs))}")
    return train_seqs, train_labs, test_seqs, test_labs

def fcgs2_signal(sequence):
    """
    论文FCGS2编码：生成时序信号（严格按论文公式实现）
    输入：DNA序列（字符串）
    输出：FCGS2时序信号（长度=len(sequence)-1，每个位置是对应2-mer的概率）
    论文公式：
    - P_2nuc = N_2nuc / N_ch（N_ch为序列总长度bp）
    - S_2nuc(k) = P_2nuc(i,k)（位置k的2-mer概率）
    - FCGS2 = sum(S_2nuc)（信号为各位置概率值）
    """
    seq_len = len(sequence)
    if seq_len < 2:
        return np.array([])
    
    # 统计所有2-mer的全局出现次数（论文：N_2nuc）
    dinuc_count = {dinuc: 0 for dinuc in DINUCLEOTIDES}
    for i in range(seq_len - 1):
        dinuc = sequence[i:i+2]
        if dinuc in dinuc_count:
            dinuc_count[dinuc] += 1
    
    # 生成时序信号：每个位置k替换为对应2-mer的概率P_2nuc
    # 论文：S_2nuc(k) = sum(P_2nuc(i,k))，即位置k的2-mer概率
    signal = []
    for i in range(seq_len - 1):
        dinuc = sequence[i:i+2]
        # 论文公式1：P_2nuc = N_2nuc / N_ch
        prob = dinuc_count.get(dinuc, 0) / seq_len
        signal.append(prob)
    
    return np.array(signal)

def manual_cwt(signal_data, scales):
    """
    手动实现连续小波变换（CWT）使用Complex Morlet小波
    严格按论文公式实现：
    - 小波函数：psi_cmor(t) = pi^(-1/4) * (exp(i*w0*t) - exp(-w0^2/2)) * exp(-t^2/2)
    - CWT公式：W(s,u) = (1/sqrt(s)) * integral(x(t) * psi*((t-u)/s) dt)
    输入：
        signal_data: 输入信号（一维数组）
        scales: 尺度数组（论文：64个尺度）
    输出：
        cwt_coeffs: 小波系数矩阵（shape: (len(scales), len(signal_data))）
    """
    n = len(signal_data)
    cwt_coeffs = np.zeros((len(scales), n), dtype=np.complex128)
    
    # 时间轴（与信号长度匹配）
    t_full = np.linspace(-10, 10, n)
    
    for i, scale in enumerate(scales):
        # 论文Complex Morlet小波公式
        # psi(t) = pi^(-1/4) * (exp(i*w0*t) - exp(-w0^2/2)) * exp(-t^2/2)
        t_scaled = t_full / scale
        wavelet_base = (np.exp(1j * OMEGA0 * t_scaled) - np.exp(-0.5 * OMEGA0**2)) * np.exp(-0.5 * t_scaled**2)
        wavelet = (np.pi ** (-0.25)) * wavelet_base / np.sqrt(scale)
        
        # CWT公式：卷积实现（考虑复共轭，对实信号等价于卷积）
        # W(s,u) = (1/sqrt(s)) * x(t) * psi*((t-u)/s) 的积分
        conv_result = np.convolve(signal_data, wavelet[::-1], mode='same')
        cwt_coeffs[i, :] = conv_result
    
    return cwt_coeffs

def cwt_feature_extraction(fcgs_signal):
    """
    论文CWT特征提取：对FCGS2时序信号进行CWT并计算能量特征
    输入：FCGS2时序信号（一维数组，长度可变）
    输出：小波能量特征向量（长度64，对应64个尺度）
    论文公式：E(s) = sum(|W(s,u)|^2)
    """
    # 处理变长信号：统一长度以便CWT计算（论文中helitron长度可变）
    # 策略：固定目标长度，长则采样，短则补零
    target_length = 2048  # 可根据实际数据调整，需足够长以体现时频特征
    
    if len(fcgs_signal) > target_length:
        # 等间隔下采样
        indices = np.linspace(0, len(fcgs_signal)-1, target_length, dtype=int)
        processed_signal = fcgs_signal[indices]
    elif len(fcgs_signal) < target_length:
        # 补零（zero-padding）
        processed_signal = np.pad(fcgs_signal, (0, target_length - len(fcgs_signal)), mode='constant')
    else:
        processed_signal = fcgs_signal
    
    # 应用CWT（64个尺度，论文设置）
    scales = np.arange(1, CWT_SCALES + 1)
    cwt_coeffs = manual_cwt(processed_signal, scales)
    
    # 计算每个尺度的能量（论文公式：E(s) = sum|W(s,u)|^2）
    energy_features = np.sum(np.abs(cwt_coeffs) ** 2, axis=1)
    
    return energy_features

def extract_features(sequences):
    """
    批量提取特征：FCGS2时序信号 + CWT能量特征
    输入：序列列表
    输出：特征矩阵（n_samples × 64）
    """
    features = []
    for seq in sequences:
        fcgs_sig = fcgs2_signal(seq)
        if len(fcgs_sig) == 0:
            # 处理过短序列：返回零向量
            features.append(np.zeros(CWT_SCALES))
        else:
            cwt_energy = cwt_feature_extraction(fcgs_sig)
            features.append(cwt_energy)
    return np.array(features)

# ===================== 主训练测试流程 =====================
def main():
    # 1. 加载并划分数据
    train_seqs, train_labs, test_seqs, test_labs = load_and_split_data(DATA_DIR)
    
    # 2. 标签编码（将字符串标签转为数字）
    label_encoder = LabelEncoder()
    train_labs_encoded = label_encoder.fit_transform(train_labs)
    test_labs_encoded = label_encoder.transform(test_labs)
    
    # 3. 特征提取（训练集+测试集）
    print("\n开始提取训练集特征...")
    train_features = extract_features(train_seqs)
    print("开始提取测试集特征...")
    test_features = extract_features(test_seqs)
    
    print(f"\n特征提取完成：")
    print(f"训练集特征形状：{train_features.shape}")
    print(f"测试集特征形状：{test_features.shape}")
    
    # 4. 初始化SVM分类器（论文最优配置：RBF核 + OAO多分类策略）
    # 注：sklearn SVM的OAO策略通过decision_function_shape='ovo'实现
    svm_classifier = SVC(
        kernel='rbf',
        C=SVM_C,
        gamma=RBF_GAMMA,
        decision_function_shape='ovo',  # 对应论文OAO多分类
        random_state=42,
        probability=True
    )
    
    # 5. 训练模型
    print("\n开始训练SVM模型...")
    svm_classifier.fit(train_features, train_labs_encoded)
    print("模型训练完成！")
    
    # 6. 保存模型和标签编码器（关键新增功能）
    model_filename = os.path.join(MODEL_SAVE_PATH, "svm_helitron_classifier.pkl")
    encoder_filename = os.path.join(MODEL_SAVE_PATH, "label_encoder.pkl")
    
    joblib.dump(svm_classifier, model_filename)
    joblib.dump(label_encoder, encoder_filename)
    print(f"模型已保存到：{model_filename}")
    print(f"标签编码器已保存到：{encoder_filename}")
    
    # 7. 测试模型
    print("\n开始测试模型...")
    train_pred = svm_classifier.predict(train_features)
    test_pred = svm_classifier.predict(test_features)
    
    # 8. 计算评估指标
    train_acc = accuracy_score(train_labs_encoded, train_pred)
    test_acc = accuracy_score(test_labs_encoded, test_pred)
    
    print(f"\n========== 模型性能评估 ==========")
    print(f"训练集准确率：{train_acc:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")
    print(f"\n测试集分类报告：")
    print(classification_report(
        test_labs_encoded, 
        test_pred, 
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    
    # 9. 生成混淆矩阵
    conf_matrix = confusion_matrix(test_labs_encoded, test_pred)
    conf_df = pd.DataFrame(
        conf_matrix,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )
    
    # 10. 保存结果
    # 保存混淆矩阵
    conf_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), encoding='utf-8')
    # 保存分类报告
    report = classification_report(
        test_labs_encoded, 
        test_pred, 
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True
    )
    pd.DataFrame(report).to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"), encoding='utf-8')
    # 保存模型参数和性能
    with open(os.path.join(OUTPUT_DIR, "model_performance.txt"), 'w', encoding='utf-8') as f:
        f.write(f"模型参数：\n")
        f.write(f" - SVM核函数：RBF\n")
        f.write(f" - C值：{SVM_C}\n")
        f.write(f" - sigma值：{SVM_SIGMA}\n")
        f.write(f" - gamma值：{RBF_GAMMA}\n")
        f.write(f" - 多分类策略：OAO（One-Against-One）\n")
        f.write(f" - 模型保存路径：{MODEL_SAVE_PATH}\n")
        f.write(f"\n性能指标：\n")
        f.write(f" - 训练集准确率：{train_acc:.4f}\n")
        f.write(f" - 测试集准确率：{test_acc:.4f}\n")
        f.write(f" - 训练集样本数：{len(train_seqs)}\n")
        f.write(f" - 测试集样本数：{len(test_seqs)}\n")
        f.write(f" - 类别数：{len(label_encoder.classes_)}\n")
        f.write(f" - 类别列表：{sorted(label_encoder.classes_)}\n")
    
    print(f"\n所有结果已保存到：{OUTPUT_DIR}")
    print("训练测试完成！")

# ===================== 模型加载示例（可选使用）=====================
def load_trained_model(model_dir):
    """
    加载训练好的模型和标签编码器
    输入：模型保存目录
    输出：svm_model, label_encoder
    """
    model_path = os.path.join(model_dir, "svm_helitron_classifier.pkl")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError("模型文件或标签编码器文件不存在")
    
    svm_model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print(f"模型加载成功！")
    return svm_model, label_encoder

if __name__ == "__main__":
    main()