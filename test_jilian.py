# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===================== 核心配置 =====================
# 1. 数据路径
ALIGNED_DATA_PATH = "/public/home/h2024319020/mgy/data/plant/aligned_data.txt"
TRAIN_TEST_PATH = "/public/home/h2024319020/mgy/data/plant/train_test.txt"

# 2. 模型路径（7个独立模型）
MODEL_PATHS = {
    "class": "/public/home/h2024319020/mgy/SVM-Helitron recognizer/models2/class/",
    "classI": "/public/home/h2024319020/mgy/SVM-Helitron recognizer/models2/classI/",
    "classII_sub1": "/public/home/h2024319020/mgy/SVM-Helitron recognizer/models2/classII_sub1/",
    "LTR": "/public/home/h2024319020/mgy/SVM-Helitron recognizer/models2/LTR/",
    "nLTR": "/public/home/h2024319020/mgy/SVM-Helitron recognizer/models2/nLTR/",
    "SINE": "/public/home/h2024319020/mgy/SVM-Helitron recognizer/models2/SINE/",
    "LINE": "/public/home/h2024319020/mgy/SVM-Helitron recognizer/models2/LINE/"
}

# 3. 级联关系（模型→下一层模型映射）
CASCADE_MAP = {
    "class": {
        "class I": "classI",
        "class II_sub1": "classII_sub1",
        "class II_sub2": None  # 无对应下一层模型
    },
    "classI": {
        "LTR": "LTR",
        "nLTR": "nLTR"
    },
    "LTR": {
        "Copia": None,
        "Gypsy": None
        
    },
    "nLTR": {
        "DIRS": None,
        "LINE": "LINE",
        "PLE": None,
        "SINE": "SINE"
        
    },
    "SINE": {
        "SINE1/7SL": None,
        "SINE2/tRNA": None
        
    },
    "LINE": {
        "I": None,
        "L1": None
        
    },
    "classII_sub1": {
        "Academ": None,
        "EnSpm/CACTA": None,
        "Harbinger": None,
        "ISL2EU": None,
        "Mariner/Tc1": None,
        "MuDR": None,
        "P": None,
        "Sola": None,
        "hAT": None,
    }
}

# 4. 置信度阈值（可调整）
CONFIDENCE_THRESHOLD = 0.9

# 5. 特征提取相关配置（与train.py完全一致）
DINUCLEOTIDES = ['AA', 'AT', 'AC', 'AG',
                 'TA', 'TT', 'TC', 'TG',
                 'CA', 'CT', 'CC', 'CG',
                 'GA', 'GT', 'GC', 'GG']
CWT_SCALES = 64
OMEGA0 = 5.4285
TARGET_LENGTH = 2048  # 与训练脚本一致的信号统一长度

# ===================== 工具函数（完全复用train.py核心逻辑）=====================
def process_sequence(seq):
    """处理序列：转为大写，非ATCG字符替换为X"""
    seq = seq.upper()
    processed = []
    for c in seq:
        if c in ['A', 'T', 'C', 'G']:
            processed.append(c)
        else:
            processed.append('X')
    return ''.join(processed)

def fcgs2_signal(sequence):
    """
    论文FCGS2编码：生成时序信号（严格按训练脚本实现，处理含X的序列）
    输入：DNA序列（字符串，可能含X）
    输出：FCGS2时序信号（长度=len(sequence)-1，每个位置是对应2-mer的概率）
    """
    seq_len = len(sequence)
    if seq_len < 2:
        return np.array([])
    
    # 统计所有2-mer的全局出现次数（跳过含X的2-mer）
    dinuc_count = {dinuc: 0 for dinuc in DINUCLEOTIDES}
    valid_dinuc_total = 0  # 有效2-mer总数（不含X）
    for i in range(seq_len - 1):
        dinuc = sequence[i:i+2]
        if dinuc in dinuc_count and 'X' not in dinuc:
            dinuc_count[dinuc] += 1
            valid_dinuc_total += 1
    
    # 生成时序信号：每个位置k替换为对应2-mer的概率P_2nuc
    signal = []
    for i in range(seq_len - 1):
        dinuc = sequence[i:i+2]
        # 跳过含X的2-mer，概率记为0
        if 'X' in dinuc or valid_dinuc_total == 0:
            prob = 0.0
        else:
            prob = dinuc_count.get(dinuc, 0) / seq_len  # 与训练脚本公式一致：N_2nuc / N_ch
        signal.append(prob)
    
    return np.array(signal)

def manual_cwt(signal_data, scales):
    """
    手动实现连续小波变换（CWT）使用Complex Morlet小波（与训练脚本完全一致）
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
        # 论文Complex Morlet小波公式（与训练脚本一致）
        t_scaled = t_full / scale
        wavelet_base = (np.exp(1j * OMEGA0 * t_scaled) - np.exp(-0.5 * OMEGA0**2)) * np.exp(-0.5 * t_scaled**2)
        wavelet = (np.pi ** (-0.25)) * wavelet_base / np.sqrt(scale)
        
        # CWT公式：卷积实现（考虑复共轭，对实信号等价于卷积）
        conv_result = np.convolve(signal_data, wavelet[::-1], mode='same')
        cwt_coeffs[i, :] = conv_result
    
    return cwt_coeffs

def cwt_feature_extraction(fcgs_signal):
    """
    论文CWT特征提取（与训练脚本完全一致）
    输入：FCGS2时序信号（一维数组，长度可变）
    输出：小波能量特征向量（长度64，对应64个尺度）
    """
    # 处理变长信号：统一长度以便CWT计算（与训练脚本一致）
    if len(fcgs_signal) > TARGET_LENGTH:
        # 等间隔下采样
        indices = np.linspace(0, len(fcgs_signal)-1, TARGET_LENGTH, dtype=int)
        processed_signal = fcgs_signal[indices]
    elif len(fcgs_signal) < TARGET_LENGTH:
        # 补零（zero-padding）
        processed_signal = np.pad(fcgs_signal, (0, TARGET_LENGTH - len(fcgs_signal)), mode='constant')
    else:
        processed_signal = fcgs_signal
    
    # 应用CWT（64个尺度，论文设置）
    scales = np.arange(1, CWT_SCALES + 1)
    cwt_coeffs = manual_cwt(processed_signal, scales)
    
    # 计算每个尺度的能量（论文公式：E(s) = sum|W(s,u)|^2）
    energy_features = np.sum(np.abs(cwt_coeffs) ** 2, axis=1)
    
    return energy_features

def extract_features(sequence):
    """提取单条序列的特征（与训练脚本逻辑一致，适配batch_size=1）"""
    fcgs_sig = fcgs2_signal(sequence)
    if len(fcgs_sig) == 0:
        # 处理过短序列：返回零向量（与训练脚本一致）
        cwt_energy = np.zeros(CWT_SCALES)
    else:
        cwt_energy = cwt_feature_extraction(fcgs_sig)
    return cwt_energy.reshape(1, -1)  # 适配模型输入（(1,64)）

# ===================== 模型加载函数 =====================
def load_all_models():
    """加载所有7个模型和对应的标签编码器"""
    models = {}
    encoders = {}
    for model_name, model_dir in MODEL_PATHS.items():
        # 加载模型
        model_path = os.path.join(model_dir, "svm_helitron_classifier.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        models[model_name] = joblib.load(model_path)
        
        # 加载标签编码器
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"标签编码器不存在：{encoder_path}")
        encoders[model_name] = joblib.load(encoder_path)
    
    print(f"✅ 成功加载所有{len(models)}个模型")
    return models, encoders

# ===================== 新增：辅助函数——根据真实标签递归获取所有后续模型 =====================
def get_all_subsequent_models(current_model, true_labels, current_level):
    """递归获取当前模型之后，按真实标签应有的所有后续模型及对应真实标签"""
    subsequent = []
    if current_level >= len(true_labels):
        return subsequent
    true_label = true_labels[current_level]
    if true_label == "invalid":
        return subsequent
    # 获取下一层模型
    next_model = CASCADE_MAP[current_model].get(true_label, None)
    if next_model is not None and next_model in MODEL_PATHS:
        subsequent.append( (next_model, true_labels[current_level + 1]) )  # 后续模型的真实标签是下一层级
        # 递归获取更下层模型
        subsequent.extend( get_all_subsequent_models(next_model, true_labels, current_level + 1) )
    return subsequent

# ===================== 级联测试核心逻辑 =====================
def cascade_test():
    # 1. 加载模型和编码器
    models, encoders = load_all_models()
    
    # 2. 加载并筛选测试集数据
    print("\n加载数据并筛选测试集...")
    # 读取train_test.txt（0=测试集，1=训练集）
    with open(TRAIN_TEST_PATH, 'r', encoding='utf-8') as f:
        train_test_flags = [int(line.strip()) for line in f if line.strip()]
    
    # 读取aligned_data.txt并匹配测试集
    test_data = []
    with open(ALIGNED_DATA_PATH, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        for idx, line in enumerate(lines):
            if idx >= len(train_test_flags):
                break  # 数据行数不匹配时终止
            if train_test_flags[idx] == 0:  # 仅保留测试集
                # 解析行：格式为 "3,class I,LTR,Gypsy,invalid,序列"
                parts = line.split(',', 5)  # 按第5个逗号分割（前5个是标签，最后是序列）
                if len(parts) != 6:
                    print(f"⚠️  跳过格式错误的行：{line}")
                    continue
                num_models = int(parts[0])  # 需调用的模型数量
                true_labels = parts[1:5]    # 4层真实标签（可能含invalid）
                sequence = parts[5]         # 基因序列
                processed_seq = process_sequence(sequence)
                test_data.append({
                    "num_models": num_models,
                    "true_labels": true_labels,
                    "sequence": processed_seq
                })
    
    print(f"✅ 筛选出测试集样本数：{len(test_data)}")
    
    # 3. 初始化评估结果存储（每个模型的真实标签+预测标签）
    eval_results = {
        model_name: {"y_true": [], "y_pred": [], "sample_count": 0}
        for model_name in MODEL_PATHS.keys()
    }
    
    # 4. 逐条测试（batch_size=1）
    print(f"\n开始级联测试（置信度阈值：{CONFIDENCE_THRESHOLD}）...")
    for idx, sample in enumerate(test_data):
        if (idx + 1) % 100 == 0:
            print(f"已测试 {idx + 1}/{len(test_data)} 条样本")
        
        num_models = sample["num_models"]
        true_labels = sample["true_labels"]
        sequence = sample["sequence"]
        
        # 提取特征（单条序列，与训练脚本逻辑一致）
        features = extract_features(sequence)
        
        # 级联测试状态
        current_model_name = "class"  # 起始模型
        cascade_failed = False         # 是否级联失败
        failed_model_index = -1        # 失败的模型层级（0=class，1=下一层，等）
        
        # 存储当前样本的各模型预测结果（用于后续评估）
        sample_preds = {model_name: "-1" for model_name in MODEL_PATHS.keys()}
        
        # ===================== 新增：提前获取该样本按真实标签应有的所有模型（包括后续）=====================
        all_involved_models = [(current_model_name, true_labels[0])]  # 起始模型+第0层真实标签
        all_involved_models.extend( get_all_subsequent_models(current_model_name, true_labels, 0) )
        
        # ===================== 新增：强制记录所有涉及模型的样本数和真实标签（预测标签初始为-1）=====================
        for model_name, tl in all_involved_models:
            eval_results[model_name]["sample_count"] += 1
            eval_results[model_name]["y_true"].append(tl)
            eval_results[model_name]["y_pred"].append("-1")
        
        # 按级联关系逐层测试（仅更新预测标签，不重复计数）
        for level in range(num_models):
            # 检查当前模型是否存在
            if current_model_name not in models:
                print(f"⚠️  模型不存在：{current_model_name}，跳过该层级")
                break
            
            # 获取当前模型和编码器
            model = models[current_model_name]
            encoder = encoders[current_model_name]
            
            # 预测：类别 + 置信度
            try:
                y_pred_proba = model.predict_proba(features)[0]  # 概率分布
                y_pred_idx = np.argmax(y_pred_proba)             # 预测类别索引
                y_pred_label = encoder.inverse_transform([y_pred_idx])[0]  # 预测标签
                y_pred_confidence = y_pred_proba[y_pred_idx]     # 预测置信度
            except Exception as e:
                print(f"⚠️  模型{current_model_name}预测失败：{str(e)}，标记为失败")
                cascade_failed = True
                failed_model_index = level
                break
            
            true_label = true_labels[level]
            
            # 检查是否级联失败
            if true_label == "invalid":
                # 真实标签为invalid，更新当前模型预测标签
                sample_preds[current_model_name] = y_pred_label if not cascade_failed else "-1"
                # 找到当前样本在该模型的最后一条记录，更新预测标签
                eval_results[current_model_name]["y_pred"][-1] = sample_preds[current_model_name]
                break
            
            if cascade_failed:
                # 之前层级已失败，保持预测标签为-1
                continue
            
            # 检查置信度和真实标签匹配性
            if y_pred_confidence >= CONFIDENCE_THRESHOLD:
                if y_pred_label == true_label:
                    # 置信度达标且预测正确 → 更新当前模型预测标签
                    eval_results[current_model_name]["y_pred"][-1] = y_pred_label
                    # 确定下一层模型
                    next_model_name = CASCADE_MAP[current_model_name].get(y_pred_label, None)
                    if next_model_name is None or level + 1 >= num_models:
                        # 无下一层模型或已达到指定层级数 → 终止级联
                        break
                    current_model_name = next_model_name
                else:
                    # 置信度达标但预测标签错误 → 记录错误预测标签（用于记为FP）
                    eval_results[current_model_name]["y_pred"][-1] = y_pred_label
                    cascade_failed = True
                    failed_model_index = level
                    break
            else:
                # 置信度低于阈值 → 记录为"LOW_CONF"（用于记为FN，不产生FP）
                eval_results[current_model_name]["y_pred"][-1] = "LOW_CONF"
                cascade_failed = True
                failed_model_index = level
                break
        
        # 处理未涉及的模型（样本不经过该模型，无需计入）
        pass
    
    # 5. 计算各模型的评估指标（根据您的需求重构的计算逻辑）
    print(f"\n开始计算各模型评估指标...")
    final_metrics = {}
    for model_name in MODEL_PATHS.keys():
        res = eval_results[model_name]
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        sample_count = res["sample_count"]
        
        if sample_count == 0 or model_name not in encoders:
            print(f"\n{model_name}模型：无测试样本")
            final_metrics[model_name] = {
                "sample_count": 0,
                "valid_sample_count": 0,
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "micro_f1": 0.0,
                "confusion_matrix": None,
                "class_metrics": {}
            }
            continue
        
        # 获取当前模型的所有合法类别
        model_classes = list(encoders[model_name].classes_)
        
        # 初始化各个类别的指标统计
        class_metrics = {c: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for c in model_classes}
        
        valid_sample_count = 0
        correct_count = 0
        
        # 逐样本计算TP、TN、FP、FN
        for yt, yp in zip(y_true, y_pred):
            if yt == "invalid":
                continue
            valid_sample_count += 1
            
            # 计算Accuracy使用
            if yt == yp:
                correct_count += 1
                
            for c in model_classes:
                is_true = (yt == c)
                is_pred = (yp == c)
                
                if yp == "LOW_CONF" or yp == "-1":
                    # 情况1：置信度低于阈值，或因为上游错误导致未到达该模型
                    # 记为该真实标签类别的FN，不产生FP（其余类记TN）
                    if is_true:
                        class_metrics[c]["FN"] += 1
                    else:
                        class_metrics[c]["TN"] += 1
                else:
                    # 情况2：置信度达标，且产生了预测标签（yp属于有效类别）
                    # 此时如果错误，会记为真实标签(yt)的FN以及预测标签(yp)的FP
                    if is_true and is_pred:
                        class_metrics[c]["TP"] += 1
                    elif is_true and not is_pred:
                        class_metrics[c]["FN"] += 1
                    elif not is_true and is_pred:
                        class_metrics[c]["FP"] += 1
                    else:
                        class_metrics[c]["TN"] += 1

        accuracy = correct_count / valid_sample_count if valid_sample_count > 0 else 0.0
        
        # 基于计算出的TP/TN/FP/FN，进一步计算各类别Precision, Recall, F1
        macro_f1_sum = 0.0
        total_TP = 0
        total_FP = 0
        total_FN = 0
        
        for c in model_classes:
            TP = class_metrics[c]["TP"]
            FP = class_metrics[c]["FP"]
            FN = class_metrics[c]["FN"]
            TN = class_metrics[c]["TN"]
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[c]["precision"] = precision
            class_metrics[c]["recall"] = recall
            class_metrics[c]["f1"] = f1
            
            macro_f1_sum += f1
            total_TP += TP
            total_FP += FP
            total_FN += FN
            
        macro_f1 = macro_f1_sum / len(model_classes) if len(model_classes) > 0 else 0.0
        
        # 计算全局微平均指标
        micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # 为兼容底部原代码的保存逻辑，将class_metrics转存为DataFrame（替代原先标准的混淆矩阵）
        if valid_sample_count > 0:
            conf_matrix_df = pd.DataFrame(class_metrics).T
            # 将列按照易读顺序排列
            conf_matrix_df = conf_matrix_df[['precision', 'recall', 'f1', 'TP', 'TN', 'FP', 'FN']]
        else:
            conf_matrix_df = None
        
        # 存储指标
        final_metrics[model_name] = {
            "sample_count": sample_count,
            "valid_sample_count": valid_sample_count,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "confusion_matrix": conf_matrix_df,
            "class_metrics": class_metrics
        }
    
    # 6. 输出最终评估结果
    print(f"\n\n========== 级联测试最终结果 ==========")
    for model_name in MODEL_PATHS.keys():
        metrics = final_metrics[model_name]
        print(f"\n【{model_name}模型】")
        print(f"测试样本总数：{metrics['sample_count']}")
        print(f"有效测试样本数（排除invalid）：{metrics['valid_sample_count']}")
        print(f"准确率（Accuracy）：{metrics['accuracy']:.4f}")
        print(f"宏平均F1（Macro-F1）：{metrics['macro_f1']:.4f}")
        print(f"微平均F1（Micro-F1）：{metrics['micro_f1']:.4f}")
        
        if metrics["class_metrics"]:
            print(f"各类别评估指标：")
            print(f"{'类别':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6}")
            print("-" * 85)
            for c, cm in metrics["class_metrics"].items():
                print(f"{c:<15} {cm['precision']:<10.4f} {cm['recall']:<10.4f} {cm['f1']:<10.4f} {cm['TP']:<6} {cm['TN']:<6} {cm['FP']:<6} {cm['FN']:<6}")
        else:
            print(f"各类别评估指标：无有效数据")
    
    # 7. 保存结果到文件
    output_dir = "/public/home/h2024319020/mgy/SVM-Helitron recognizer/cascade_test_results/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存汇总指标
    summary_data = []
    for model_name in MODEL_PATHS.keys():
        metrics = final_metrics[model_name]
        summary_data.append({
            "模型名称": model_name,
            "测试样本总数": metrics["sample_count"],
            "有效样本数": metrics["valid_sample_count"],
            "准确率": metrics["accuracy"],
            "宏平均F1": metrics["macro_f1"],
            "微平均F1": metrics["micro_f1"]
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "cascade_test_summary.csv"), index=False, encoding='utf-8-sig')
    
    # 保存各模型混淆矩阵
    for model_name in MODEL_PATHS.keys():
        metrics = final_metrics[model_name]
        if metrics["confusion_matrix"] is not None:
            metrics["confusion_matrix"].to_csv(
                os.path.join(output_dir, f"{model_name}_confusion_matrix.csv"),
                encoding='utf-8-sig'
            )
    
    print(f"\n所有结果已保存到：{output_dir}")
    print("\n级联测试完成！")

# ===================== 主函数 =====================
if __name__ == "__main__":
    cascade_test()