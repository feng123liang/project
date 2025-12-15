import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import precision_recall_curve, auc
from sklearn.compose import ColumnTransformer
from eval_utils import evaluate_anomaly_model, precision_at_k

def preprocess_data(df, numeric_features: list, binary_features: list, preprocess=None, columns_drop=None):
    """
        预处理数值和二元特征
        preprocess: str, \n
            可选的预处理组合，如"log", "standard", "robust"
        columns_drop: list or None,
    """
    X_processed = df.copy()
    
    # log transform if specified
    if "log" in preprocess:
        for feature in numeric_features:
            X_processed[feature] = np.log1p(X_processed[feature])
    
    # drop columns if specified
    if columns_drop is not None:
        X_processed = X_processed.drop(columns=columns_drop)

    # choose scaler based on preprocess argument
    transformers = [('bin', 'passthrough', [f for f in binary_features if f in X_processed.columns])]
    if "robust" in preprocess:
        transformers.insert(0, ('num', RobustScaler(), [f for f in numeric_features if f in X_processed.columns]))
    elif "standard" in preprocess:
        transformers.insert(0, ('num', StandardScaler(), [f for f in numeric_features if f in X_processed.columns]))
    else:
        transformers.insert(0,('num', 'passthrough', [f for f in numeric_features if f in X_processed.columns]))
   
    preprocessor = ColumnTransformer(transformers=transformers)
    X_processed = preprocessor.fit_transform(X_processed)
    
    feature_names = (numeric_features + 
                    [f for f in binary_features if f in df.columns])
    
    return X_processed, preprocessor, feature_names

# ==================== 步骤2: 创建上下文分组 ====================

def create_context_groups(df, drop = False, limits = None):
    """基于领域知识创建有意义的上下文分组"""
    contexts = []
    
    # 上下文1: 按年龄和性别分组 (最重要的医学上下文)
    df['age_group'] = pd.cut(df['Age'], 
                            bins=[0, 0.30, 0.50, 0.70, float('inf')], 
                            labels=['young', 'middle', 'senior', 'elderly'])
    
    # 上下文2: 按治疗状态分组
    df['treatment_status'] = 'no_treatment'
    df.loc[df['on_thyroxine'] == 1, 'treatment_status'] = 'on_thyroxine'
    df.loc[df['on_antithyroid_medication'] == 1, 'treatment_status'] = 'on_antithyroid'
    df.loc[(df['thyroid_surgery'] == 1) | (df['I131_treatment'] == 1), 'treatment_status'] = 'post_treatment'
    
    # 上下文3: 按特殊生理状态分组
    df['special_status'] = 'general'
    df.loc[df['pregnant'] == 1, 'special_status'] = 'pregnant'
    df.loc[df['lithium'] == 1, 'special_status'] = 'on_lithium'
    
    # 生成上下文组合 - 选择最有医学意义的组合
    context_combinations = [
        # 组合1: 年龄 × 性别 (基础分组)
        {'name': 'age_sex', 'attributes': ['age_group', 'Sex']},
        
        # 组合2: 年龄 × 治疗状态
        {'name': 'age_treatment', 'attributes': ['age_group', 'treatment_status']},
        
        # 组合3: 性别 × 特殊状态
        {'name': 'sex_special', 'attributes': ['Sex', 'special_status']},
        
        # 组合4: 仅治疗状态 (对于药物相关的异常)
        {'name': 'treatment_only', 'attributes': ['treatment_status']},
        
        # 组合5: 仅特殊状态
        {'name': 'special_only', 'attributes': ['special_status']}
    ]
    
    # 为每个组合创建上下文
    for combo in context_combinations:
        if len(combo['attributes']) == 1:
            # 单属性分组
            for value in df[combo['attributes'][0]].unique():
                mask = df[combo['attributes'][0]] == value
                if mask.sum() > 15:  # 确保有足够样本
                    context_name = f"{combo['name']}_{value}"
                    contexts.append({
                        'name': context_name,
                        'mask': mask,
                        'data': df[mask],
                        'indices': df[mask].index
                    })
        else:
            # 双属性组合分组
            attr1, attr2 = combo['attributes']
            for value1 in df[attr1].unique():
                for value2 in df[attr2].unique():
                    mask = (df[attr1] == value1) & (df[attr2] == value2)
                    if mask.sum() > 15:  # 确保有足够样本
                        context_name = f"{combo['name']}_{value1}_{value2}"
                        contexts.append({
                            'name': context_name,
                            'mask': mask,
                            'data': df[mask],
                            'indices': df[mask].index
                        })
    # 检查没有覆盖到的样本
    mask = np.zeros(len(df), dtype=bool)
    # 随机drop1-8个上下文
    if drop:
        random_indices = np.random.choice(range(len(contexts)), len(contexts) - num_to_drop, replace=False)
        contexts = [contexts[i] for i in random_indices]    
        num_to_drop = np.random.randint(1, 9)
    else:
        num_to_drop = 0
    # # print(type(contexts))
    
    if limits is not None:
        contexts = [context for context in contexts if context['name'] in limits]
    
    for context in contexts:
        mask |= context['mask']
        # 输出该上下文覆盖的样本数量
        print(f"上下文 '{context['name']}' 覆盖样本数: {len(context['indices'])}")
    
    print(f"未被任何上下文覆盖的样本数: {(~mask).sum()}")
    # if (~mask).sum() > 0:
    #     print(df.loc[~mask])
    print(f"创建了 {len(contexts)} 个上下文分组")
    return contexts, df

# groups = ['age_treatment_elderly_post_treatment', 'age_sex_senior_0.0', 'age_treatment_senior_no_treatment', 'treatment_only_on_antithyroid', 'age_treatment_young_no_treatment', 'age_treatment_senior_on_thyroxine', 'special_only_on_lithium', 'sex_special_0.0_pregnant', 'age_treatment_middle_post_treatment', 'age_sex_middle_0.0', 'age_sex_elderly_1.0', 'age_treatment_elderly_on_thyroxine', 'age_treatment_senior_on_antithyroid', 'treatment_only_post_treatment', 'age_sex_middle_1.0', 'special_only_pregnant', 'age_treatment_middle_on_thyroxine', 'sex_special_0.0_on_lithium', 'special_only_general', 'age_treatment_middle_no_treatment', 'age_treatment_elderly_no_treatment', 'age_sex_young_0.0', 'age_treatment_senior_post_treatment', 'age_sex_senior_1.0', 'age_sex_young_1.0', 'treatment_only_on_thyroxine', 'sex_special_1.0_on_lithium', 'age_treatment_young_on_thyroxine']
# contexts, df_with_context = create_context_groups(X, limits=groups)

# ==================== 步骤3: 初始化检测器 ====================

# 修改检测器配置为可扩展的版本
def create_detectors(n_neighbors_grid=[50, 75, 90, 110, 130], contamination_grid=[0.01, 0.05, 'auto']):
    """创建多个LOF检测器配置 \n
        param n_neighbors_grid: list of int, 邻居数列表 \n
        param contamination_grid: list of float or 'auto', 污染率列表
    """
    detectors = {}

    for n_neighbors in n_neighbors_grid:
        for contamination in contamination_grid:
            detector_name = f'LOF_k{n_neighbors}_c{contamination}'
            detectors[detector_name] = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                novelty=False
            )
    
    # 添加Isolation Forest作为基准
    detectors['IsolationForest'] = IsolationForest(
        n_estimators=100,
        contamination='auto',
        random_state=42
    )
    
    print(f"创建了 {len(detectors)} 个检测器")
    return detectors

# 替换原来的detectors定义
# detectors = create_detectors()
# print(detectors)

# print("步骤3: 检测器初始化完成")

# ==================== 步骤4: 上下文异常检测 ====================

def run_contextual_detection(contexts, numeric_features, binary_features, detectors, y_true, method="discrete", preprocess="None", columns_drop=None, verbose=False):

    """
    在每个上下文中运行异常检测 - 支持LOF 和Isolation Forest \n
    method: 
        "discrete" - 分数为离散值 (LOF)
        "continuous" - 分数为连续值
        
    preprocess: str, 可选的预处理组合\n
        如"log", "standard", "robust"
    """
    all_context_results = []
    
    for i, context in enumerate(contexts):
        context_name = context['name']
        context_data = context['data']
        context_indices = context['indices']
        
        if verbose:
            print(f"\n处理上下文 {i+1}/{len(contexts)}: '{context_name}' (样本数: {len(context_data)})")
        
        # 跳过样本数过少的上下文
        if len(context_data) < 25:
            continue
        
        try:
            # 准备特征数据
            features_to_use = numeric_features + binary_features
            context_data_features = context_data[features_to_use].copy()
            X_context, preprocessor, feature_names = preprocess_data(context_data_features, numeric_features, binary_features, 
                                                                     preprocess=preprocess, columns_drop=columns_drop)
            
            context_labels = y_true.loc[context_indices]
            
            # 对每个检测器运行
            context_detector_results = []
            
            for detector_name, detector_config in detectors.items():
                try:
                    # 动态调整参数避免错误
                    current_detector = None
                    
                    if 'LOF' in detector_name:
                        # 对于LOF，动态调整邻居数
                        n_neighbors = detector_config.n_neighbors
                        if isinstance(n_neighbors, int):
                            n_neighbors = min(n_neighbors, len(context_data) - 1)
                            if n_neighbors < 2:  # 最少需要2个邻居
                                continue
                        
                        current_detector = LocalOutlierFactor(
                            n_neighbors=n_neighbors,
                            contamination=detector_config.contamination,
                            novelty=False
                        )
                        
                        if method == "discrete":
                            # 离散分数
                            scores = -current_detector.fit_predict(X_context)
                        else:
                            predction = current_detector.fit_predict(X_context)
                            scores = -current_detector.negative_outlier_factor_
                            
                    elif 'IsolationForest' in detector_name:
                        current_detector = IsolationForest(
                            n_estimators=min(100, len(context_data)),
                            contamination=detector_config.contamination,
                            random_state=42
                        )
                        current_detector.fit(X_context)
                        scores = -current_detector.decision_function(X_context)
                    
                    # 计算评估指标
                    k_val = min(50, len(scores) // 5)
                    if k_val > 0:
                        precision_val = precision_at_k(context_labels, scores, k=k_val)
                        
                        result = {
                            'context': context_name,
                            'detector': detector_name,
                            'sample_size': len(context_data),
                            'precision': precision_val,
                            'scores': scores,
                            'indices': context_indices
                        }
                        
                        context_detector_results.append(result)
                        all_context_results.append(result)
                        
                        # 打印最佳结果
                        if verbose and precision_val > 0.5:  # 只打印较好的结果避免输出过多
                            print(f"  {detector_name}: Precision@{k_val} = {precision_val:.4f}")
                    
                except Exception as e:
                    # 静默处理错误，继续下一个检测器
                    continue
            
            # 打印该上下文的最佳检测器
            if context_detector_results:
                best_result = max(context_detector_results, key=lambda x: x['precision'])
                if verbose:
                    print(f"  最佳: {best_result['detector']} (Precision: {best_result['precision']:.4f})")
                    
        except Exception as e:
            if verbose:
                print(f"处理上下文 {context_name} 时出错: {e}")
            continue
    
    if verbose:
        print(f"成功完成 {len(all_context_results)} 个检测器-上下文组合")
    return all_context_results

# ==================== 步骤5: 结果聚合 ====================

def aggregate_scores(context_results, detectors, total_samples, contexts, normalization="min-max", ensemble_method="avg"):
    """
    聚合所有上下文的异常分数\n
    normalization: 
        "min-max" - 归一化到0-1范围 \n
        "z-score" - 标准化为z-score \n
        "none" - 不进行归一化 \n
    ensemble_method: 
        "avg" - 平均分数 \n
        "max" - 最大分数 \n
    """
    # 为每个检测器创建分数存储
    detector_scores = {}
    detector_counts = {}
    # 初始化所有检测器的存储
    for detector_name in detectors.keys():
        detector_scores[detector_name] = np.zeros((total_samples, len(contexts)))
        detector_counts[detector_name] = np.zeros(total_samples)
    # 累加分数
    for result in context_results:
        detector_name = result['detector']
        indices = result['indices']
        scores = result['scores']
        
        # if 'LOF' in detector_name and normalization != "none":
        #     # LOF分数通常集中在1.0附近，异常点>1.0
        #     scores = scores - 1.0 
        
        if normalization == "none":
            pass
        elif normalization == "min-max":
            # Min-Max归一化
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min() )
        elif normalization == "z-score":
            # Z-score标准化
            scores = (scores - np.mean(scores)) / np.std(scores)
            
        # 记录分数
        detector_scores[detector_name][indices, contexts.index(result['context'])] = scores
        detector_counts[detector_name][indices] += 1
        
    # 对每个检测器不同context下的分数进行ensemble
    final_scores = {}
    for detector_name in detectors.keys():
        scores_combined = detector_scores[detector_name]
        counts = detector_counts[detector_name]
        
        final_scores[detector_name] = np.zeros(total_samples)
        mask = counts > 0
        
        if np.sum(mask) > 0:
            if ensemble_method == "max":
                # 取最大分数
                final_scores[detector_name][mask] = np.max(scores_combined[mask], axis=1)
            elif ensemble_method == "avg":
                # 取平均分数
                final_scores[detector_name][mask] = np.sum(scores_combined[mask], axis=1) / counts[mask]
    
    return final_scores

# ==================== 步骤6: 评估结果 ====================


def evaluate_final_results(final_scores, y_true):
    """评估最终结果 - 多检测器版本"""
    evaluation_results = {}
    detector_performance = []
    
    for detector_name, scores in final_scores.items():
        # print(f"评估检测器: {detector_name}, 有效样本数: {len(scores)}", len(y_true))
        result = evaluate_anomaly_model(y_true, scores, K=50)
        roc_auc = result['ROC-AUC']
        pr_auc = result['PR-AUC']
        precision_50 = result['Precision@50']
        
        evaluation_results[detector_name] = {
            'Precision@50': precision_50,
            'PR-AUC': pr_auc,
            'ROC-AUC': roc_auc,
        }
        
        detector_performance.append({
            'detector': detector_name,
            'PR-AUC': pr_auc,
            'Precision@50': precision_50
        })
        
    # 按性能排序
    detector_performance.sort(key=lambda x: x['PR-AUC'], reverse=True)
    
    return evaluation_results, detector_performance

def save_best_detector_scores(final_scores, best_detector, contexts, y):
    """保存最佳检测器的分数到CSV文件"""
    best_detector_name = best_detector['detector']
    best_scores = final_scores[best_detector_name]
    context_name_list = [context['name'] for context in contexts]
    df_best_scores = pd.DataFrame({
        'anomaly_score': best_scores,
        'Outlier_label': y,
    })
    best_scores_path = f'contextual_anomaly_scores_{best_detector_name}.csv'
    # 如果存在，则增加编号
    
    base_path = f'contextual_anomaly_scores_{best_detector_name}'
    ext = '.csv'
    counter = 1
    while os.path.exists(best_scores_path):
        best_scores_path = f"{base_path}_{counter}{ext}"
        counter += 1
    df_best_scores.to_csv(best_scores_path, index=False)
    print(f"最佳检测器分数已保存到: {best_scores_path}")
    with open(best_scores_path.replace('.csv', '.txt'), 'w') as f:
        f.write(f"Best Detector: {best_detector_name}\n")
        f.write(f"PR-AUC: {best_detector['PR-AUC']:.4f}\n")
        f.write(f"Precision@50: {best_detector['Precision@50']:.4f}\n")
        f.write(str(context_name_list))
        f.write(str(len(context_name_list)))
    print(f"最佳检测器信息已保存到: {best_scores_path.replace('.csv', '.txt')}")
