import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve
import numpy as np
import pandas as pd

plt.rcParams.update({
        'font.size': 20,           # 默认字体大小
        'axes.titlesize': 20,      # 子图标题大小
        'axes.labelsize': 16,      # 坐标轴标签大小
        'xtick.labelsize': 16,     # x轴刻度标签大小
        'ytick.labelsize': 16,     # y轴刻度标签大小
        'legend.fontsize': 20,     # 图例字体大小
        'figure.titlesize': 20     # 总标题大小
})

def plot_2D_PCA(X_processed, anomaly_scores, K=50, title_suffix="", feature_description=""):
    """绘制2D PCA图，突出显示前K个异常点"""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_processed)
    
    topK_idx = np.argsort(-anomaly_scores)[:K]
    
    plt.figure(figsize=(8, 6))
    
    sc = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=anomaly_scores,
        s=10,
        alpha=0.7
    )
    
    plt.scatter(
        X_pca[topK_idx, 0],
        X_pca[topK_idx, 1],
        facecolors='none',
        edgecolors='red',
        s=60,
        linewidths=1.5,
        label=f"Top {K} anomalies"
    )
    # plt.xlim(-2500, 2500)
    plt.colorbar(sc, label="Anomaly score (higher = more anomalous)")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(f"{title_suffix}\nFeature: {feature_description}")
    plt.savefig(f"pca_topK_{title_suffix.replace(' ', '_')}_{feature_description.replace(' ', '_')}.png", dpi=300)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# plot_2D_PCA(X_processed, anomaly_scores_best, K=50, title_suffix=best_detector['detector'], feature_description="RobustScaler_context_filter_detail")
def plot_score_distribution(anomaly_scores_fw, y):
    normal_scores = anomaly_scores_fw[y == 0]
    outlier_scores = anomaly_scores_fw[y == 1]
    plt.figure(figsize=(12, 6))

    # 创建主坐标轴
    ax1 = plt.gca()

    # 绘制正常样本（主y轴）
    sns.histplot(normal_scores, bins=50, kde=True, color='blue', alpha=0.6, 
                label='Normal Samples', ax=ax1)

    # 创建第二个y轴（共享x轴）
    ax2 = ax1.twinx()

    # 绘制异常样本（次y轴）
    sns.histplot(outlier_scores, bins=30, kde=True, color='red', alpha=0.6, 
                label='True Outliers', ax=ax2)

    # 设置第二个y轴的范围（根据异常样本数量调整）
    ax2.set_ylim(0, len(outlier_scores) * 0.3)  # 调整为异常样本数量的30%

    # 设置标签
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Normal Samples Frequency', color='blue')
    ax2.set_ylabel('Outliers Frequency', color='red')

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title('Anomaly Score Distribution (Separate Y-axis for Outliers)')
    plt.show()
    
def plot_score_distribution(anomaly_scores_fw, y, title = ""):
    normal_scores = anomaly_scores_fw[y == 0]
    outlier_scores = anomaly_scores_fw[y == 1]
    plt.figure(figsize=(12, 6))

    # 创建主坐标轴
    ax1 = plt.gca()

    # 绘制正常样本（主y轴）
    sns.histplot(normal_scores, bins=50, kde=True, color='blue', alpha=0.6, 
                label='Normal Samples', ax=ax1)

    # 创建第二个y轴（共享x轴）
    ax2 = ax1.twinx()

    # 绘制异常样本（次y轴）
    sns.histplot(outlier_scores, bins=30, kde=True, color='red', alpha=0.6, 
                label='True Outliers', ax=ax2)

    # 设置第二个y轴的范围（根据异常样本数量调整）
    ax2.set_ylim(0, len(outlier_scores) * 0.2)  # 调整为异常样本数量的30%

    # 设置标签
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Normal Samples Frequency', color='blue')
    ax2.set_ylabel('Outliers Frequency', color='red')

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title(title)
    plt.show()