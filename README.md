

```markdown
# Unsupervised Anomaly Detection on Thyroid Disease Data

## 📌 Project Overview

This project implements unsupervised anomaly detection on the **Annthyroid Unsupervised Anomaly Detection dataset**, focusing on identifying clinically meaningful thyroid dysfunction patterns. Our work demonstrates how advanced outlier detection methods can uncover subtle abnormalities in medical data, with particular emphasis on context-aware detection.

### 🔍 Key Contributions
1. **Data Quality Enhancement**: Identified and corrected inconsistent hormone scaling in the original dataset
2. **Feature-Weighted Isolation Forest (FWIF)**: Improved baseline performance through feature importance weighting
3. **Contextual Ensemble Framework**: Novel approach considering demographic and treatment contexts
4. **Clinical Pattern Discovery**: Revealed important inconsistencies in data labeling logic

## 📊 Performance Highlights

| Method | ROC-AUC | PR-AUC | Precision@50 |
|--------|---------|--------|--------------|
| Baseline Isolation Forest | 0.5975 | 0.0532 | 0.0400 |
| Feature-Weighted IF (FWIF) | 0.7330 | 0.1835 | 0.4600 |
| **Contextual Ensemble (Ours)** | **0.9624** | **0.5649** | **0.6600** |

*3.08× improvement in PR-AUC over baseline methods*


## 🗂️ Project Structure
.
│  annthyroid_unsupervised_anomaly_detection.csv        # original dataset from Kaggle
│  anomaly_association_rules.csv        # rule mined on outlier samples by fpgrowth (code omitted)
│  context_utils.py         # helper functions to run contextual ensemble method
│  data_cleaning.ipynb      # code comparing difference between raw dataset and Kaggle dataset, and generate the cleaned dataset named "thyroid_processed_data_cleaned.csv"
│  eval_utils.py        # helper functions of eval
│  final_anomaly_detection_results.csv      # results that storing performance of difference configuration of contextual ensemble 
│  pca_topK_Feature-Weighted_IF_Detector_FWIF_top_features.png      # PCA visualization of FWIF
│  pca_topK_LOF_Detector_LOF_top_features.png       # PCA visualization of single LOF
│  project.ipynb        # main code! generate all experiments in the essay
│  README.md        # this file
│  requirements.txt         # some package requirement to run code in this folder
│  robust_standard_comp.ipynb       # code examine the difference between StandardScaler and RobustScaler
│  run_autoencoder.py       # code to run autoencoer
│  scaler_comparison.png        # figure examine the difference between StandardScaler and RobustScaler
│  thyroid_processed_data_cleaned.csv       # cleaned dataset
│  thyroid_raw_data.csv         # raw dataset found by reference
│  top_anomalies_with_notes.csv         # csv file to analyze clinical features of outlier
│  visual_utils.py      # helper function help with visualization
│  

```

## 🚀 Quick Start

### Prerequisites
- Python 3.8.20
- Windows

### Installation

```bash
pip install -r requirements.txt
```

3. **Run the main analysis**
```bash
# Open and execute project.ipynb in Jupyter
jupyter notebook project.ipynb
```

## 🔬 Methodology

### 1. Data Understanding & Preprocessing 
- **Missing value handling**: Removed empty columns
- **Distribution normalization**: Log(1+x) transformation for skewed hormone distributions
- **Feature scaling**: StandardScaler for uniform feature scales

### 2. Baseline Models 
- **Isolation Forest**: Base unsupervised detector
- **Local Outlier Factor (LOF)**: Density-based contrastive baseline
- **Autoencoder**: Deep reconstruction-based approach

### 3. Feature-Weighted Improvement 
- **FWIF**: Weighted features based on importance scores
- **11.5× improvement** in Precision@50 over baseline

### 4. Contextual Ensemble Framework 
- **Multi-context modeling**: Demographic, treatment, and physiological contexts
- **Overlapping subgroups**: Patients belong to multiple relevant contexts
- **Intelligent aggregation**: Context-specific scores combined via averaging/maximum

### 5. Data Quality Enhancement 
- **Scale correction**: Identified and fixed inconsistent hormone scaling
- **Clinical validation**: Aligned data with physiological reference ranges
- **Age normalization**: Standardized age representation

## 📈 Key Findings

See `report.pdf`, `top_anomalies_with_notes.csv` or `project.ipynb` step 11 for more details.

## 👥 Team Contributions

### Yuqi Ren (Member A)
- Data preprocessing and exploratory analysis
- Baseline Isolation Forest implementation
- Feature-Weighted Isolation Forest (FWIF) development
- PCA visualization and robustness studies

### Anyi Wang (Member B)
- Local Outlier Factor (LOF) experiments and optimization
- Autoencoder implementation and evaluation
- Comparative analysis of different detectors
- Visualization of detector performance

### Feng Liang (Member C)
- Data cleaning and scale correction pipeline
- Contextual ensemble framework design and implementation
- Rule mining on outlier samples
- Comprehensive experimental evaluation

## 📊 Reproducing Results

### Step-by-Step Execution
1. **Data Preparation** (`data_cleaning.ipynb`)
   - Compare raw vs. processed data scales
   - Generate cleaned dataset with consistent clinical units

2. **Main Analysis** (`project.ipynb`)
   - Execute cells sequentially
   - All experiments are modular and self-contained

3. **Additional Studies** (`robust_standard_comp.ipynb`, `run_autoencoder.py`)
   - Scaler comparison
   - Autoencoder training

### Expected Outputs
- Cleaned dataset: `thyroid_processed_data_cleaned.csv`
- Performance results: `final_anomaly_detection_results.csv`
- Visualizations: All PNG files in working directory
- Clinical analysis: `top_anomalies_with_notes.csv`

## 📚 References

1. **Dataset Sources**
   - Annthyroid Dataset (Kaggle): https://www.kaggle.com/datasets/zhonglifr/thyroid-disease-unsupervised-anomaly-detection
   - UCI Thyroid Disease Dataset: https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease

2. **Key Algorithms**
   - Liu, F. T. , Ting, K. M. , & Zhou, Z. H. . (2009). Isolation forest. IEEE.
   - Breunig, M. M. , Kriegel, H. P. , Ng, R. T. , & Jörg Sander. (2000). Lof: identifying density-based local outliers. Proc.acm Sigmod Int.conf.on Management of Data.

3. **Clinical References**
   - Deepseek
   - <<诊断学>>

## 🙏 Acknowledgments

- HKUST(GZ) for academic support
- Kaggle and UCI for dataset provision
- Open-source community for Python libraries
- Medical guidance by AI and open-source websites and books

---

*For detailed methodology, results, and discussion, please refer to the complete project report (`report.pdf`).*
