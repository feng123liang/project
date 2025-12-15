
## Project File Sturcutre

```
.
│  annthyroid_unsupervised_anomaly_detection.csv # original dataset from Kaggle
│  anomaly_association_rules.csv # rule mined on outlier samples by fpgrowth (code omitted)
│  context_utils.py # helper functions to run contextual ensemble method
│  data_cleaning.ipynb # code comparing difference between raw dataset and Kaggle dataset, and generate the cleaned dataset named "thyroid_processed_data_cleaned.csv"
│  eval_utils.py # helper functions of eval
│  final_anomaly_detection_results.csv # results that storing performance of difference configuration of contextual ensemble 
│  pca_topK_Feature-Weighted_IF_Detector_FWIF_top_features.png # PCA visualization of FWIF
│  pca_topK_LOF_Detector_LOF_top_features.png # PCA visualization of single LOF
│  project.ipynb # main code! generate all experiments in the essay
│  README.md # this file
│  requirements.txt # some package requirement to run code in this folder
│  robust_standard_comp.ipynb # code examine the difference between StandardScaler and RobustScaler
│  run_autoencoder.py # code to run autoencoer (member B)
│  scaler_comparison.png # figure examine the difference between StandardScaler and RobustScaler
│  thyroid_processed_data_cleaned.csv # cleaned dataset
│  thyroid_raw_data.csv # raw dataset found by reference
│  top_anomalies_with_notes.csv # csv file to analyze clinical features of outlier
│  visual_utils.py # helper function help with visualization
│  
```
## Setup

We run the code under python 3.8.20, in Windows system. Before run the code in project.ipynb, data_cleaning.ipynb, run_autoencoder.py and robust_standard_comp.ipynb. run `pip install -r requirements.txt` first 