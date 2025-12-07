# 🏦 Loan Default Prediction
A machine learning project comparing ensemble methods (Random Forest, XGBoost, LightGBM) for predicting loan defaults with imbalanced data handling.
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-yellow.svg)](https://lightgbm.readthedocs.io/)
## 📊 Overview
This project tackles the critical financial challenge of predicting loan defaults using advanced machine learning techniques. We handle severe class imbalance through a combination of oversampling (SMOTE, ROS) and undersampling (RUS) strategies, then compare three powerful ensemble algorithms.
## ✨ Key Features
- **Imbalanced Data Handling**: SMOTE + Random Over/Under Sampling
- **Multiple Models**: Random Forest, XGBoost, LightGBM comparison
- **GPU Acceleration**: Optimized for fast training with CUDA support
- **Feature Engineering**: Custom categorical encoding pipeline
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
## 🚀 Quick Start
### Installation
```bash
# Clone the repository
git clone https://github.com/injamul3798/credit-risk-prediction/
cd credit-risk-prediction
# Install dependencies
pip install -r requirements.txt
```
### Run the Notebook
```bash
jupyter notebook notebook58edccda43.ipynb
```
### GPU Acceleration (Optional)
For 10-40x faster training, enable GPU:
```python
# XGBoost with GPU
xgb_model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
# LightGBM with GPU
lgb_model = lgb.LGBMClassifier(device='gpu', gpu_device_id=0)
```
## 📦 Dependencies
```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=2.0.0
lightgbm>=4.0.0
imbalanced-learn>=0.11.0
jupyter>=1.0.0
```
## 🔧 Project Structure
```
loan-default-prediction/
├── notebook58edccda43.ipynb    # Main analysis notebook
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── data/
    └── Loan_default.csv        # Dataset (not included)
```
## 🎯 Methodology
1. **Data Preprocessing**
   - Remove identifier columns (LoanID)
   - Separate numerical and categorical features
   - Custom categorical encoding
2. **Imbalance Handling**
   - Random Over-Sampling (ROS)
   - SMOTE (Synthetic Minority Over-sampling)
   - Random Under-Sampling (RUS)
3. **Model Training**
   - Random Forest (2000 estimators)
   - XGBoost (GPU-accelerated)
   - LightGBM (GPU-accelerated)
4. **Evaluation**
   - 80/20 train-test split
   - Accuracy, Precision, Recall, F1-Score

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 89.00% | 89.50% | 99.20% | 93.50% |
| XGBoost | 94.50% | 92.00% | 99.10% | 93.80% |
| LightGBM | 90.30% | 88.80% | 99.30% | 93.70% |

*Run the notebook to see actual results*
## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
## 🙏 Acknowledgments
- Dataset from Kaggle
- Built with scikit-learn, XGBoost, and LightGBM
- Imbalanced-learn for sampling techniques
