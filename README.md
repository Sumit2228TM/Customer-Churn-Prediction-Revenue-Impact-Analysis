# 📉 Customer Churn Prediction & Revenue Impact Analysis

A machine learning solution to predict customer churn and quantify revenue impact, enabling data-driven retention strategies for subscription-based businesses.

## 🎯 Project Overview

This project tackles customer churn prediction for a subscription service, identifying at-risk customers and estimating revenue impact. The model analyzes 5,000+ customer records to predict churn with **75% accuracy** and provides actionable insights for retention campaigns.

**Business Impact:** Estimated **20% reduction in lost revenue** through targeted retention offers.

---

## ✨ Key Features

- **Predictive Churn Model**: XGBoost classifier with 75% accuracy
- **Advanced Feature Engineering**: 6 custom features including engagement scores
- **Revenue Impact Analysis**: Cost-benefit framework for retention ROI
- **Customer Segmentation**: Risk-based tiering (High/Medium/Low)
- **Actionable Insights**: Prioritized list of at-risk customers
- **Interactive Dashboard**: Visualizations for business stakeholders

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **ML Framework**: XGBoost, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Feature Engineering**: Custom transformers
- **Metrics**: AUPRC, Precision-Recall, ROC-AUC

---

## 📊 Dataset

- **Records**: 5,000+ customer profiles
- **Features**: 
  - **Demographics**: Age, gender, location
  - **Account Info**: Tenure, contract type, payment method
  - **Engagement**: Last_login, avg_watch

**Target Variable**: Churn (1 = Churned, 0 = Retained)

**Churn Rate**: ~27% (moderate imbalance)

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0
```


---

## 💻 Usage

### 1. Data Preprocessing
```python
from src.data_processing import load_and_clean_data

# Load data
df = load_and_clean_data('data/raw/customer_data.csv')
```

### 2. Feature Engineering
```python
from src.feature_engineering import create_features

# Generate custom features
df_enhanced = create_features(df)

# New features created:
# - tenure
# - service_usage
```

### 3. Train Model
```python
from src.model import train_churn_model

# Train XGBoost classifier
model, metrics = train_churn_model(X_train, y_train)
```

### 4. Predict Churn
```python
# Get churn probability
churn_prob = model.predict_proba(X_test)[:, 1]

# Classify customers by risk
high_risk = X_test[churn_prob > 0.7]
medium_risk = X_test[(churn_prob > 0.4) & (churn_prob <= 0.7)]
low_risk = X_test[churn_prob <= 0.4]
```

### 5. Revenue Impact Analysis
```python
from src.revenue_impact import calculate_revenue_impact

# Estimate savings from retention campaign
impact = calculate_revenue_impact(
    churn_predictions=churn_prob,
    customer_values=monthly_charges,
    retention_cost=50,  # Cost per retention offer
    retention_success_rate=0.3
)

print(f"Expected ROI: ${impact['net_savings']:,.2f}")
```

---

## 📈 Model Performance

### Classification Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.75 |
| **Precision** | 0.68 |
| **Recall** | 0.72 |
| **F1-Score** | 0.70 |
| **AUPRC** | 0.87 |
| **ROC-AUC** | 0.84 |

### Feature Engineering Impact

**Before Feature Engineering**: 69% accuracy  
**After Feature Engineering**: 75% accuracy  
**Improvement**: +6 percentage points

### Top Engineered Features
1. **tenure**: Customer lifecycle stage
2. **service_usage**: Time spent using service

---

## 🎨 Visualizations

### Generated Plots
1. **Churn Distribution** by customer segments
2. **Feature Importance** rankings
3. **ROC & Precision-Recall Curves**
4. **Confusion Matrix** with business cost overlay
5. **Customer Risk Segmentation** 
6. **Revenue Impact Waterfall** chart

---

## 🔍 Key Insights

### Churn Drivers
1. **Contract Type**: Month-to-month contracts have 3x higher churn
2. **Tenure**: 70% of churn occurs within first 12 months
3. **Payment Method**: Electronic check users churn 2x more
4. **Service Usage**: Low engagement score is strongest predictor
5. **Monthly Charges**: Customers paying >$80/month churn 40% more

### Actionable Recommendations
- **Onboarding Program**: Focus on first-year customers
- **Contract Incentives**: Offer discounts for annual contracts
- **Payment Migration**: Encourage auto-pay enrollment
- **Engagement Campaigns**: Target low-activity users
- **Pricing Strategy**: Optimize pricing for high-value segments

---

## 🎯 Model Optimization

### Hyperparameter Tuning
```python
best_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'scale_pos_weight': 2.7,  # Handle class imbalance
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

**Key Insight**: `scale_pos_weight=2.7` significantly improved recall for churned customers (minority class).

---


## 👤 Author

**Sumit Gatade**

- 📧 Email: sumitgatade05@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/sumit-gatade-b30142295/
