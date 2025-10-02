
# Predictive Analytics for Early Diabetes Diagnosis

**Goal:** Build and compare ML models to predict diabetes from clinical indicators (Pima Indians Dataset).  
**Highlights:** End-to-end pipeline (cleaning → EDA → baselines → tuning → comparison) with reproducible code.

## Dataset
Pima Indians Diabetes (UCI/Kaggle).  
**Target:** `Outcome` (1=diabetes, 0=non-diabetes)  
**Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.  
**Note:** Zeros in some medical features are invalid → treated as missing and imputed (median).

## Methods
- Train/test split (80/20, stratified)
- Scaling for LR/KNN; trees used raw features
- Models: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Improvements: class weights (`balanced`) + GridSearchCV for KNN/RF/XGB

## Results
| Model                        | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------------------------------|----------|-----------|--------|----------|---------|
| **XGBoost (Tuned)**          | 0.734    | 0.644     | 0.537  | 0.586    | **0.818** |
| Balanced Logistic Regression | 0.734    | 0.603     | **0.704** | **0.650** | 0.813 |
| Random Forest (Tuned)        | 0.740    | 0.667     | 0.519  | 0.583    | 0.810 |
| KNN (Tuned)                  | **0.760**| **0.698** | 0.556  | 0.619    | 0.807 |

**Best by ROC-AUC:** XGBoost (0.818).  
**Best Recall:** Balanced Logistic Regression (0.704).  
**Best Accuracy:** KNN (0.760).  

## Key Insights
- **Glucose** is the strongest predictor of diabetes, followed by **BMI** and **Age**.  
- Models show different trade-offs:  
  - XGBoost gives the best ROC-AUC (overall discrimination).  
  - Balanced Logistic Regression gives the highest Recall (catching more diabetic cases).  
  - KNN slightly outperforms others in raw Accuracy.  
- In a healthcare context, Recall is often more critical than Accuracy.

## Limitations & Next Steps
- Dataset is limited to **Pima Indian women aged ≥21**, so generalizability is restricted.  
- Zero values for several features were imputed with medians.  
- Future work could include SHAP/LIME for explainability, threshold tuning for Recall vs. Precision trade-off, and testing on more diverse datasets.

## Reproducibility
```bash
pip install -r requirements.txt
# open notebook and run all; or load models/best_model_xgboost.pkl for inference
