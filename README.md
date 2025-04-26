```markdown
# 💓 Heart Disease Prediction App

This project is a web application that predicts the risk of heart disease based on a patient's medical data.  
It uses a trained **XGBoost** machine learning model and provides visual explanations using **SHAP**.



## 📋 Project Overview

- Users input patient details such as age, cholesterol level, blood pressure, etc.
- The trained model predicts whether the patient has a high or low risk of heart disease.
- SHAP plots explain how each feature contributed to the prediction.

---

## 🚀 How to Run the App Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/Abdullahibishar/heart-disease-prediction-app.git
   cd heart-disease-prediction-app
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```



## 🛠 Project Files

| File | Description |
|-----|-------------|
| `app.py` | Streamlit app interface |
| `best_xgb_model.pkl` | Trained XGBoost model |
| `scaler.pkl` | Scaler for input features |
| `heart_disease_dataset.csv` | Dataset used for model training |
| `capstone.ipynb` | Notebook showing model training process |
| `requirements.txt` | List of necessary Python libraries |
| `README.md` | This project documentation |



## 📚 About XGBoost

**XGBoost (Extreme Gradient Boosting)** is a highly efficient machine learning algorithm that builds models step-by-step, learning from previous errors to improve performance.  
It is widely used in real-world problems because it is both **fast** and **accurate**.







 Built with ❤️ using Streamlit, XGBoost, and SHAP for Explainable AI.
```



