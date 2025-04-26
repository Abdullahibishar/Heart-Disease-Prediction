
 💓 Heart Disease Prediction App

This project is a web application that predicts the risk of heart disease based on a patient's medical data.  
It uses a trained **XGBoost** machine learning model and provides visual explanations using **SHAP**.

---

 📋 Project Overview

- Users input patient details such as age, cholesterol level, blood pressure, etc.
- The trained model predicts whether the patient has a high or low risk of heart disease.
- SHAP plots explain how each feature contributed to the prediction.

🧠 How to Understand the SHAP Chart

- 🔴 **Red bars** push the prediction toward higher heart disease risk.
- 🔵 **Blue bars** push the prediction toward lower heart disease risk.
- The longer the bar, the stronger its influence on the model's decision.
- Top features are the most important in an individual prediction.

---

 🚀 How to Run the App Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/Abdullahibishar/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

 🛠 Project Files

| File                      | Description                           |
|----------------------------|---------------------------------------|
| `app.py`                   | Streamlit app interface               |
| `best_xgb_model.pkl`        | Trained XGBoost machine learning model |
| `scaler.pkl`                | Scaler for feature normalization     |
| `heart_disease_dataset.csv` | Dataset used for model training       |
| `capstone.ipynb`            | Notebook showing model building      |
| `requirements.txt`          | Required Python libraries            |
| `README.md`                 | Project documentation                |

---

 📚 About XGBoost

**XGBoost (Extreme Gradient Boosting)** is a fast and powerful machine learning algorithm that builds models step-by-step.  
It improves its predictions by learning from previous mistakes and is known for its high performance in real-world problems.

---

❤️ Built With

- Streamlit
- XGBoost
- SHAP
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
```

---

✅ This version will now properly render:
- Correct spacing
- Bullet points neat
- SHAP section readable
- Table properly aligned and spaced
- Headings consistent
- Professional appearance 📋



