# 🎓 Student Performance Prediction using Machine Learning

A complete mini-project where I predicted student final grades (regression) and classified pass/fail status (classification) using real-world data from the UCI repository.

---

## 📌 Project Goals

- Predict students' **final grade (G3)** using regression models
- Classify whether a student will **pass or fail**
- Apply complete ML workflow: EDA → Preprocessing → Modeling → Evaluation → Saving model

---

## 📊 Dataset

- **Source**: [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- `395` student records
- 30+ features including:
  - Academic scores (`G1`, `G2`, `failures`)
  - Demographics (`sex`, `address`, `Mjob`, `Fjob`)
  - Lifestyle & support (`studytime`, `romantic`, `internet`, `schoolsup`, `absences`)

---

## 🔍 Exploratory Data Analysis (EDA)

- Grade distribution analysis
- Correlation heatmaps
- Comparison of numerical & categorical features with final grade
- Visualizations: bar plots, scatter plots, histograms

---

## 🤖 ML Models Used

### Classification (Target = Pass/Fail)

| Model                   | Metric Used   | Best Accuracy |
|------------------------|---------------|----------------|
| Logistic Regression     | F1 Score      | ✅ Tested  
| Decision Tree Classifier | Accuracy     | ✅ Tuned  
| Random Forest Classifier | Accuracy     | ⭐ Best (Tuned)
| Support Vector Classifier | Accuracy     | ✅  
| K-Nearest Neighbors     | Accuracy     | ✅  

---

### Regression (Target = Final Grade `G3`)

| Model                   | Metric Used   | Best R² Score |
|------------------------|---------------|----------------|
| Linear Regression       | R² Score      | ✅  
| Decision Tree Regressor | R² Score      | ✅  
| Random Forest Regressor | R² Score      | ⭐ Best (0.91)
| Support Vector Regressor | R² Score     | ✅  
| K-Nearest Neighbors     | R² Score      | ✅  

---

## 🛠️ Tools & Libraries

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn` – Pipelines, GridSearchCV, models
- `joblib` – Save and load trained models

---

## 📈 Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Regression**: MAE, RMSE, R² Score
- **Visuals**: Residual plots, predicted vs actual, heatmaps

---

## 💾 Model Deployment

- Best model saved using `joblib`
- Real-time prediction function built
- Can be extended to a **Streamlit** web app

---

## 🚀 Future Work

- Build a Streamlit UI for user inputs
- Use SHAP for model interpretability
- Add API or web dashboard for predictions

---

## 🧠 What I Learned

This project helped me:
- Apply the end-to-end ML lifecycle on a real dataset
- Practice both regression and classification
- Use pipelines and cross-validation
- Improve feature engineering and visualization skills

---

## 📁 Project Structure

