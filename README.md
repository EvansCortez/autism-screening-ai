# 🧠 ASD-Predict: AI-Powered Autism Screening Tool

An end-to-end Machine Learning and Software Engineering project designed to provide accessible, early-stage behavioral screening for Autism Spectrum Disorder (ASD). This tool leverages the AQ-10 (Autism Quotient) framework to predict the likelihood of ASD traits using optimized classification models.

---

## 🚀 Overview
Early detection is critical for ASD support. This project transforms clinical behavioral data into a predictive system, featuring a **Scikit-Learn** backend and a **FastAPI** web interface.

* **Goal:** Provide a high-precision screening tool for initial assessment.
* **Tech Stack:** Python, Scikit-Learn, FastAPI, Pandas, Joblib.
* **Dataset:** AQ-10 Behavioral Screening Dataset (Adult/Child/Toddler).

---

## 🏗️ Project Architecture
The system is divided into three main layers:

1.  **Data Layer:** Preprocessing, handling missing values (`?`), and balancing classes using `RandomOverSampler`.
2.  **Logic Layer (ML):** A trained **Logistic Regression** and **SVC** ensemble with feature engineering (Age grouping and Summed Scores).
3.  **Service Layer (API):** A RESTful API that accepts user inputs and returns predictions in real-time.

---

## 📊 Machine Learning Pipeline
### 1. Feature Engineering
- **Score Summation:** Aggregated A1-A10 clinical scores to create a high-impact `sum_score` feature.
- **Log Transformation:** Applied to skewed 'Age' data to normalize distribution.
- **Handling Imbalance:** Used **SMOTE** or **RandomOverSampler** to fix the minority class bias.

### 2. Model Performance
| Model | Accuracy | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Logistic Regression | ~98% | 0.97 | 0.98 |
| SVC (RBF Kernel) | ~96% | 0.95 | 0.96 |
| XGBoost | ~95% | 0.94 | 0.94 |

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/autism-screening-ai.git](https://github.com/YOUR_USERNAME/autism-screening-ai.git)
   cd autism-screening-ai

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

   Run the API:

4. **Run the API:**
   ```bash
   uvicorn src.api.main:app --reload
