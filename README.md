# HealthSphere: 360° Health Insights

## 🚀 Overview

HealthSphere is an intelligent, ML-driven clinical decision support platform designed to predict diabetes risk and provide personalized health guidance. It combines time-series deep learning, risk classification, and GenAI recommendations to enable proactive and informed health management.

📽️ **[Project Demo](https://youtu.be/eKJrDBX-v30)**

---

## 🔍 Features

* **📈 Glucose Forecasting**: Predicts short-term glucose trends using LSTM with attention mechanism on time-series data (MAE: **0.020**, R² Score: **0.983**).

* **🧠 Diabetes Risk Classification**: Uses XGBoost with hyperparameter tuning, threshold optimization, and class imbalance handling (F1-score: **0.69**, Accuracy: **75.3%**).

* **📋 Personalized Health Recommendations**:  
  * Tailored lifestyle and diet advice based on BMI, glucose, and blood pressure.  
  * Preventive suggestions powered by Gemini API and fallback clinical heuristics.

* **🌐 Streamlit Web App**: Interactive and user-friendly interface for uploading data, viewing predictions, and receiving recommendations.

* **🔗 GenAI Integration**: Google Gemini API generates concise, actionable, and medically aligned suggestions in real-time.

---

## 🧠 Tech Stack

* **Machine Learning**: `XGBoost`, `LSTM with Attention`
* **Frameworks**: `TensorFlow`, `Keras`, `Scikit-learn`
* **Web Interface**: `Streamlit`
* **Data Engineering**: `Pandas`, `NumPy`, `MinMaxScaler`, `Feature Engineering`
* **GenAI Integration**: `Google Gemini API`

---

## ⚙️ Installation

```bash
# 1. Clone the repository
https://github.com/its308/HealthSphere.git

# 2. Navigate to the project folder
cd HealthSphere

# 3. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# 4. Install required packages
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
# Run the Streamlit app
streamlit run scripts/app.py
```

Then open `http://localhost:8501` in your browser.

**You can:**

* Enter current health parameters (BMI, Glucose, BP, etc.)
* Provide recent glucose readings
* Get:

  * **Risk Level** (Normal, Warning, Critical)
  * **Predicted Glucose**
  * **Actionable Recommendations**

---

## 🔂 File Structure

```
HealthSphere/
├── models/
│   ├── lstm_glucose_model.h5           # LSTM model for glucose prediction
│   └── xgboost_diabetes_model.pkl      # XGBoost model for risk classification
├── scripts/
│   ├── app.py                          # Streamlit app logic
│   └── recommender.py                  # Core model logic + GenAI integration
├── data/
│   └── sample_data.csv                 # Sample user health data
├── requirements.txt                    # Dependency list
└── README.md                           # Project documentation
```


---

## ✅ Model Integrity

- 🧪 **No Data Leakage**: Glucose forecasting LSTM model was **trained and tested on entirely disjoint patient sets** — ensuring true generalization to unseen individuals.

- 📊 **Evaluation on Unseen Patients**: The model’s performance (R²: **0.983**, MAE: **0.020**) is validated on completely **unseen patient data**, not just random time splits — making the results trustworthy.

- 📁 Refer to the code in [`prepare_time_series_data()`](scripts/lstm_model.py) where the split is done **patient-wise** and normalization is performed using **only training data**.

---

> *📈 Glucose prediction graph below is from a test user not seen during training — confirming model generalization.*



---

## 🔮 Future Enhancements

* 📱 Integration with wearables (e.g., Fitbit, Apple Health) for real-time glucose tracking.
* 📊 Visual dashboards for health progression and glucose trends.
* 🧠 Add mental health screening tools (e.g., PHQ-9, GAD-7).
* ☕️ Deploy to cloud platforms (e.g., AWS, GCP) with multi-user support.

---

## 🤝 Contributing

Pull requests are welcome!

1. Fork the repository
2. Create a branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push to your fork (`git push origin feature-name`)
5. Open a Pull Request 🚀

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

## 📌 Citation

This project: [https://github.com/its308/HealthSphere](https://github.com/its308/HealthSphere)
