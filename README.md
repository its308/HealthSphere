
---

## **HealthSphere: 360° Health Insights**

### **Overview**
HealthSphere is a machine learning-powered clinical decision support system designed to predict diabetes risk and provide personalized health recommendations. It leverages advanced AI models and healthcare guidelines to assist users in managing their metabolic health effectively.

---

Project Demo:https://youtu.be/eKJrDBX-v30
---

### **Features**
- **Risk Assessment**: Predicts diabetes risk levels (Normal, Warning, Critical) using XGBoost.
- **Glucose Prediction**: Forecasts glucose trends using LSTM based on historical readings.
- **Personalized Recommendations**:
  - Lifestyle advice tailored to BMI, glucose levels, and blood pressure.
  - Dietary changes based on glycemic control needs.
  - Preventive measures for long-term health management.
- **Interactive UI**: Built with Streamlit for seamless user experience.
- **Google Gemini API Integration**: AI-driven insights supplement manual recommendations.

---

### **Technologies Used**
- **Machine Learning**:
  - XGBoost for risk classification
  - LSTM for time-series glucose prediction
- **Frameworks**:
  - TensorFlow/Keras
  - Scikit-learn
- **UI/UX**:
  - Streamlit for interactive web application
- **API Integration**:
  - Google Gemini API for AI-enhanced recommendations
- **Data Processing**:
  - Pandas, NumPy

---

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/its308/HealthSphere.git
   ```
2. Navigate to the project directory:
   ```bash
   cd HealthSphere
   ```
3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### **Usage**
1. Run the Streamlit app locally:
   ```bash
   streamlit run scripts/app.py
   ```
2. Open the app in your browser at `http://localhost:8501`.

3. Input your health metrics (e.g., BMI, glucose levels, blood pressure) and historical glucose readings to get:
   - Risk assessment
   - Predicted glucose trends
   - Personalized recommendations

---

### **File Structure**
```
HealthSphere/
├── models/
│   ├── lstm_glucose_model.h5           # Pre-trained LSTM model for glucose prediction
│   └── xgboost_diabetes_model.pkl      # Pre-trained XGBoost model for risk classification
├── scripts/
│   ├── app.py                          # Main Streamlit application file
│   └── recommender.py                  # Core recommendation logic and ML model integration
├── data/
│   └── sample_data.csv                 # Example input data (optional)
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation (this file)
```



### **Screenshots**
#### Risk Assessment Section:
![Risk Assessment](https://via.placeholder.com/800x400?text=Risk+ Recommendations Section:
![Recommendations](https://via.placeholder.com/800x400?text=Recommendations+Future Enhancements**
1. Integrate wearable device APIs (e.g., Fitbit, Apple Health) for real-time data.
2. Add visualizations for glucose trends and risk progression.
3. Expand recommendations to include mental health assessments (e.g., PHQ-9/GAD-7).
4. Deploy on scalable platforms like AWS or Google Cloud.

---

### **Contributing**
Contributions are welcome!  
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

### **License**
This project is licensed under the MIT License.

---


---

This README is structured to highlight your project's strengths while making it easy for others to understand and use. Let me know if you'd like further customization! 🚀

Citations:
[1] https://github.com/its308/HealthSphere

---
Answer from Perplexity: pplx.ai/share
