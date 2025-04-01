import streamlit as st
import numpy as np
import pandas as pd
from recommender import HealthRecommender  # Assuming your class is in this module

# Initialize the recommender system
recommender = HealthRecommender(
    'models/lstm_glucose_model.h5',
    'models/xgboost_diabetes_model.pkl',
    'models/glucose_scaler.pkl',
    'models/xgboost_scaler.pkl'
)

# Streamlit UI Configuration
st.set_page_config(page_title="HealthSphere", layout="wide")


def main():
    st.title("HealthSphere : 360° Health Insights and Guide")

    with st.sidebar:
        st.header("Patient Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)

        # Keep mg/dL input for XGBoost compatibility
        glucose_mgdl = st.number_input("Glucose (mg/dL)", min_value=50, max_value=300, value=135,
                                       help="Normal fasting range: 70-99 mg/dL")

        bp = st.number_input("Blood Pressure (mmHg)", min_value=30, max_value=200, value=85)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=30)
        insulin = st.number_input("Insulin (Î¼U/mL)", min_value=0, max_value=300, value=150)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=27.5,
                             help="Normal range: 18.5-24.9")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Glucose History Input")
        glucose_history = st.text_area(
            "Enter last 20 glucose readings (comma-separated)(mmol/L):",
            value="6.8,7.2,7.5,8.1,8.9,9.2,8.7,8.3,7.8,7.2,6.9,7.1,7.3,7.8,8.2,8.6,8.3,7.9,7.5,7.1",
            height=150
        )

    with col2:
        st.header("Health Recommendations")

        if st.button("Generate Comprehensive Analysis"):
            try:
                # Prepare health data dictionary with mg/dL for XGBoost
                health_data = {
                    'Pregnancies': pregnancies,
                    'Glucose': glucose_mgdl,  # Keep as mg/dL for XGBoost
                    'BloodPressure': bp,
                    'SkinThickness': skin_thickness,
                    'Insulin': insulin,
                    'BMI': bmi,
                    'DiabetesPedigreeFunction': dpf,
                    'Age': age
                }

                # Convert glucose history to list
                try:
                    glucose_list = list(map(float, glucose_history.split(',')))
                except ValueError:
                    st.error("Invalid glucose history format! Please enter comma-separated numbers.")
                    return

                # Get recommendations
                results = recommender.generate_recommendations(health_data, glucose_list)

                # Display enhanced risk assessment
                with st.container():
                    st.subheader("Risk Assessment")

                    # Enhanced risk level display with context
                    risk_context = {
                        'Normal': "Your metrics are within normal ranges.",
                        'Warning': "Based on your metrics, this indicates potential prediabetes.",
                        'Critical': "Your metrics suggest high risk for diabetes. Medical consultation recommended."
                    }

                    risk_level = results['risk_assessment']['risk_level']
                    st.write(f"**Risk Level:** {risk_level} - {risk_context.get(risk_level, '')}")

                    # Enhanced probability with percentage conversion
                    risk_prob = results['risk_assessment']['risk_prob']
                    st.write(f"**Risk Probability:** {risk_prob:.2f} - This means there is a {risk_prob * 100:.0f}% chance of developing diabetes.")

                    # Display glucose with both units
                    if results['predicted_glucose']:
                        glucose_mmol = results['predicted_glucose']
                        glucose_mgdl = glucose_mmol * 18  # Convert to mg/dL
                        st.write(f"**Predicted Glucose Level:** {glucose_mmol:.1f} mmol/L ({glucose_mgdl:.0f} mg/dL)")

                    st.divider()

                    # Group recommendations logically
                    lifestyle_recommendations = []
                    dietary_recommendations = []
                    preventive_measures = []

                    for rec in results['recommendations']:
                        if rec.strip():  # Skip empty recommendations
                            if any(term in rec.lower() for term in ["exercise", "activity", "stress", "sleep", "weight"]):
                                lifestyle_recommendations.append(rec)
                            elif any(term in rec.lower() for term in ["diet", "carbohydrate", "fiber", "food", "eat", "sugar"]):
                                dietary_recommendations.append(rec)
                            elif any(term in rec.lower() for term in ["monitor", "doctor", "screening", "consult", "check"]):
                                preventive_measures.append(rec)
                            # If it doesn't match any category, add to the one with fewest recommendations
                            else:
                                min_list = min([lifestyle_recommendations, dietary_recommendations, preventive_measures], key=len)
                                min_list.append(rec)

                    # Display grouped recommendations with context
                    if lifestyle_recommendations:
                        st.subheader("Lifestyle Recommendations")
                        for i, rec in enumerate(lifestyle_recommendations):
                            with st.expander(f"Lifestyle Recommendation {i + 1}"):
                                # Improved context logic
                                if "weight" in rec.lower() or "exercise" in rec.lower():
                                    context = f"This is recommended because your BMI is {bmi:.1f}, which is above the healthy threshold of 25."
                                elif "stress" in rec.lower():
                                    context = f"Stress management helps regulate blood sugar levels and blood pressure."
                                elif "sleep" in rec.lower():
                                    context = f"Quality sleep improves insulin sensitivity and helps regulate glucose levels."
                                else:
                                    context = "This recommendation supports overall metabolic health."

                                st.write(f"{rec}\n\n*{context}*")

                    if dietary_recommendations:
                        st.subheader("Dietary Changes")
                        for i, rec in enumerate(dietary_recommendations):
                            with st.expander(f"Dietary Change {i + 1}"):
                                # Add context based on recommendation content
                                if "carbohydrate" in rec.lower() or "glucose" in rec.lower():
                                    context = f"This is suggested because your glucose level is {glucose_mgdl:.0f} mg/dL, which is above the ideal range of 70-99 mg/dL."
                                else:
                                    context = "Dietary changes can significantly impact blood glucose levels and overall health."

                                st.write(f"{rec}\n\n*{context}*")

                    if preventive_measures:
                        st.subheader("Preventive Measures")
                        for i, rec in enumerate(preventive_measures):
                            with st.expander(f"Preventive Measure {i + 1}"):
                                context = f"Regular monitoring is important with a diabetes risk probability of {risk_prob:.2f}."
                                st.write(f"{rec}\n\n*{context}*")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()