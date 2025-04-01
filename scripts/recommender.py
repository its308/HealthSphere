import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model
import google.generativeai as genai
import joblib
from datetime import datetime
class HealthRecommender:
    def __init__(self,lstm_model_Path,xgb_model_Path,lstm_scaler_Path,xgb_scaler_Path):
        from tensorflow.keras.losses import MeanSquaredError

        self.lstm_model = load_model(lstm_model_Path, custom_objects={"mse": MeanSquaredError()})

        with open(xgb_model_Path,'rb') as f:
            self.xgb_model=pickle.load(f)

        # with open(lstm_scaler_Path,'rb') as f:
        #     self.global_scaler=joblib.load(f)

        self.scalers = joblib.load(lstm_scaler_Path)

        with open(xgb_scaler_Path,'rb') as f:
            self.xgb_scaler=pickle.load(f)

    def _create_features(self, glucose_values):
        """Create features for real-time predictions"""
        df = pd.DataFrame({'glucose': glucose_values})

        # Add timestamp
        df['timestamp'] = range(len(df))

        # Create datetime temporarily for feature extraction
        now = datetime.now()
        temp_datetime = pd.date_range(end=now, periods=len(df), freq='1min')

        # Extract time features
        df['hour'] = temp_datetime.hour
        df['day_of_week'] = temp_datetime.dayofweek

        # Add cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Drop original time columns (ONLY DO THIS ONCE)
        df = df.drop(columns=['hour', 'day_of_week'])

        # Add lag features
        for lag in [1, 2, 3, 5]:
            df[f'glucose_lag_{lag}'] = df['glucose'].shift(lag)

        # Add rolling features
        for window in [3, 5, 7]:
            df[f'glucose_rolling_mean_{window}'] = df['glucose'].rolling(window).mean()
            df[f'glucose_rolling_std_{window}'] = df['glucose'].rolling(window).std()

        # Add rate of change
        df['glucose_diff'] = df['glucose'].diff()

        # Fill missing values
        df.fillna(0, inplace=True)

        return df

    def predict_glucose(self, glucose_history):
        if len(glucose_history) < 20:
            raise ValueError("Minimum 20 readings required")

        try:
            df = self._create_features(glucose_history)

            # Get feature order from training
            feature_order = self.scalers['feature_scaler'].feature_names_in_

            # Add missing features with 0 value
            for col in feature_order:
                if col not in df.columns:
                    df[col] = 0

            # Reorder columns to match training data
            df = df[['glucose'] + feature_order.tolist()]

            # Scaling and prediction
            df['glucose'] = self.scalers['global_scaler'].transform(df[['glucose']])
            df[feature_order] = self.scalers['feature_scaler'].transform(df[feature_order])

            sequence = df[-20:].values.reshape(1, 20, -1)
            pred = self.lstm_model.predict(sequence)[0][0]

            # Inverse transform and return
            return self.scalers['global_scaler'].inverse_transform([[pred]])[0][0]

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return np.mean(glucose_history[-5:])  # Fallback to moving average

    def classify_risk(self, health_data):
        if isinstance(health_data, dict):
            health_data = pd.DataFrame([health_data])

        try:
            # Get standard diabetes features
            standard_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

            # Ensure all standard features exist in health_data
            for feature in standard_features:
                if feature not in health_data.columns:
                    health_data[feature] = 0

            # Use only the standard features for prediction
            input_data = health_data[standard_features]

            # Scale and predict
            scaled_input = self.xgb_scaler.transform(input_data)
            risk_prob = self.xgb_model.predict_proba(scaled_input)[:, 1][0]
            risk_class = self.xgb_model.predict(scaled_input)[0]

            # Determine risk level
            if risk_prob < 0.3:
                risk_level = 'Normal'
            elif risk_prob < 0.7:
                risk_level = 'Warning'
            else:
                risk_level = 'Critical'

            return {
                'risk_class': int(risk_class),
                'risk_prob': float(risk_prob),
                'risk_level': risk_level
            }

        except Exception as e:
            print(f"Risk classification error: {e}")
            # Fallback to a default medium risk
            return {
                'risk_class': 0,
                'risk_prob': 0.5,
                'risk_level': 'Warning'
            }

    def generate_recommendations(self, health_data, glucose_history=None):
        recommendations = []
        predicted_glucose = None

        # Convert thresholds from mg/dL to mmol/L for manual rules
        GLUCOSE_THRESHOLD_WARNING_MGDL = 120  # mg/dL
        GLUCOSE_THRESHOLD_CRITICAL_MGDL = 140  # mg/dL

        GLUCOSE_THRESHOLD_WARNING_MMOL = 6.7  # mmol/L (120 mg/dL)
        GLUCOSE_THRESHOLD_CRITICAL_MMOL = 7.8  # mmol/L (140 mg/dL)

        # Convert health_data glucose to mmol/L for recommendation thresholds
        glucose_mgdl = health_data.get('Glucose', 0)
        glucose_mmol = glucose_mgdl * 0.0555  # Convert to mmol/L for thresholds

        if glucose_history is not None:
            try:
                predicted_glucose = self.predict_glucose(glucose_history)
            except Exception as e:
                print(f"Prediction error: {e}")

        # Get risk assessment using original health_data with mg/dL
        risk_info = self.classify_risk(health_data)

        try:
            genai.configure(api_key="AIzaSyCTw17MjvPUGZ6_1gIPLarQCcR8cB_A9O4")

            # System instruction with expanded guidelines
            system_instruction = {
                "role": "system",
                "content": """You are a clinical decision support system. Follow these rules strictly:
                1. Prioritize these recommendations based on user metrics:
                   - If BMI >25: "Consider a moderate exercise routine to manage weight"
                   - If Glucose >120 mg/dL: "Monitor carbohydrate intake and check glucose levels"
                   - If BloodPressure >90: "Reduce sodium intake and practice stress management"
                2. Format recommendations as concise bullet points
                3. Do NOT add explanations or disclaimers
                4. Use only medically validated advice
                5. Include recommendations about sleep, stress management, and hydration"""
            }

            # Safety settings
            safety_settings = {
                "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
                "SEXUALLY_EXPLICIT": "BLOCK_LOW_AND_ABOVE",
                "DANGEROUS": "BLOCK_NONE"
            }

            prompt = f"""
            Given these health metrics:
            - BMI: {health_data.get('BMI', 0)}
            - Glucose: {glucose_mgdl} mg/dL
            - Blood Pressure: {health_data.get('BloodPressure', 0)} mmHg
            - Age: {health_data.get('Age', 0)}
            - Predicted Glucose: {predicted_glucose if predicted_glucose is not None else 'Not available'} mmol/L

            Generate 3-5 concise, actionable health recommendations that:
            - Address the specific health metrics above
            - Include lifestyle, diet, and preventive measures
            - Use simple language
            - Are medically accurate
            - Do not include explanations or disclaimers
            """

            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(
                contents=[system_instruction, {"role": "user", "content": prompt}],
                safety_settings=safety_settings
            )

            if response and response.text:
                # Post-processing cleanup
                raw_recommendations = response.text.strip().split('\n')
                clean_recommendations = [
                    rec for rec in raw_recommendations
                    if rec.strip() and not any(keyword in rec.lower() for keyword in
                                               ["note:", "disclaimer", "however", "remember"])
                ]
                recommendations.extend(clean_recommendations)

        except Exception as e:
            print(f'Gemini API call failed: {e}, using manual recommendations')

        # Expanded manual fallback with unit-aware thresholds
        # Handle Normal Risk Level
        if risk_info['risk_level'] == 'Normal':
            recommendations.append('Your health indicators are within normal range.')
            recommendations.append('Continue with your current healthy lifestyle.')
            recommendations.append('Maintain a balanced diet with plenty of fruits and vegetables.')
            recommendations.append('Get at least 150 minutes of moderate exercise per week.')

        # Handle Warning Risk Level
        elif risk_info['risk_level'] == 'Warning':
            # recommendations.append('Some of your health indicators suggest increased risk.')

            # BMI-related recommendations
            if health_data.get('BMI', 0) > 25:
                recommendations.append("Consider a moderate exercise routine to help manage your weight.")
                recommendations.append("Incorporate strength training exercises twice a week to build muscle mass.")

            # Glucose-related recommendations - using mg/dL threshold for XGBoost input
            if glucose_mgdl > GLUCOSE_THRESHOLD_WARNING_MGDL:
                recommendations.append("Monitor your carbohydrate intake and consider more frequent glucose checks.")
                recommendations.append(
                    "Focus on whole grains, lean proteins, and healthy fats to stabilize blood sugar levels.")

            # Blood pressure recommendations
            if health_data.get('BloodPressure', 0) > 85:
                recommendations.append("Limit sodium intake to help manage your blood pressure.")

            # General recommendations
            recommendations.append("Aim for 7-9 hours of quality sleep each night to improve metabolic health.")
            recommendations.append("Drink plenty of water throughout the day to stay hydrated.")

        # Handle Critical Risk Level
        else:  # Critical case
            recommendations.append(
                "Your health indicators suggest high risk. Consider consulting a healthcare professional.")

            # Glucose-related recommendations - using mg/dL threshold for XGBoost input
            if glucose_mgdl > GLUCOSE_THRESHOLD_CRITICAL_MGDL:
                recommendations.append(
                    "Your glucose levels are elevated. Limit sugar intake and increase physical activity.")
                recommendations.append("Consider working with a registered dietitian for personalized meal planning.")

            if health_data.get("BloodPressure", 0) > 90:
                recommendations.append(
                    "Your blood pressure is elevated. Consider reducing sodium intake and stress management techniques.")

            if health_data.get("BMI", 0) > 30:
                recommendations.append(
                    "Your BMI indicates obesity. A structured weight management program may be beneficial.")

        # Handle predicted glucose values from LSTM (already in mmol/L)
        if predicted_glucose is not None:
            if predicted_glucose < 3.9:  # Hypoglycemia in mmol/L
                recommendations.append(
                    f"Your glucose is predicted to drop to {predicted_glucose:.1f} mmol/L. Consider having a small snack soon.")
            elif predicted_glucose > 10.0:  # Hyperglycemia in mmol/L
                recommendations.append(
                    f"Your glucose is predicted to rise to {predicted_glucose:.1f} mmol/L. Consider light physical activity to help manage this increase.")

        return {
            'risk_assessment': risk_info,
            "predicted_glucose": predicted_glucose,
            "recommendations": recommendations
        }






