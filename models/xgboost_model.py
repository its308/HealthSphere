import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle
import os

def preprocess_data(data_path):
    data = pd.read_csv(data_path)


    print("Dataset head:\n", data.head())

    print("\nMissing values:\n", data.isnull().sum())

    #replacing 0 with na where 0 is not possible
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        data[col]=data[col].replace(0,np.nan)

    # replacing nan with median # previous step done to clarify the values in which 0 not possible
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        data[col] = data[col].replace(np.nan,data[col].median())

    X = data.drop(columns='Outcome',axis=1)
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    return X_train_scaled,X_test_scaled,y_train,y_test,scalar,X.columns

def train_xgboost(data_path):
    X_train, X_test, y_train, y_test, scalar, feature_names=preprocess_data(data_path)
    xgb_model=xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    xgb_model.fit(X_train,y_train)

    preds=xgb_model.predict(X_test)
    y_pred_prob=xgb_model.predict_proba(X_test)[:,1]

    accuracy=accuracy_score(y_test,preds)
    print(f'Accuracy score={accuracy:.4f}')
    print('\n Classification Report :') # it consist of precision(false positive), recall,f1 score
    print(classification_report(y_test,preds))

    # confusion matrix
    cm=confusion_matrix(y_test,preds)
    plt.figure(figsize=(10,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig('../models/xgboost_confusion_matrix.png')
    plt.show()

    #Feature importance refers to how much each input feature (column) contributes to the predictions made by a machine-learning model.
    plt.figure(figsize=(10,6))
    xgb.plot_importance(xgb_model)
    plt.title('Feature importance')
    plt.savefig('../models/xgboost_feature_importance.png')
    plt.show()

    with open('../models/xgboost_diabetes_model.pkl', 'wb') as file:
        pickle.dump(xgb_model, file)

    with open('../models/xgboost_scaler.pkl', 'wb') as file:
        pickle.dump(scalar, file)

    return xgb_model,scalar

if __name__=="__main__":
    train_xgboost('/Users/itishachoudhary/AI_Projects/HealthGuardAI/data/diabetes.csv')





