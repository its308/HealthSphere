import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense,Input,Attention,Concatenate
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

import re
from datetime import datetime

import os
# print(f"Path exists: {os.path.exists('../data/diabetes_subset_sensor_data')}")
# test_file = "../data/diabetes_subset_sensor_data/001/sensor_data/2014_10_01-10_09_39/2014_10_01-10_09_39_BB.csv"
# df = pd.read_csv(test_file)
# print(df.head())
# def prepare_time_series_data(base_path, sequence_length=20):
#     all_glucose_data=[]
#     for root, dirs, files in os.walk(base_path):
#         # print(f"Directory: {root}")
#         # print(f"Files: {files}")
#         for file in files:
#             if file.endswith('_BB.csv'):
#                 file_path = os.path.join(root, file)
#                 participant_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))# will return 001 etc as per patient no.
#                 '''os.path.dirname(file_path)- diabetes_subset_sensor_data/001/sensor_data
#                     os.path.dirname() - again diabetes_subset_sensor_data/001
#                     os.path.basename() - on this result	001'''
#                 try:
#                     df = pd.read_csv(file_path)
#                     # print(df.head())
#                     # print(df.columns)
#
#                     # print(f"File read successfully: {file_path}, shape: {df.shape}")
#                     # df['glucose']=pd.to_numeric(df['glucose'],errors='coerce')
#                     if 'BtoB' in df.columns:
#                         df = df.rename(columns={'BtoB': 'glucose'})
#                     # df['timestamp']=pd.to_numeric(df['timestamp'],errors='coerce')
#                     df['timestamp'] = range(len(df))
#                     df['glucose'] = pd.to_numeric(df['glucose'], errors='coerce')
#                     # df['datetime']=pd.date_range(start=file_datetime,periods=len(df),freq='1min')
#                     df = df.dropna(subset=['glucose'])
#
#
#                     if len(df)>0:
#                         df['participant'] = participant_id
#                         file_datetime=pd.to_datetime(os.path.basename(file_path).split('_BB.csv')[0],format='%Y_%m_%d-%H_%M_%S')
#                         # print(f"File: {file_path}, Extracted datetime: {file_datetime}")
#
#                         df['datetime'] = pd.date_range(start=file_datetime, periods=len(df), freq='1min')
#
#                         # OUTLIER REMOVAL - Add this section
#                         # Method 1: IQR-based outlier removal
#                         Q1 = df['glucose'].quantile(0.25)
#                         Q3 = df['glucose'].quantile(0.75)
#                         IQR = Q3 - Q1
#                         lower_bound = Q1 - 1.5 * IQR
#                         upper_bound = Q3 + 1.5 * IQR
#
#                         # Filter out extreme values
#                         df = df[(df['glucose'] >= lower_bound) & (df['glucose'] <= upper_bound)]
#
#                         # Add time features
#                         df['hour'] = df['datetime'].dt.hour
#                         df['day_of_week'] = df['datetime'].dt.dayofweek
#                         df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
#
#                         # Add lag features
#                         for lag in [1, 2, 3, 5]:
#                             df[f'glucose_lag_{lag}'] = df['glucose'].shift(lag)
#
#                         # Add rolling window features
#                         for window in [3, 5, 7]:
#                             df[f'glucose_rolling_mean_{window}'] = df['glucose'].rolling(window=window).mean()
#                             df[f'glucose_rolling_std_{window}'] = df['glucose'].rolling(window=window).std()
#
#                         # Add rate of change features
#                         df['glucose_diff'] = df['glucose'].diff()
#
#                         # print("BEFORE DROP--Unique glucose values:", df['glucose'].unique())
#                         # *************************Drop rows with NaN from feature creation-problem since many rows will get dropped
#                         # df = df.dropna()
#                         # print("AFTER DROP--Unique glucose values:", df['glucose'].unique())
#                         df.fillna(0, inplace=True)  # Replaces NaNs with 0
#
#                         if len(df) > 0:
#                             all_glucose_data.append(df)
#
#                 except Exception as e:
#                     print(f"Error processing file {file_path}: {e}")
#                     continue
#
#     if not all_glucose_data:
#
#         raise ValueError("No valid glucose data found in the provided path")
#
#
#     # now, combining the data of all participants
#     combined_glucose_data=pd.concat(all_glucose_data,ignore_index=True)
#     # Debug: Print columns of combined DataFrame
#     print(f"Combined DataFrame columns: {combined_glucose_data.columns.tolist()}")
#     combined_glucose_data = combined_glucose_data.drop(columns=['Time'], errors='ignore')
#     combined_glucose_data = combined_glucose_data.sort_values(['participant', 'datetime'])
#
#     # scaler=MinMaxScaler(feature_range=(0,1))
#     # combined_glucose_data['glucose_normalised']=scaler.fit_transform(combined_glucose_data[['glucose']]).flatten()
#     #
#     #************ TRIED PATIENT SPECIFIC NORMALISATION- CAUSED PROBLEM FOR GENERALISATION
#     # patient_normalised_dfs=[]
#     # patient_scalers={}
#     # for participant in combined_glucose_data['participant'].unique():
#     #     participant_data=combined_glucose_data[combined_glucose_data['participant']==participant].copy() # it will fetch all the data for that particular participant
#     #     scaler=MinMaxScaler(feature_range=(0,1))
#     #     participant_data['glucose_normalised']=scaler.fit_transform(participant_data[['glucose']]).flatten()
#     #     patient_normalised_dfs.append(participant_data)
#     #     patient_scalers[participant] = scaler
#     # combined_glucose_data=pd.concat(patient_normalised_dfs,ignore_index=True)
#
#     global_scaler = MinMaxScaler(feature_range=(0, 1))
#     combined_glucose_data['glucose_normalised'] = global_scaler.fit_transform(
#         combined_glucose_data[['glucose']]
#     )
#
#     # normalization of some other numerical feature columns
#
#     feature_scaler = MinMaxScaler(feature_range=(0, 1))
#     numerical_cols=[col for col in combined_glucose_data.columns if col not in ['participant', 'datetime', 'glucose', 'glucose_normalised']]
#     if numerical_cols:
#         combined_glucose_data[numerical_cols]=feature_scaler.fit_transform(combined_glucose_data[numerical_cols])
#
#     X,y=[],[]
#     input_features=['glucose_normalised']+numerical_cols
#
#
#     # for col in combined_glucose_data.columns:
#     #     if col.startswith('glucose_lag_') or col.startswith('glucose_rolling_') or col in ['glucose_diff', 'glucose_pct_change']:
#     #         input_features.append(col)
#
#     for participant in combined_glucose_data['participant'].unique():
#         participant_data=combined_glucose_data[combined_glucose_data['participant']==participant] # it will fetch all the data for that particular participant
#         # glucose_values=participant_data['glucose_normalised'].values
#         # for i in range(len(glucose_values) - sequence_length):
#         #     X.append(glucose_values[i:i + sequence_length])
#         #     y.append(glucose_values[i + sequence_length])
#         for i in range(len(participant_data)-sequence_length):
#             seq=participant_data.iloc[i:i+sequence_length]
#             target = participant_data.iloc[i+sequence_length]['glucose_normalised']
#             # seq_features = seq[input_features].values
#             X.append(seq[input_features].values)
#             y.append(target)
#
#     print(f'combined_glucose_data.columns:{combined_glucose_data.columns}')
#
#
#
#     X=np.array(X)#.reshape(-1, sequence_length, 1)
#     y=np.array(y)
#
#
#     print(f"X shape: {X.shape}")
#     print(f"y shape: {y.shape}")
#
#     train_size=int(len(X)*0.8)
#     X_train=X[:train_size]
#     X_test=X[train_size:]
#     y_train = y[:train_size]
#     y_test = y[train_size:]
#     # train_patient_ids=patient_ids[:train_size]
#     # test_patient_ids = patient_ids[train_size:]
#     scalers={'global_scaler': global_scaler, 'feature_scaler': feature_scaler}
#
#     return X_train, y_train, X_test, y_test,scalers
#
def prepare_time_series_data(base_path, sequence_length=20):
    all_glucose_data = []

    # Iterate through all glucose.csv files in subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'glucose.csv':
                file_path = os.path.join(root, file)
                participant_id = os.path.basename(os.path.dirname(file_path))

                try:
                    # Load and process data
                    df = pd.read_csv(file_path)
                    df = df.drop(columns=['comments'], errors='ignore')
                    required_cols = ['date', 'time', 'glucose']

                    if not all(col in df.columns for col in required_cols):
                        print(f"Missing columns in {file_path}. Skipping.")
                        continue

                    # Convert and clean data
                    df['glucose'] = pd.to_numeric(df['glucose'], errors='coerce')
                    df = df.dropna(subset=['glucose'])
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                    df['timestamp'] = range(len(df))
                    df['participant'] = participant_id

                    # Outlier removal
                    Q1 = df['glucose'].quantile(0.25)
                    Q3 = df['glucose'].quantile(0.75)
                    df = df[(df['glucose'] >= Q1 - 1.5 * (Q3 - Q1)) & (df['glucose'] <= Q3 + 1.5 * (Q3 - Q1))]

                    # Feature engineering
                    df['hour'] = df['datetime'].dt.hour
                    df['day_of_week'] = df['datetime'].dt.dayofweek
                    df['is_weekend'] = df['datetime'].dt.dayofweek >= 5

                    # Cyclical encoding for temporal features
                    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

                    # Cleanup
                    df = df.drop(columns=['date', 'time', 'datetime', 'hour', 'day_of_week'])

                    # Lag and rolling features
                    for lag in [1, 2, 3, 5]:
                        df[f'glucose_lag_{lag}'] = df['glucose'].shift(lag)
                    for window in [3, 5, 7]:
                        df[f'glucose_rolling_mean_{window}'] = df['glucose'].rolling(window).mean()
                        df[f'glucose_rolling_std_{window}'] = df['glucose'].rolling(window).std()
                    df['glucose_diff'] = df['glucose'].diff()



                    df.fillna(0, inplace=True)

                    if len(df) > sequence_length:
                        all_glucose_data.append(df)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    if not all_glucose_data:
        raise ValueError("No valid glucose data found")

    # Combine and normalize data
    combined_data = pd.concat(all_glucose_data, ignore_index=True)

    # Select only numerical columns for scaling
    numerical_cols = combined_data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('glucose')  # We'll normalize this separately

    # Normalization
    global_scaler = MinMaxScaler(feature_range=(0, 1))
    combined_data['glucose_normalised'] = global_scaler.fit_transform(combined_data[['glucose']])

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    combined_data[numerical_cols] = feature_scaler.fit_transform(combined_data[numerical_cols])

    # Create sequences
    X, y = [], []
    input_features = ['glucose_normalised'] + numerical_cols

    for participant in combined_data['participant'].unique():
        participant_data = combined_data[combined_data['participant'] == participant]

        for i in range(len(participant_data) - sequence_length):
            seq = participant_data.iloc[i:i + sequence_length]
            X.append(seq[input_features].values)
            y.append(participant_data.iloc[i + sequence_length]['glucose_normalised'])

    X = np.array(X)
    y = np.array(y)

    # Train/test split
    train_size = int(len(X) * 0.8)
    return (X[:train_size], y[:train_size],
            X[train_size:], y[train_size:],
            {'global_scaler': global_scaler, 'feature_scaler': feature_scaler})

    # glucose_vals=df['Glucose'].values.reshape(-1,1)
                # scalar=MinMaxScaler(feature_range=(0, 1))
                # glucose_scaled=scalar.fit_transform(glucose_vals)
                # n_input = 3  # no. of timesteps
                # n_features = 1  # which is glucse as of now
                # generator=TimeseriesGenerator(glucose_scaled,glucose_scaled,length=n_input,batch_size=1)
                # X,y=[],[]
                # for i in range(len(generator)):
                #     x_seq,y_seq=generator[i]
                #     X.append(x_seq)
                #     y.append(y_seq)
                #
                # X = np.array(X).reshape(-1, n_input, n_features)
                # y = np.array(y)
                #
                # train_size=int(len(X)*0.8)
                # X_train=X[:train_size]
                # y_train = y[:train_size]
                # X_test = X[train_size:]
                # y_test = y[train_size:]
                # return X_train,y_train,X_test,y_test,scalar

def build_LSTM(input_shape):
    model=Sequential()
    model.add(LSTM(64,activation='relu',input_shape=input_shape,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse',metrics=['mae'])
    return model


def build_LSTM_with_attention(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)

    # First LSTM layer
    lstm_out = LSTM(64, activation='relu', return_sequences=True)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)

    # Attention layer
    attention = Attention()([lstm_out, lstm_out])

    # Concatenate LSTM output and attention output
    concat = Concatenate()([lstm_out, attention])

    # Second LSTM layer
    lstm_out2 = LSTM(32, activation='relu')(concat)
    lstm_out2 = Dropout(0.2)(lstm_out2)

    # Dense layers
    dense1 = Dense(16, activation='relu')(lstm_out2)
    output = Dense(1)(dense1)

    # Create model
    model = Model(inputs=inputs, outputs=output)

    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model
def train_lstm_model(data_path):
    try:
        X_train, y_train, X_test, y_test, scalers=prepare_time_series_data(data_path)
        print(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
        print(f"Testing data shapes - X: {X_test.shape}, y: {y_test.shape}")

        model=build_LSTM_with_attention((X_train.shape[1],X_train.shape[2]))
        print(model.summary())
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )

        model.save('../models/lstm_glucose_model.h5')
        import joblib
        joblib.dump(scalers, '../models/glucose_scaler.pkl')

        loss,mae=model.evaluate(X_test,y_test,verbose=0)
        print(f'Test loss={loss}, Test MAE={mae}')

        preds=model.predict(X_test)
        global_scaler = scalers['global_scaler']

        preds_actual = global_scaler.inverse_transform(preds.reshape(-1, 1))
        y_test_actual=global_scaler.inverse_transform(y_test.reshape(-1,1))

        plt.figure(figsize=(12,6))
        plt.plot(y_test_actual[:100],label='Absolute Glucose')
        plt.plot(preds_actual[:100],label='Predicted Glucose')
        plt.title('LSTM Model: Glucose Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Glucose Level')
        plt.legend()
        plt.savefig('../models/lstm_predictions.png')
        plt.show()



    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


    # import pickle
    # with open('../models/glucose_scaler.pkl','wb') as f:
    #     pickle.dump(scalar,f)

    # import joblib
    # joblib.dump({'glucose_scaler': global_scaler, 'feature_scaler': feature_scaler},
    #             '../models/glucose_scaler.pkl')

    print('LSTM model saved successfully')

    return model,scalers

if __name__ == "__main__":
    train_lstm_model('../data/diabetes_subset_pictures-glucose-food-insulin')













