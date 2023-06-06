import math
import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import save_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from pickle import dump,load
import warnings
warnings.simplefilter("ignore", UserWarning)
from ipynb.fs.full.MEMD_all import memd
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


%matplotlib inline

plt.style.use('bmh')

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


from datetime import time
cc1=time.fromisoformat("06:55:00")
cc12=time.fromisoformat("17:05:00")
df=df[(df['time']>cc1)&(df['time']<cc12)]


def select_and_update_column_names(df):
    expected_column_names = ['Air Temp', 'RH', 'Wind Speed', 'Solar Radiation', 'Passive Cloud Cover', 'Dew Pt. Temp.']
    updated_column_names = []

    for expected_name in expected_column_names:
        for actual_name in df.columns:
            if expected_name.lower() == actual_name.lower():
                updated_column_names.append(actual_name)
                break
        else:
            # Handle the case when no match is found (e.g., assume the expected name as is)
            updated_column_names.append(expected_name)

    df = df[updated_column_names]
    df.columns = expected_column_names
    return df



df2=df[['Air Temp','RH','Wind Speed','Solar Radiation','Passive Cloud Cover','Dew Pt. Temp.']]
df2 = select_and_update_column_names(df)
df2.index=df.date
duplicate = df2[df2.duplicated()]
df2=df2.drop_duplicates()


std=np.std(df2)
std_noise=std*0.3
#normalize features
noise = np.random.normal(0, std_noise, df2.shape)
train2=np.concatenate((df2,noise[:,0:3]),axis=1)
s=np.array(train2)
imfs=memd(s)

num_imfs=imfs.shape[0]

def create_imf_dataframe(imf_data, columns):
    imf_df = pd.DataFrame(imf_data)
    imf_df = imf_df.transpose()
    imf_df.rename(columns=columns, inplace=True)
    return imf_df

imf_data = [
    {'data': imfs[:, 0, :], 'columns': {i: f"t{i+1}" for i in range(num_imfs-1)}},
    {'data': imfs[:, 1, :], 'columns': {i: f"rh{i+1}" for i in range(num_imfs-1)}},
    {'data': imfs[:, 2, :], 'columns': {i: f"ws{i+1}" for i in range(num_imfs-1)}},
    {'data': imfs[:, 4, :], 'columns': {i: f"pcc{i+1}" for i in range(num_imfs-1)}},
    {'data': imfs[:, 5, :], 'columns': {i: f"dt{i+1}" for i in range(num_imfs-1)}},
    {'data': imfs[:, 3, :], 'columns': {i: f"sr{i+1}" for i in range(num_imfs-1)}},
]
imf_dfs = []
for imf in imf_data:
    imf_df = create_imf_dataframe(imf['data'], imf['columns'])
    imf_dfs.append(imf_df)

imf_all = pd.concat(imf_dfs, axis=1)

srs = {}
for i in range(1, num_imfs):
    col_names = [f"t{i}", f"rh{i}", f"ws{i}", f"pcc{i}", f"dt{i}", f"sr{i}"]
    srs[f"sr{i}"] = imf_all[col_names]

for i in range(1, num_imfs):
    globals()[f"sr{i}"] = srs[f"sr{i}"]

high1=sr1
high2=sr2
high3=pd.DataFrame()
for i in range(3, 6):
    high3['t'] = high3.get('t', 0) + globals()['sr{}'.format(i)]['t{}'.format(i)]
    high3['rh'] = high3.get('rh', 0) + globals()['sr{}'.format(i)]['rh{}'.format(i)]
    high3['ws'] = high3.get('ws', 0) + globals()['sr{}'.format(i)]['ws{}'.format(i)]
    high3['pcc'] = high3.get('pcc', 0) + globals()['sr{}'.format(i)]['pcc{}'.format(i)]
    high3['dt'] = high3.get('dt', 0) + globals()['sr{}'.format(i)]['dt{}'.format(i)]
    high3['sr'] = high3.get('sr', 0) + globals()['sr{}'.format(i)]['sr{}'.format(i)]


medium=pd.DataFrame()
for i in range(6, 11):
    medium['t'] = medium.get('t', 0) + globals()['sr{}'.format(i)]['t{}'.format(i)]
    medium['rh'] = medium.get('rh', 0) + globals()['sr{}'.format(i)]['rh{}'.format(i)]
    medium['ws'] = medium.get('ws', 0) + globals()['sr{}'.format(i)]['ws{}'.format(i)]
    medium['pcc'] = medium.get('pcc', 0) + globals()['sr{}'.format(i)]['pcc{}'.format(i)]
    medium['dt'] = medium.get('dt', 0) + globals()['sr{}'.format(i)]['dt{}'.format(i)]
    medium['sr'] = medium.get('sr', 0) + globals()['sr{}'.format(i)]['sr{}'.format(i)]


low = pd.DataFrame()

for i in range(10, num_imfs+1):
    low['t'] = low.get('t', 0) + globals()['sr{}'.format(i)]['t{}'.format(i)]
    low['rh'] = low.get('rh', 0) + globals()['sr{}'.format(i)]['rh{}'.format(i)]
    low['ws'] = low.get('ws', 0) + globals()['sr{}'.format(i)]['ws{}'.format(i)]
    low['pcc'] = low.get('pcc', 0) + globals()['sr{}'.format(i)]['pcc{}'.format(i)]
    low['dt'] = low.get('dt', 0) + globals()['sr{}'.format(i)]['dt{}'.format(i)]
    low['sr'] = low.get('sr', 0) + globals()['sr{}'.format(i)]['sr{}'.format(i)]


# Define a list of datasets to normalize
datasets = [high1, high2, high3, medium, low]

# Create an empty list to store the normalized datasets
normalized_datasets = []

# Define a list of variable names to use for the normalized datasets
variable_names = ['high1_norm', 'high2_norm', 'high3_norm', 'med_norm', 'low_norm']

# Loop through each dataset and normalize it
for i, dataset in enumerate(datasets):
    normalized_dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    normalized_datasets.append(normalized_dataset)
    # Assign the normalized dataset to a variable with the original variable name
    globals()[variable_names[i]] = normalized_dataset
    
def split_data(df2,train_size):
    train_days=math.floor(len(df2)*train_size/44)
    train_data,test_data=df2.iloc[0:train_days*44],df2.iloc[train_days*44:len(df2)]
    return train_data,test_data

def create_trainable_dataset(dataframe,n_inputs,n_outputs):
    X,Y=list(),list()
    for i in range(len(dataframe)-n_inputs-n_outputs+1):
        X.append(dataframe.iloc[i:(i+n_inputs), :])
        Y.append(dataframe.iloc[i + n_inputs:i + n_inputs + n_outputs, -1])
    return np.array(X), np.array(Y)

from attention import Attention
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adamax

import tensorflow as tf
from kerastuner.tuners import RandomSearch

def build_and_tune_model(train_data, n_input, n_output,project_name):
    # Split data into training and testing sets
    train_size = 0.80
    train_data, test_data = split_data(train_data, train_size)

    # Create trainable datasets
    X_train, Y_train = create_trainable_dataset(train_data, n_input, n_output)
    X_test, Y_test = create_trainable_dataset(test_data, n_input, n_output)

    # Define the hyperparameter search space
    def build_model(hp):
        model = Sequential()
        model_input = Input(shape=(X_train.shape[1], X_train.shape[2]))

        # Define hyperparameters
        num_layers = 5
        lstm_units = [16,16,16,16, 16]
        dropout_rate = 0.3

        # Add LSTM and Attention layers
        for i in range(num_layers):
            if i == 0:
                x = LSTM(lstm_units[i], return_sequences=True)(model_input)
            else:
                x = LSTM(lstm_units[i], return_sequences=True)(x)
            x = Dropout(0.2)(x)

        x = Attention(units=16)(x)
        x = Dense(44, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        model = Model(model_input, x)
        optimizer = Adamax(learning_rate=0.001)
        model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        return model



    # Perform hyperparameter tuning
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
    tuner = RandomSearch(build_model, objective='accuracy', max_trials=1, executions_per_trial=1,
                         project_name=project_name)
    tuner.search(X_train, Y_train, epochs=20, batch_size=32,
                 validation_data=(X_test, Y_test),callbacks=[stop_early])


    # Get the best model and return it
    best_model = tuner.get_best_models(num_models=1)[0]
    
    preds=best_model.predict(X_test)
    return preds,Y_test, best_model
n_input=132#3 days
n_output=44#1 

high1_predicted,high1_actual,best_model=build_and_tune_model(high1_norm, 
                                                             n_input, 
                                                             n_output,project_name='project_helios_1.1')
best_model.save('high1_location.h5')
high2_predicted,high2_actual,best_model=build_and_tune_model(high2_norm, n_input, n_output,project_name='project_helios_1.2')
best_model.save('high2_location.h5')
high3_predicted,high3_actual,best_model=build_and_tune_model(high3_norm, n_input, n_output,project_name='project_helios_1.3')
best_model.save('high3_location.h5')
med_predicted,med_actual,best_model=build_and_tune_model(med_norm, n_input, n_output,project_name='project_helios_1.4')
best_model.save('med_location.h5')
low_predicted,low_actual,best_model=build_and_tune_model(low_norm, n_input, n_output,project_name='project_helios_1.5')
best_model.save('low_location.h5')

Y_results=pd.DataFrame()
Y_results['Actual_imf1']=high1_actual[0]
Y_results['imf1']=high1_predicted[0]
Y_results['Actual_imf2']=high2_actual[0]
Y_results['imf2']=high2_predicted[0]
Y_results['Actual_imf3']=high3_actual[0]
Y_results['imf3']=high3_predicted[0]
Y_results['Actual_imf4']=med_actual[0]
Y_results['imf4']=med_predicted[0]
Y_results['Actual_imf5']=low_actual[0]
Y_results['imf5']=low_predicted[0]

Y_results['Actual_imf1'] = (Y_results['Actual_imf1'] * (high1['sr1'].max() - high1['sr1'].min())) + high1['sr1'].min()
Y_results['imf1'] = (Y_results['imf1'] * (high1['sr1'].max() - high1['sr1'].min())) + high1['sr1'].min()
Y_results['Actual_imf2'] = (Y_results['Actual_imf2'] * (high2['sr2'].max() - high2['sr2'].min())) + high2['sr2'].min()
Y_results['imf2'] = (Y_results['imf2'] * (high2['sr2'].max() - high2['sr2'].min())) + high2['sr2'].min()
Y_results['Actual_imf3'] = (Y_results['Actual_imf3'] * (high3['sr'].max() - high3['sr'].min())) + high3['sr'].min()
Y_results['imf3'] = (Y_results['imf3'] * (high3['sr'].max() - high3['sr'].min())) + high3['sr'].min()
Y_results['Actual_imf4'] = (Y_results['Actual_imf4'] * (medium['sr'].max() - medium['sr'].min())) + medium['sr'].min()
Y_results['imf4'] = (Y_results['imf4'] * (medium['sr'].max() - medium['sr'].min())) + medium['sr'].min()
Y_results['Actual_imf5'] = (Y_results['Actual_imf5'] * (low['sr'].max() - low['sr'].min())) + low['sr'].min()
Y_results['imf5'] = (Y_results['imf5'] * (low['sr'].max() - low['sr'].min())) + low['sr'].min()

predicted_=Y_results['imf1']+Y_results['imf2']+Y_results['imf3']+Y_results['imf4']Y_results['imf5']
actual_=Y_results['Actual_imf1']+Y_results['Actual_imf2']+Y_results['Actual_imf3']+Y_results['Actual_imf4']+Y_results['Actual_imf5']

final_results=pd.DataFrame()
final_results['Preds']=predicted_
final_results['Actual']=actual_


