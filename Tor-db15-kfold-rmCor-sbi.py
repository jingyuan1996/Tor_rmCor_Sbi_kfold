import os 
import sys 
from pathlib import Path 
from datetime import datetime
from itertools import combinations 
import pandas as pd 
import numpy as np 

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from xgboost import *

from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, Activation, Flatten, TimeDistributed, RepeatVector
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Model 
from tensorflow.keras import Sequential
import tensorflow as tf 
#tf.config.threading.set_inter_op_parallelism_threads(16)
#tf.config.set_soft_device_placement(16)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.random.set_seed(1)
np.random.seed(101)

def creatCsvlog3(run):
    csvFilePWD = "./" + str(Path(__file__).stem) + str(run) +".csv"
    with open(csvFilePWD, 'w') as f:
        f.close()
    print("creat csv:csvFilePWD3")
    return csvFilePWD

def logwrite(write, filepwd):
    with open(filepwd, 'a') as f:
        f.write(write)
        f.write('\n')
        f.close()

def correlation(dataset, threshold):

    col_corr = set()

    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):
        
        for j in range(i):
        
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                
                #print(abs(corr_matrix.iloc[i, j]), corr_matrix.columns[i], corr_matrix.columns[j])
                colname = corr_matrix.columns[j]
                # and add it to our correlated set
                col_corr.add(colname)
                
    return col_corr


def remove_cor(train_data, test_data, threshold):

    corr_feature = correlation(train_data, threshold)

    run_train = train_data.copy()
    run_test = test_data.copy()

    run_train.drop(labels = corr_feature, axis = 1, inplace = True)
    run_test.drop(labels = corr_feature, axis = 1, inplace = True)

    return run_train, run_test

def load_data(filepwd):
    df = pd.read_csv(filepwd, header = [0])
    return df 

def CJ_kfold(use_kf_number, data, group_number):

    kf_data = KFold(n_splits = group_number, shuffle = False)
    kf_data.get_n_splits(data)

    kf_number = 1
    for train_index, test_index in kf_data.split(data):

        if kf_number == use_kf_number:
            break

        kf_number = kf_number+1

    return train_index, test_index

def CJ_random_kfold(use_kf_number, data, group_number):

    kf_data = KFold(n_splits = group_number, shuffle = True)
    kf_data.get_n_splits(data)

    kf_number = 1
    for train_index, test_index in kf_data.split(data):

        if kf_number == use_kf_number:
            break

        kf_number = kf_number+1

    return train_index, test_index


def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(30, return_sequences = True ,input_length=shape[1], input_dim = shape[2], recurrent_dropout=0.1))
    #model.add(GRU(250, input_length=shape[1], input_dim = shape[2]))
    model.add(Dropout(0.08))
    #model.add(LSTM(50, return_sequences = True, recurrent_dropout=0.1))
    #model.add(Dropout(0.2))
    #model.add(LSTM(50, return_sequences = True, recurrent_dropout=0.1))
    #model.add(Dropout(0.2))
    model.add(LSTM(20))
    model.add(Dropout(0.1)) 
    model.add(Dense(1))
    #optimizers = tf.keras.optimizers.Adam(learning_rate = 0.001)
    #model.compile(loss = "mse", optimizer = optimizers)
    opt = Adam(learning_rate = 0.001)
    model.compile(loss = "mse", optimizer = opt)
    model.summary()
    return model

def remove_ScalerFunction(predict_label_scaler, Scaler):

    predict_label_scaler_numpy = predict_label_scaler

    predict_label_scaler_reshape = np.reshape(predict_label_scaler_numpy, (len(predict_label_scaler_numpy), 1))

    predict_label_reshape = Scaler.inverse_transform(predict_label_scaler_reshape)

    predict_label_reshape_pd = pd.Series(predict_label_reshape.flatten())

    return predict_label_reshape_pd

def CJ_minmaxfunV3_reg(train_data, train_label, test_data, test_label):

    column_name = train_data.columns

    MinMax = MinMaxScaler()

    MinMax.fit(train_data)

    train_data_minmax = MinMax.transform(train_data)
    test_data_minmax = MinMax.transform(test_data)

    train_data_minmax_pd = pd.DataFrame(train_data_minmax, columns = column_name)
    test_data_minmax_pd = pd.DataFrame(test_data_minmax, columns = column_name)

    label_column_name = train_label.name

    label_minmax = MinMaxScaler()

    train_label_numpy = train_label.to_numpy()
    test_label_numpy = test_label.to_numpy()

    train_label_reshape = np.reshape(train_label_numpy, (len(train_label_numpy), 1))
    test_label_reshape = np.reshape(test_label_numpy, (len(test_label_numpy), 1))

    label_minmax.fit(train_label_reshape)

    train_label_minmax = label_minmax.transform(train_label_reshape)
    test_label_minmax = label_minmax.transform(test_label_reshape)

    train_label_minmax_pd = pd.Series(train_label_minmax.flatten(), name = label_column_name)
    test_label_minmax_pd = pd.Series(test_label_minmax.flatten(), name = label_column_name)

    return train_data_minmax_pd, train_label_minmax_pd, test_data_minmax_pd, test_label_minmax_pd, label_minmax


def Adjust_r2(test_value, predict_value, train_x):
    r2 = r2_score(test_value, predict_value)
    datapoint = len(test_value)
    number_feature = train_x.shape[1]
    abj_r2 = 1 - ((1 - r2)*(datapoint - 1))/(datapoint - number_feature - 1)
    #print(r2)
    #print(datapoint)
    #print(number_feature)
    return abj_r2

def model_performaceV2(test_value_minmax, predict_value_minmax, Scaler, train_x):

    test_value = remove_ScalerFunction(test_value_minmax, Scaler)
    predict_value = remove_ScalerFunction(predict_value_minmax, Scaler)

    mse = mean_squared_error(test_value, predict_value)
    mae = mean_absolute_error(test_value, predict_value)
    mape = mean_absolute_percentage_error(test_value, predict_value)
    r2 = r2_score(test_value, predict_value)
    abjr2 = Adjust_r2(test_value, predict_value, train_x)

    return mse, mae, mape, r2, abjr2

def LSTM_dataV2(data, label, gnumber):

    data_numpy = data.to_numpy()
    reshape_data = data_numpy.reshape(len(label), gnumber, data.shape[1])

    return reshape_data

def remove_constant(train_data, test_data):
    
    remove_constact_engine = VarianceThreshold(threshold = 0)
    remove_constact_engine.fit(train_data)
    contant = train_data.columns[~remove_constact_engine.get_support()]
    train_data.drop(labels = contant, axis = 1, inplace = True)
    test_data.drop(labels = contant, axis = 1, inplace = True)

    return train_data, test_data

def remove_object(train_data, test_data):

    train_data = train_data.select_dtypes(exclude = ['object'])
    use_columns = train_data.columns
    test_data_col = test_data[use_columns]
    test_data = test_data_col.copy()

    return train_data, test_data

def remove_QC(train_data, test_data, threshold_value):
    
    remove_constact_engine = VarianceThreshold(threshold = threshold_value)
    remove_constact_engine.fit(train_data)
    contant = train_data.columns[~remove_constact_engine.get_support()]
    train_data.drop(labels = contant, axis = 1, inplace = True)
    test_data.drop(labels = contant, axis = 1, inplace = True)

    return train_data, test_data

def creatCsvlog4(run):
    csvFilePWD =  str(run) +".csv"
    with open(csvFilePWD, 'w') as f:
        f.close()
    print("creat csv:csvFilePWD3")
    return csvFilePWD

def buildManyToOneModel_with(shape, lr_in, lstm_node, dropout_number):
    model = Sequential()
    model.add(LSTM(lstm_node, input_length = shape[1], input_dim = shape[2]))
    model.add(Dropout(dropout_number))
    model.add(Dense(1))
    opt = Adam(learning_rate = lr_in)
    model.compile(loss = "mse", optimizer = opt)
    model.summary()
    return model

def model_performace_classfication(train_y, test_y):

	acc_score = accuracy_score(train_y, test_y)
	bal_acc = balanced_accuracy_score(train_y, test_y)

	return acc_score, bal_acc 

def buildManyToOneModel_with_v2(shape, lr_in, lstm_node, dropout_number):
    model = Sequential()
    model.add(LSTM(lstm_node, return_sequences = True, input_length = shape[1], input_dim = shape[2]))
    model.add(Dropout(dropout_number))
    model.add(LSTM(lstm_node))
    model.add(Dropout(dropout_number))
    model.add(Dense(1))
    opt = Adam(learning_rate = lr_in)
    model.compile(loss = "mse", optimizer = opt)
    return model

def lstm_data_to_2d(LSTM_data_3D):
    lstm_data_2d = LSTM_data_3D.reshape(LSTM_data_3D.shape[0], LSTM_data_3D.shape[1]*LSTM_data_3D.shape[2])
    return lstm_data_2d

def lstm_data_invers_3d(LSTM_data_3D, LSTM_data_2D):
    re_lstm_data_3d = LSTM_data_2D.reshape(LSTM_data_2D.shape[0], LSTM_data_3D.shape[1], LSTM_data_3D.shape[2])
    return re_lstm_data_3d

df = load_data("./wlStatic-db15-TorAccCur-ABCDEFHI.csv")
df_2 = load_data("sbi.csv")

data = df.copy()
label = df_2.pop("sbi")

pwd = creatCsvlog3("-result")
logwrite("kf, mse, mae, mape, r2, acc, bal_acc", pwd)

pwd2 = creatCsvlog3("-allresult")
logwrite("kf, avgmse, stdmse, avgmae, stdmae, avgmape, stdmape, avgr2, stdr2, avgacc, stdacc, avgbal_acc, stdbal_acc", pwd2)

axlist = ["acc1","acc2","acc3","cur"]
wtname = ["or", "wa", "wd", "waa", "wad", "wda", "wdd"]
staticlist = ["sg_mean", "sg_std", "sg_var", "sg_skew", "sg_kur", "sg_max", "sg_min", "sg_mad", "sg_iqr", "sg_rms", "sg_n5", "sg_n25", "sg_n50", "sg_n75", "sg_n95"]
colname = []
for a in axlist:
#   print(a)
    for b in wtname:
#       print(b)
        for c in staticlist:
#           print(a + "-" + b +"-" + c)
            data.pop(a + "-" + b +"-" + c)

for a in range(1, 9):

    train_index, test_index = CJ_kfold(a, data, 8)
    train_index_label, test_index_label = CJ_kfold(a, label, 8)

    train_data = data.loc[train_index, ]
    train_label = label.loc[train_index_label, ]

    test_data = data.loc[test_index, ]
    test_label = label.loc[test_index_label, ]

    train_data_minmax, train_label_minmax, \
    test_data_minmax, test_label_minmax, \
    minmax_scaler = CJ_minmaxfunV3_reg(train_data, train_label, test_data, test_label)

    train_data_minmax, test_data_minmax = remove_QC(train_data_minmax, test_data_minmax, 0.01)

    train_data_minmax, test_data_minmax = remove_cor(train_data_minmax, test_data_minmax, 0.8)

    train_data_lstm = LSTM_dataV2(train_data_minmax, train_label_minmax, 5)
    test_data_lstm = LSTM_dataV2(test_data_minmax, test_label_minmax, 5)

    mselist = []
    maelist = []
    mapelist = []
    r2list = []
    acclist = []
    balacclist = []

    callback = EarlyStopping(monitor = "loss", patience = 200, verbose = 1, mode = "auto")

    train_data_lstm_2d = lstm_data_to_2d(train_data_lstm)

    for b in range(5):

        train_index_model, valid_index_model = CJ_random_kfold(b, train_data_lstm_2d, 5)

        train_data_lstm_model_2d = train_data_lstm_2d[train_index_model]
        train_label_lstm_model_2d = train_label_minmax.to_numpy()[train_index_model]

        valid_data_lstm_model_2d = train_data_lstm_2d[valid_index_model]
        valid_label_lstm_model_2d = train_label_minmax.to_numpy()[valid_index_model]

        print(train_data_lstm_model_2d.shape)
        print(train_label_lstm_model_2d.shape)

        train_data_lstm_model_3d = lstm_data_invers_3d(train_data_lstm, train_data_lstm_model_2d)
        valid_data_lstm_model_3d = lstm_data_invers_3d(train_data_lstm, valid_data_lstm_model_2d)

        model = buildManyToOneModel(train_data_lstm.shape)

        model.fit(train_data_lstm_model_3d, train_label_lstm_model_2d, epochs = 200, batch_size = 4, validation_data = (valid_data_lstm_model_3d, valid_label_lstm_model_2d))

        predict_value = model.predict(test_data_lstm)

        mse, mae, mape, r2, adjr2 = model_performaceV2(test_label_minmax.to_numpy(), predict_value, minmax_scaler, train_data_lstm)

        test_rmScaler = remove_ScalerFunction(test_label_minmax.to_numpy(), minmax_scaler)
        predict_rmScaler = remove_ScalerFunction(predict_value, minmax_scaler)

        test_classlist = []
        predict_classlist = []

        for abc in range(len(test_rmScaler)):
            test_class = 4
            predict_class = 4

            if test_rmScaler[abc] < 0.64:
                test_class = 0
            elif test_rmScaler[abc] > 0.7:
                test_class = 2
            else:
                test_class = 1 

            if predict_rmScaler[abc] < 0.64:
                predict_class = 0
            elif predict_rmScaler[abc] > 0.7:
                predict_class = 2
            else:
                predict_class = 1

            test_classlist.append(test_class)
            predict_classlist.append(predict_class)

        acc, bal_acc = model_performace_classfication(test_classlist, predict_classlist)

        pwd4 = creatCsvlog3(str(a) + "-" + str(b))
        logwrite("test,predict,testlabel,predictlabel", pwd4)

        mselist.append(mse)
        maelist.append(mae)
        mapelist.append(mape)
        r2list.append(r2)
        acclist.append(acc)
        balacclist.append(bal_acc)

        for abc in range(len(test_rmScaler)):
            logwrite(str(test_rmScaler[abc]) + ',' + str(predict_rmScaler[abc]) + ',' + str(test_classlist[abc]) + ',' + str(predict_classlist[abc]), pwd4)

        logwrite(str(a)+',' +str(mse) + ',' + str(mae) + ',' + str(mape) + ',' + str(r2) + ',' + str(acc) + ',' + str(bal_acc), pwd)

    avgmse = np.mean(mselist)
    avgmae = np.mean(maelist)
    avgmape = np.mean(mapelist)
    avgr2 = np.mean(r2list)
    avgacc = np.mean(acclist)
    avgbalacc = np.mean(balacclist)

    stdmse = np.std(mselist)
    stdmae = np.std(maelist)
    stdmape = np.std(mapelist)
    stdr2 = np.std(r2list)
    stdacc = np.std(acclist)
    stdbalacc = np.std(balacclist)

    logwrite(str(a) + ',' +str(avgmse) + ',' +str(stdmse) + ',' + str(avgmae) + ',' +str(stdmae) + ',' + str(avgmape) + ',' +str(stdmape) + ',' + str(avgr2) + ',' +str(stdr2) + ',' + str(avgacc) + ',' +str(stdacc) + ',' + str(avgbalacc) + ',' +str(stdbalacc), pwd2)









































