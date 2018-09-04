# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:09:50 2018

@author: xuanm
"""
import numpy as np
import tensorflow as tf
import nexfile
from my_nex_class import my_nex_class
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from scipy import signal
from scipy import stats

def get_one_minute_data(bin_size,start_min,X,y):
    start_ind = int(np.floor(start_min*60/bin_size))
    end_ind = int(start_ind + np.floor(60/bin_size))
    X_one_minute = X[start_ind:end_ind,:]
    y_one_minute = y[start_ind:end_ind]
        
    return X_one_minute, y_one_minute

def get_n_minute_data(bin_size, start_min, end_min, X, y):
    X_n = np.empty((0,np.size(X,1)))    
    y_n = np.empty((0,np.size(y,1)))
    for i in range(start_min, end_min):
        temp1, temp2 = get_one_minute_data(0.05,i,X,y)
        X_n = np.vstack((X_n, temp1))
        y_n = np.vstack((y_n, temp2))
    return X_n, y_n

def form_rnn_data(X, y, N):
    n_X, dim_X, n_y, dim_y = np.size(X, 0), np.size(X, 1), np.size(y, 0), np.size(y,1)
    X_out, y_out = np.empty((n_X - N + 1, N, dim_X)), np.empty((n_y - N + 1, dim_y))
    for i in range(N-1, n_X):
        temp = np.vstack((X[j,:] for j in range(i-N+1,i+1)))
        X_out[i-N+1, :, :] = temp
        y_out[i-N+1,:] = y[i, :]
    return X_out, y_out        

def next_batch(batch_size, X, y):
    t0 = np.random.randint(0, len(X))
    tn = t0 + batch_size
    return X[t0:tn, :, :], y[t0:tn, :]

def reconstruct_vaf(y, y_hat):
    return 1 - np.sum((y-y_hat)**2)/np.sum((y - np.mean(y, axis = 0))**2)

def EMG_processing(data):
    filtered_EMG = []
    fs = 2010
    bhigh, ahigh = signal.butter(4,50/(fs/2), 'high')
    blow, alow = signal.butter(4,10/(fs/2), 'low')
    for each in data:
        temp = signal.filtfilt(bhigh, ahigh, each)
        f_abs_emg = signal.filtfilt(blow ,alow, np.abs(temp))
        filtered_EMG.append(f_abs_emg)
        
    return filtered_EMG
   
def EMG_downsample(new_fs, data):
    fs = 2010
    data = np.asarray(data).T
    n = int(np.floor(fs/new_fs))
    new_data = np.empty((0, np.size(data,1)))
    for i in range(1,int(np.size(data,0)/n)+1):
        new_data = np.vstack((new_data, data[i*n, :]))
    return new_data




file_name = "Z:/data/Greyson_17L2/NeuroexplorerFile/20180831_Greyson_PG_003.nex5"

reader = nexfile.Reader(useNumpy=True)
data = reader.ReadNex5File(file_name)
myNex = my_nex_class(file_name)
spike_data = myNex.grab_spike_data()
spike_names = myNex.grab_spike_names()
firing_rate = myNex.bin_spike_data(0.05)
spike = myNex.smooth_firing_rate(firing_rate, 0.05)
spike = spike[:,5:]

EMG_list = {'FCR1', 'FCR2', 'FCU1', 'FCU2', 'FDP2', 'FDP3', 'FDS1', 'FDS2', 'FPB', 'PT', 'LUM', 'FDI', 'EDC3'};
ch1 = [28, 25, 26, 27, 24,  0, 30, 31,  3,  1,  4,  5,  6];  
ch2 = [19, 22, 21, 20, 23, 15, 17, 16, 12, 14, 11, 10,  9];

from load_intan_rhd_format import read_data
filename = "Z:/data/Greyson_17L2/Cerebusdata/20180831/20180831-Greyson_PG_001.rhd"
data = read_data(filename)
raw_EMG_single = data['amplifier_data']
raw_EMG_diff = []
t_amplifier = data['t_amplifier']
s = int(np.where(t_amplifier == 0)[0])
for i in range(len(EMG_list)):
    d = raw_EMG_single[ch1[i]] - raw_EMG_single[ch2[i]]
    raw_EMG_diff.append(d[s:])
    
filtered = EMG_processing(raw_EMG_diff)
EMG = EMG_downsample(1/0.05,filtered)
EMG = EMG[0:len(spike),:]
#%%
from sklearn.preprocessing import MinMaxScaler
tf.reset_default_graph() 
n_steps = 10
n_inputs = 91
n_neurons = n_inputs
n_outputs = 13

spike_train, EMG_train = get_n_minute_data(0.05,0,10,spike,EMG)
spike_train, EMG_train = form_rnn_data(spike_train, EMG_train, 10)

spike_test, EMG_test = get_n_minute_data(0.05,10,15,spike,EMG)
spike_test, EMG_test = form_rnn_data(spike_test, EMG_test, 10)

scaler = MinMaxScaler()
EMG_train = scaler.fit_transform(EMG_train)
EMG_test = scaler.fit_transform(EMG_test)

X = tf.placeholder(tf.float64, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float64, [None, n_outputs])    

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons)
gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)

rnn_outputs, states = tf.nn.dynamic_rnn(gru_cell, X, dtype=tf.float64)

out_temp = rnn_outputs[:, n_steps-1, :]
learning_rate = 0.001

outputs = tf.layers.dense(out_temp, n_outputs, activation = tf.identity)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epoch = 20
batch_size = 50
n_batches = np.size(spike_train,0)//batch_size

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoch):
        print(epoch)
        for iteration in range(n_batches):
            X_batch, y_batch = next_batch(batch_size, spike_train, EMG_train)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 2 == 0:
           mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
           print(epoch, "\tMSE:", mse)
    
    y_pred_basic = sess.run(outputs, feed_dict={X: spike_test})
res_basic = explained_variance_score(EMG_test, y_pred_basic, multioutput='raw_values')
print(res_basic) 
#%%
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(EMG_test[3000:3400, 12],'k')
plt.plot(y_pred_basic[3000:3400, 12],'b')   





















