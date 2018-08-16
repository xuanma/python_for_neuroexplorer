# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:32:33 2018

@author: Xuan
"""

""" Loading the file """
import tensorflow as tf
import numpy as np
import scipy.io as sio
from sklearn.metrics import r2_score

import nexfile
from my_nex_class import my_nex_class

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

def get_data(file_name):
    reader = nexfile.Reader(useNumpy=True)
    data = reader.ReadNex5File(file_name)
    myNex = my_nex_class(file_name)
    spike_data = myNex.grab_spike_data()
    spike_names = myNex.grab_spike_names()
    cont_data = myNex.grab_cont_data()
    cont_names = myNex.grab_cont_names()
    firing_rate = myNex.bin_spike_data(0.05)
    spike = myNex.smooth_firing_rate(firing_rate, 0.05)
    kin_p = myNex.cont_downsample(1/0.05)
    return spike, kin_p
    

file_name1 = "Z:/data/Greyson_17L2/NeuroexplorerFile/20180813_Greyson_PG_001.nex5"
file_name2 = "Z:/data/Greyson_17L2/NeuroexplorerFile/20180813_Greyson_PG_002.nex5"
file_name3 = "Z:/data/Greyson_17L2/NeuroexplorerFile/20180813_Greyson_PG_003.nex5"

spike1, kin_p1 = get_data(file_name1)
spike2, kin_p2 = get_data(file_name2)
spike3, kin_p3 = get_data(file_name3)
#%%
from sklearn.preprocessing import MinMaxScaler
tf.reset_default_graph() 
n_steps = 10
n_inputs = 96
n_neurons = n_inputs
n_outputs = 1

spike_train, kin_p_train = get_n_minute_data(0.05,0,10,spike3,kin_p3)
spike_train, kin_p_train = form_rnn_data(spike_train, kin_p_train, 10)
kin_p_train = kin_p_train[:,0] - kin_p_train[:,1]
kin_p_train = kin_p_train.reshape((np.size(kin_p_train,0),1))
scaler = MinMaxScaler()
kin_p_train = scaler.fit_transform(kin_p_train)

spike_test, kin_p_test = get_n_minute_data(0.05,10,15,spike3,kin_p3)
spike_test, kin_p_test = form_rnn_data(spike_test, kin_p_test, 10)
kin_p_test = kin_p_test[:,0] - kin_p_test[:,1]
kin_p_test = kin_p_test.reshape((np.size(kin_p_test,0),1))
kin_p_test = scaler.fit_transform(kin_p_test)

X = tf.placeholder(tf.float64, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float64, [None, n_outputs])    
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons)
gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)
rnn_outputs, states = tf.nn.dynamic_rnn(gru_cell, X, dtype=tf.float64)
out_temp = rnn_outputs[:, n_steps-1, :]
learning_rate = 0.001

outputs = tf.layers.dense(out_temp, n_outputs, activation = tf.nn.softplus)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epoch = 8
batch_size = 50
n_batches = np.size(spike_train,0)//batch_size

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoch):
        print(epoch)
        for iteration in range(n_batches):
            X_batch, y_batch = next_batch(batch_size, spike_train, kin_p_train)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 2 == 0:
           mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
           print(epoch, "\tMSE:", mse)
    
    y_pred_gru = sess.run(outputs, feed_dict={X: spike_test})
res_gru = r2_score(kin_p_test, y_pred_gru)
print(res_gru)
#%%
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(kin_p_test[200:700,0],'b')
plt.plot(y_pred_basic[200:700,0],'r')
#%%
plt.figure(1)
fig,ax = plt.subplots(figsize = (10,4),dpi = 100)

#plt.ylim(60,73)  
#plt.xlim(0, 34)

plt.subplots_adjust(bottom = 0.15)  

t = np.arange(0,400,1)  
ax.set_title('Decoding grasping force from cortical signals (Greyson)',fontsize=14)
#plt.xticks(t*0.05)
plt.grid()

ax.set_xlabel(xlabel='time (s)',fontsize=12)
ax.set_ylabel(ylabel='force', fontsize = 12)

p1, = plt.plot(t*0.05,kin_p_test[300:700], 'k-')
p2, = plt.plot(t*0.05,y_pred_basic[300:700], 'b-')
#p3 = plt.plot(t*0.05,y_pred_lstm[200:700], 'r-')
p4, = plt.plot(t*0.05,y_pred_gru[300:700], 'r-')

l1 = plt.legend([p1, p2, p4], ["Actual", "Basic RNN Cell (VAF = 61.07%)", "LSTM Cell (VAF = 68.15%)"], loc='upper right',fontsize = 10)    





