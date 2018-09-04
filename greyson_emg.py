# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 21:40:12 2018

@author: xuanm
"""
import numpy as np
import tensorflow as tf
import nexfile
from my_nex_class import my_nex_class
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

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

file_name = "Z:/data/Greyson_17L2/NeuroexplorerFile/20180831_Greyson_PG_003.nex5"

reader = nexfile.Reader(useNumpy=True)
data = reader.ReadNex5File(file_name)
myNex = my_nex_class(file_name)
spike_data = myNex.grab_spike_data()
spike_names = myNex.grab_spike_names()

EMG_data = myNex.EMG_processing(['APB', 'FCR1', 'FCR2', 'FCU1', 'FCU2', 
                                 'FDP1', 'FDP2', 'FDP3', 'FDS1', 'FDS2',
                                 'FPB', 'PT'])

firing_rate = myNex.bin_spike_data(0.05)
spike = myNex.smooth_firing_rate(firing_rate, 0.05)
EMG = myNex.EMG_downsample(1/0.05, EMG_data)   
#%%
tf.reset_default_graph() 
n_steps = 10
n_inputs = 96
n_neurons = n_inputs
n_outputs = 12

spike_train, EMG_train = get_n_minute_data(0.05,0,10,spike,EMG)
spike_train, EMG_train = form_rnn_data(spike_train, EMG_train, 10)

spike_test, EMG_test = get_n_minute_data(0.05,10,15,spike,EMG)
spike_test, EMG_test = form_rnn_data(spike_test, EMG_test, 10)


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
plt.plot(EMG_test[3000:3400, 2],'k')
plt.plot(y_pred_basic[3000:3400, 2],'b')








    