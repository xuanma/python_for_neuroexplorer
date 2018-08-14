# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 22:57:08 2018

@author: Xuan
"""
import nexfile
import numpy as np
import scipy.stats as stats

class my_nex_class:
    def __init__(self, file_name):
        reader = nexfile.Reader(useNumpy=True)
        self.data = reader.ReadNex5File(file_name)
        self.total_time = self.data['FileHeader']['End']
        
    def grab_spike_data(self):
        spike_data = []
        for each in self.data['Variables']:
            if each['Header']['Type'] == 0:
               spike_data.append(each['Timestamps'])
        return spike_data
        
        
    def grab_cont_data(self):
        cont_var = []
        for each in self.data['Variables']:
            if each['Header']['Type'] == 5:
               cont_var.append(each['ContinuousValues'])
               self.cont_var_fs = each['Header']['SamplingRate'] 
        return cont_var
        
        
    def grab_waveform_data(self):
        waveforms = []
        for each in self.data['Variables']:
            if each['Header']['Type'] == 3:
               waveforms.append(each['WaveformValues'])
        return waveforms
        
    def grab_spike_names(self):
        spike_names = []
        for each in self.data['Variables']:
            if each['Header']['Type'] == 0:
               spike_names.append(each['Header']['Name'])
        return spike_names
    
    def grab_cont_names(self):
        cont_names = []
        for each in self.data['Variables']:
            if each['Header']['Type'] == 5:
               cont_names.append(each['Header']['Name'])
        return cont_names
        
    def bin_spike_data(self, bin_size):
        self.timeframe = np.arange(bin_size, self.total_time, bin_size)
        spike = self.grab_spike_data()
        firing_rate = np.empty((len(self.timeframe), len(spike)))
        for i in range(len(self.timeframe)):
            if i == 0:
               for j in range(len(spike)):
                   temp = np.where(spike[j]<self.timeframe[i])
                   firing_rate[i][j] = np.size(temp)
            else:
               for j in range(len(spike)):
                   temp = np.where((spike[j]>self.timeframe[i-1])&(spike[j]<self.timeframe[i]))
                   firing_rate[i][j] = np.size(temp)
        return firing_rate
    
    def smooth_firing_rate(self, firing_rate, kernel_SD):
        print('Smoothing the firing rates...')
        bin_size = self.timeframe[1] - self.timeframe[0]
        n_sample, n_ch = np.size(firing_rate, 0), np.size(firing_rate, 1)
        smoothed_firing_rate = np.zeros((n_sample, n_ch))
        kernel_hl = np.ceil( 3 * kernel_SD / (bin_size) )
        normalDistribution=stats.norm(0, kernel_SD)
        x = np.arange(-kernel_hl*bin_size, kernel_hl*bin_size, bin_size)
        kernel = normalDistribution.pdf(x)
        nm = np.convolve(kernel, np.ones((n_sample))).T
        for i in range(n_ch):
            aux_smoothed_FR = np.convolve(kernel,firing_rate[:,i]) / nm
            smoothed_firing_rate[:,i] = aux_smoothed_FR[int(kernel_hl)-1:-int(kernel_hl)]
        return smoothed_firing_rate
    
    def cont_downsample(self, new_fs):
        fs = self.cont_var_fs
        data = np.asarray(self.grab_cont_data()).T
        n = int(np.floor(fs/new_fs))
        new_data = np.empty((0, np.size(data,1)))
        for i in range(1,int(np.size(data,0)/n)+1):
            new_data = np.vstack((new_data, data[i*n, :]))
        return new_data
       
    

if __name__ == "__main__":
   file_name = "Jango_IsoBoxCO_HC_SpikesEMGs_07312015_SN_001.nex5"
   myNex = my_nex_class(file_name)
   spike_data = myNex.grab_spike_data()
   spike_names = myNex.grab_spike_names()
   cont_data = myNex.grab_cont_data()
   cont_names = myNex.grab_cont_names()
   waveforms = myNex.grab_waveform_data()
   firing_rate = myNex.bin_spike_data(0.05)
   firing_rate = myNex.smooth_firing_rate(firing_rate, 0.05)


#%%  
#import matplotlib.pyplot as plt
#import numpy as np
#plt.figure()
#neuron_Number = 10
#n_waveform = np.size(waveforms[neuron_Number], 0)
#for i in range(n_waveform):
#    plt.plot(waveforms[10][i,:],'k')
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
        