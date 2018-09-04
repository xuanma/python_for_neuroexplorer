# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:31:59 2018

@author: xuanm
"""

from my_nex_class import my_nex_class
file_name = "Z:/data/Greyson_17L2/NeuroexplorerFile/20180813_Greyson_PG_001.nex5"
myNex = my_nex_class(file_name)
spike_data = myNex.grab_spike_data()
spike_names = myNex.grab_spike_names()
cont_data = myNex.grab_cont_data()
cont_names = myNex.grab_cont_names()
firing_rate = myNex.bin_spike_data(0.05)
spike = myNex.smooth_firing_rate(firing_rate, 0.05)
#%%
import matplotlib.pyplot as plt

plt.plot(spike[0:100,2])