# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 22:27:48 2018

@author: Xuan
"""
import nexfile
from my_nex_class import my_nex_class
file_name = "Z:/data/Greyson_17L2/NeuroexplorerFile/20180813_Greyson_PG_001.nex5"
reader = nexfile.Reader(useNumpy=True)
data = reader.ReadNex5File(file_name)
myNex = my_nex_class(file_name)
spike_data = myNex.grab_spike_data()
spike_names = myNex.grab_spike_names()
cont_data = myNex.grab_cont_data()
cont_names = myNex.grab_cont_names()
firing_rate = myNex.bin_spike_data(0.05)
firing_rate = myNex.smooth_firing_rate(firing_rate, 0.05)
force = myNex.cont_downsample(1/0.05)
f = force[:,0] - force[:,1]



