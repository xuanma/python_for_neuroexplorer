# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 23:02:25 2018

@author: xuanm
"""
import nexfile
from load_intan_rhd_format import read_data
filename = "Z:/data/Greyson_17L2/Cerebusdata/20180831/20180831-Greyson_PG_001.rhd"
data = read_data(filename)
EMG_1 = data['amplifier_data'][0]
#%%
w = nexfile.NexWriter(30000)
#w.fileData['FileHeader']['Comment'] = 'this is a comment'
#w.AddNeuron('neuron1', [1, 2, 3, 4])
w.AddContVarWithSingleFragment('EMG1', 0, 2010, EMG_1)
#w.WriteNexFile('C:\\Data\\testFileWrittenInPython.nex')
w.WriteNex5File('C:\\mcode\\python_for_neuroexplorer\\20180831_Greyson_PG_003.nex5', 1)