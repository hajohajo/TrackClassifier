#Restrict to one gpu
import imp
try:
        imp.find_module('setGPU')
        import setGPU
except ImportError:
        found = False
#/////////////////////
import matplotlib.pyplot as plt
import keras.backend as K
K.set_learning_phase(0)
import pylab as P
import pandas as pd
import numpy as np
import keras.callbacks

from sklearn.metrics import roc_auc_score
from root_pandas import read_root

from keras.models import Model,load_model
from keras.layers import Input,Dense,Dropout,Activation

model=load_model('model.h5')

print model.summary()

arch = model.to_json()

with open('architecture.json','w') as arch_file:
    arch_file.write(arch)
model.save_weights('weights.h5')
