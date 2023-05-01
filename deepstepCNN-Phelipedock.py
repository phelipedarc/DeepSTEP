import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential , Model, load_model
from tensorflow.keras.layers import BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Conv2D, Input,LeakyReLU,GlobalMaxPooling2D
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import h5py
#import autokeras as ak
import numpy as np
import glob
#Define the path to the stamps
path_to_the_stamps = '/tf/dados10Tdock/STEP/cutouts'
path_model ='/tf/dados10Tdock/STEP/Andre_phelipe/SPLUS/Model_Splus/NEWcorrectSplus.h5'
path=glob.glob(path_to_the_stamps +'/*/*')
path.sort()
print(path)
model = load_model(path_model)
#Define Normalize Function
def normalize_function(img):
    imag = img
    maximo = np.max(img)
    minimo = np.min(img)
    delta = abs(maximo - minimo)
    imag = (imag - minimo)/abs(delta)

    return imag
images=[]
names = []
for i in path:
    savepath = i
    a = glob.glob(i+'/*.fits')
    names = []
    images=[]
    for b in range(len(a)):
        z = fits.open(a[b])
        names.append(a[b].split('/')[-1].split('_')[-1].split('.')[0])
        images.append(fits.getdata(a[b]))
    images = np.array(images).astype(np.float64)
    names = np.array(names)
    P = [99.2, 99.2, 99.2]
    m = 0.01
    channels = ['search','template','DIA']
    for idx, (ch, percent) in enumerate(zip(channels, P)):
        for j in range(len(images)):
            ch_max = np.percentile(images[j, :, :, idx], percent)
            ch_min = (np.percentile(images[j, :, :, idx], m))
            images[j, :, :, idx] = np.clip(images[j, :, :, idx], ch_min, ch_max)
    for k in range(len(images)):
        for z in range(3):
            images[k,:,:,z]=normalize_function(images[k,:,:,z])
    print('Input Shape -->',images.shape)
    score = model.predict(images)
    score2 = (score>0.55)*1

    cnnscore=np.column_stack((names,score2[:,0],score[:,0]))
    #Saving on path:
    namecsv = savepath.split('/')[-2].split('_')[0]
    cnnscore = pd.DataFrame(cnnscore,columns=['ID','Transient_treshold','Score'])
    cnnscore = cnnscore.astype({'Transient_treshold':'float','Score':'float'})
    print('Most Probable Transient --> ','ID-->',names[cnnscore['Score'].idxmax()] ,'<--Score-->',cnnscore['Score'][cnnscore['Score'].idxmax()])
    print('Number of Probable Transients: ',cnnscore['Transient_treshold'].value_counts()[1])
    print('Number of Probable Artifacts: ',cnnscore['Transient_treshold'].value_counts()[0])
    print('Rate: ',100*cnnscore['Transient_treshold'].value_counts()[0]/(cnnscore['Transient_treshold'].value_counts()[1]+cnnscore['Transient_treshold'].value_counts()[0]))
    cnnscore.to_csv(savepath+'/'+namecsv+'CNNscore.csv',index=False)





