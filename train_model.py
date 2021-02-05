import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold

from load_data import create_eeg_data

from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
import pickle
from scipy.stats import zscore

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import cv2
import mne
from mne import io
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, compute_source_psd
from PIL import Image

#set path
path = 'D:/NISE_project_data/eeg-motor-movementimagery-dataset-1.0.0/files/'
#path = '/home/matthijspals/physionet.org/files/eegmmidb/1.0.0'

def create_spectrogram(epochs,trial,frequencies=np.arange(8, 30, 1),n_cycles=10,average=False,tmin=0.25,tmax=4,verbose=True, n_ch=3):
    power = mne.time_frequency.tfr_morlet(epochs ,n_cycles=n_cycles,
                                      freqs=frequencies, average=False, return_itc=False,verbose=False, use_fft=True)
    times=np.linspace(start=tmin,stop=tmax,num=epochs.get_data().shape[2])
    if verbose:
        #plt.figure(figsize=(15, 10))
        #plt.subplot(1,3,1)
        #plt.pcolormesh(times,frequencies,power.data[trial,0,:,:])
        #plt.subplot(1,3,2)
        #plt.pcolormesh(times,frequencies,power.data[trial,1,:,:])
        #plt.subplot(1,3,3)
        #plt.pcolormesh(times,frequencies,power.data[trial,2,:,:])
        img=plt.imshow(power.data[trial,0:n_ch,:,:].reshape(n_ch*frequencies.shape[0],601),aspect="auto",origin="lower")
    
    return power.data[trial,0:n_ch,:,:].reshape(n_ch*frequencies.shape[0],601),img #(trials,channels,spectogram1,spectogram2)

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )
    
import sys
#participants = [1, 2, 3, 6, 7, 12, 14, 15, 18, 20, 21, 22, 29, 30, 31, 32, 33, 36, 37, 40]
n_ch = 3
participants =[int(sys.argv[1])]
tmax = 4
tmin = -0.25
fs = 160.
trials=21
samples = int((tmax+tmin)*fs+1)
data = np.zeros((0,samples,n_ch))
labels = np.zeros(0)
lfreq, hfreq= 8, 30
freq_spec =  np.logspace(*np.log10([lfreq, hfreq]), num=110)
part_idx=0
n_cycles = freq_spec/2
im_chan=4
for part in participants:

    #load and filter data
    eeg_data = create_eeg_data(path, part = part, task = [0,1,0,1],baseline=True, filt=[lfreq,hfreq])
    epochs2 = eeg_data['t2'][0].all_sessions().copy()
    epochs4 = eeg_data['t4'][0].all_sessions().copy()




    count = 0

    sampl = np.random.randint(low=0, high=eeg_data["cl_eyes"][0].get_data().shape[1]-samples, size=(trials,), dtype='l')
    events_array = np.zeros([trials,samples,64])

    raw0 = eeg_data['cl_eyes'][0].copy()
    count = 0
    events=[]
    for i in range(trials):
        events_array[i,:,:]=raw0.get_data()[:,sampl[count]:sampl[count]+samples].T

        count = count+1
    labels0 = np.zeros([trials,])

    new_baseline_raw=mne.io.RawArray(events_array.reshape([64,trials*samples]),epochs2.info,verbose=False)
    onsets=np.arange(0,new_baseline_raw.get_data().shape[1]/fs,samples/fs)
    duration = np.ones([trials,])*(samples/fs)
    description = []
    for i in range(trials):
        description.append("T0")

    annot_base= mne.Annotations(onset=onsets,duration=duration,description=description)
    new_baseline_raw = new_baseline_raw.copy().set_annotations(annot_base)
    epochs0 = mne.EpochsArray(events_array.transpose([0,2,1]), epochs2.info, events=mne.events_from_annotations(new_baseline_raw,
                                    event_id={"T0":0})[0], tmin=0.25, event_id=None, reject=None, flat=None, reject_tmin=None,
                                    reject_tmax=None, baseline=None, proj=True, on_missing='raise', metadata=None, 
                                      selection=None, verbose=None)
    epochs0.info["highpass"]=lfreq
    epochs0.info["lowpass"]=hfreq


    epochs_train0 = epochs0.copy().crop(tmin=-tmin, tmax=tmax)
    ica = ICA(n_components=64, random_state=97)
    ica.fit(epochs_train0)
    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(epochs_train0,ch_name='Fpz')
    ica.exclude = eog_indices
    #aply ICA
    ica.apply(epochs0)
    epochs_train0 = epochs0.copy().crop(tmin=-tmin, tmax=tmax).pick_channels(["C3","Cz","C4"])
    
    
    spec_data_train0 = np.zeros([len(participants),trials,n_ch*freq_spec.shape[0],samples])
    img_data_train0 = np.zeros([len(participants),trials,n_ch*freq_spec.shape[0],samples,im_chan]) 
    for i in range(trials):
        spec_data_train0[part_idx,i,:,:],img0=create_spectrogram(epochs_train0,i,frequencies=freq_spec,n_cycles=n_cycles,verbose=True)
        img_data_train0[part_idx,i,:,:,:]= img0.cmap(img0.norm(img0.get_array()))
    
    

    
    #crop
    epochs_train2 = epochs2.copy().crop(tmin=-tmin, tmax=tmax)
    
    #ICA epochs2
    ica = ICA(n_components=64, random_state=97)
    ica.fit(epochs_train2)
    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(epochs_train2,ch_name='Fpz')
    ica.exclude = eog_indices
    #aply ICA
    ica.apply(epochs2)
    
    #ICA applied
    epochs_train2 = epochs2.copy().crop(tmin=-tmin, tmax=tmax).pick_channels(["C3","Cz","C4"])
    
    spec_data_train2 = np.zeros([len(participants),trials*2,n_ch*freq_spec.shape[0],samples]) 
    img_data_train2 = np.zeros([len(participants),trials*2,n_ch*freq_spec.shape[0],samples,im_chan]) 
    for i in range(trials*2):
            spec_data_train2[part_idx,i,:,:],img2=create_spectrogram(epochs_train2,i,frequencies=freq_spec,n_cycles=n_cycles,verbose=True)
            img_data_train2[part_idx,i,:,:,:]= img2.cmap(img2.norm(img2.get_array()))
    labels2 = epochs2.events[:, -1] - 1
    epochs_data_train2 = epochs_train2.get_data().transpose([0,2,1])
    
    
    
    #crop
    epochs_train4 = epochs4.copy().crop(tmin=-tmin, tmax=tmax)
    
    #ICA preprocessing
    ica = ICA(n_components=64, random_state=97)
    ica.fit(epochs_train4)
    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(epochs_train4,ch_name='Fpz')
    ica.exclude = eog_indices
    #aply ICA
    ica.apply(epochs4)
    
    #ICA applied
    epochs_train4 = epochs4.copy().crop(tmin=-tmin, tmax=tmax).pick_channels(["C3","Cz","C4"])
    
    spec_data_train4 = np.zeros([len(participants),trials*2,n_ch*freq_spec.shape[0],samples])
    img_data_train4 = np.zeros([len(participants),trials*2,n_ch*freq_spec.shape[0],samples,im_chan])
    for i in range(trials*2):
            spec_data_train4[part_idx,i,:,:],img4=create_spectrogram(epochs_train4,i,frequencies=freq_spec,n_cycles=n_cycles,verbose=True)
            img_data_train4[part_idx,i,:,:,:]= img4.cmap(img4.norm(img4.get_array()))
            
    labels4 = epochs4.events[:, -1] +1
    epochs_data_train4 = epochs_train4.get_data().transpose([0,2,1])

    #concatenate to bigg file
    labels240 = np.concatenate((labels2, labels4,labels0))
    #data24 = np.concatenate((epochs_data_train2, epochs_data_train4, epochs_data_train0),axis=0)
    spec240 = np.concatenate((spec_data_train2,spec_data_train4,spec_data_train0),axis=1)
    img240 = np.concatenate((img_data_train2,img_data_train4,img_data_train0),axis=1)
    #data = np.concatenate((data, data24))
    part_idx = part_idx+1
    with open('data.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    with open('labels.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)
        
img_old = img240
img240 = img240.squeeze()[:,:,:,:3]
img_comp = (np.zeros((np.shape(img240)[0],32,32, 3)))
for i in range(np.shape(img240)[0]):
    img_comp[i] =  cv2.resize(img240[i], dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
img_comp -= np.min(img_comp)
img_comp /=np.max(img_comp)
img240 = img_comp
#img_comp = np.expand_dims(img_comp, axis = 3)

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in sss.split(img240.squeeze(), labels240):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = img240.squeeze()[train_index,:,:,:3], img240.squeeze()[test_index,:,:,:3]
    y_train, y_test = labels240[train_index], labels240[test_index]
    
#ONE HOT ENCODING
from keras.utils import to_categorical
y_train_OH = to_categorical(y_train)
y_test_OH = to_categorical(y_test)


X_grey = np.zeros((len(X_train),32,32))
for i in range(len(X_train)):
    X_grey[i]=cv2.cvtColor(np.float32(X_train[i]), cv2.COLOR_BGR2GRAY)
X_train = X_grey


X_grey = np.zeros((len(X_test),32,32))
for i in range(len(X_test)):
    X_grey[i]=cv2.cvtColor(np.float32(X_test[i]), cv2.COLOR_BGR2GRAY)
X_test= X_grey

X_test = np.expand_dims(X_test, axis = 3)
X_train = np.expand_dims(X_train, axis = 3)


np.savez_compressed("x_train", X_train)
np.savez_compressed("x_test", X_test)
np.savez_compressed("x_norm", X_train[::20])
np.savez_compressed("y_train", y_train_OH)
np.savez_compressed("y_test", y_test_OH)

#from keras.models import Sequential, InputLayer
#from keras.layers import Dense, Activation, Flatten
#from keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D
#from tensorflow import keras

#from tensorflow.keras import Sequential, Input
#from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, BatchNormalization, Dropout, Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, \
    Dropout


input_shape = X_test[0].shape
input_layer = Input(input_shape)

layer = Conv2D(filters=16,
               kernel_size=(5, 5),
               strides=(2, 2),
               activation='relu',
               use_bias=False)(input_layer)

layer = Conv2D(filters=32,
               kernel_size=(3, 3),
               activation='relu',
               use_bias=False)(layer)
#layer = AveragePooling2D()(layer)
layer = Conv2D(filters=8,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               use_bias=False)(layer)

layer = Flatten()(layer)
layer = Dropout(0.01)(layer)
layer = Dense(units=5,
              activation='softmax')(layer)

model = Model(input_layer, layer)

# compile the model
opt = Adam(learning_rate = 3e-3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#print(model.summary())


import tensorflow as tf
class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["val_accuracy"]
        if accuracy >= self.threshold:
            self.model.stop_training = True
            
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
batch_size=32
no_epochs= 48
# Fit data to model
history = model.fit(X_train,  y_train_OH,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=True,
        validation_data=(X_test, y_test_OH), callbacks=[MyThresholdCallback(threshold=0.6)])

# Generate generalization metrics
scores = model.evaluate(X_test, y_test_OH, verbose=1)

save_model = True
if save_model:
    model.save("CNN_60.h5", save_format = 'h5')