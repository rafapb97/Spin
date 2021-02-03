import numpy as np
import os
import glob
import mne
from pathlib import Path
from mne.channels import make_standard_montage
from mne import concatenate_epochs

class Task:
    """
    Holds data of one task for one participant
    ...

    Attributes
    ----------
    s1, s2, s3 : mne epoch objects?

    """

    def __init__(self):
        self.s1=[]
        self.s2=[]
        self.s3=[]
        
    def __call__(self, s):
        if self.s1==[]:
            self.s1= s
        elif self.s2 ==[]:
            self.s2= s
        else:
            self.s3=s
        
    def all_sessions(self):
        return concatenate_epochs([self.s1,self.s2,self.s3])
        
        
def create_eeg_data(path, part =1, baseline = False, task=[1,1,1, 1], tmin =-1., tmax = 5., filt=[7., 30.]):    
    """Creates eeg_data object

    Parameters
    ----------
    path : string
        location of data
    n_part : int
        number of participants to load data from
    baseline : boolean
        whether or not to load the baseline (eyes open, eyes closed)
    task : binary array
        index corresponds to task number, 1 indicates load, 0 indicates no loading
    tmin : float
        time before stim to include
    tax : float
        time after stim to include
    filt : array
        filter thresholds

    Returns: 
    --------
    eeg_data : dictionary
        holds loaded data

    """
    
    eeg_data = {
        'op_eyes':[],
        'cl_eyes':[],
        't1':[],
        't2':[],
        't3':[], 
        't4':[]
    }
    
    #montage contains channel positions, for topoplots etc
    montage = make_standard_montage('standard_1005')
    
    task1 = ['*03.edf','*07.edf','*11.edf']
    task2 = ['*04.edf','*08.edf','*12.edf']
    task3 = ['*05.edf','*09.edf','*13.edf']
    task4 = ['*06.edf','*10.edf','*14.edf']

    if part<10:
        i='S00' + str(part)
    elif part<100:
        i='S0' + str(part)
    else:
        i='S' + str(part)
    i = os.path.join(path,i)

    if Path(i).is_dir():
        pat = i+'/'
        print(pat)
        if baseline:
            base_eyes_op = glob.glob(pat+'*01.edf')[0]
            base_eyes_cl = glob.glob(pat+'*02.edf')[0]

            raw = mne.io.read_raw_edf(base_eyes_op,verbose=False,preload=True)

            raw.rename_channels(lambda x: x.strip('.'))
            raw.set_montage(montage, match_case=False)

            raw.filter(filt[0], filt[1], fir_design='firwin', skip_by_annotation='edge',verbose=False)
            eeg_data["op_eyes"].append(raw)


            raw.rename_channels(lambda x: x.strip('.'))
            raw.set_montage(montage, match_case=False)

            raw = mne.io.read_raw_edf(base_eyes_cl,verbose=False,preload=True)
            raw.filter(filt[0], filt[1], fir_design='firwin', skip_by_annotation='edge',verbose=False)
            eeg_data["cl_eyes"].append(raw)

        if task[0]:
            t = Task()
            for i in task1:
                tsk1 = glob.glob(pat+i)[0]
                raw = mne.io.read_raw_edf(tsk1,verbose=False,preload=True)

                raw.rename_channels(lambda x: x.strip('.'))
                raw.set_montage(montage, match_case=False)

                events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3),verbose=False)
                picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,exclude='bads')
                raw.filter(filt[0], filt[1], fir_design='firwin', skip_by_annotation='edge',verbose=False)
                epochs = mne.Epochs(raw, events, event_id = dict(T1=2, T2=3), tmin=tmin, tmax=tmax,  
                                    proj=True, picks=picks, baseline=None, preload=True,verbose=False)
                t(epochs)

            eeg_data["t1"].append(t)

        if task[1]:
            t = Task()
            for i in task2:
                tsk2 = glob.glob(pat+i)[0]
                raw=mne.io.read_raw_edf(tsk2,verbose=False,preload=True)

                raw.rename_channels(lambda x: x.strip('.'))
                raw.set_montage(montage, match_case=False)

                events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3),verbose=False)
                picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,exclude='bads')
                raw.filter(filt[0], filt[1], fir_design='firwin', skip_by_annotation='edge',verbose=False)
                epochs = mne.Epochs(raw, events, event_id=dict(T1=2, T2=3), tmin=tmin, tmax=tmax, 
                                    proj=True, picks=picks, baseline=None, preload=True,verbose=False)
                t(epochs)

            eeg_data["t2"].append(t)

        if task[2]:
            t = Task()
            for i in task3:
                tsk3 = glob.glob(pat+i)[0]
                raw=mne.io.read_raw_edf(tsk3,verbose=False,preload=True)

                raw.rename_channels(lambda x: x.strip('.'))
                raw.set_montage(montage, match_case=False)

                events, _ = mne.events_from_annotations(raw, event_id = dict(T1=2, T2=3),verbose=False)
                picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,exclude='bads')
                raw.filter(filt[0], filt[1], fir_design='firwin', skip_by_annotation='edge',verbose=False)
                epochs = mne.Epochs(raw, events, event_id=dict(T1=2, T2=3), tmin=tmin, tmax=tmax, 
                                    proj=True, picks=picks, baseline=None, preload=True,verbose=False)
                t(epochs)

            eeg_data["t3"].append(t)

        if task[3]:
            t = Task()
            for i in task4:
                tsk4 = glob.glob(pat+i)[0]
                raw=mne.io.read_raw_edf(tsk4,verbose=False,preload=True)

                raw.rename_channels(lambda x: x.strip('.'))
                raw.set_montage(montage, match_case=False)

                events, _ = mne.events_from_annotations(raw, event_id = dict(T1=2, T2=3),verbose=False)
                picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,exclude='bads')
                raw.filter(filt[0], filt[1], fir_design='firwin', skip_by_annotation='edge',verbose=False)
                epochs = mne.Epochs(raw, events, event_id=dict(T1=2, T2=3), tmin=tmin, tmax=tmax, 
                                    proj=True, picks=picks, baseline=None, preload=True,verbose=False)
                t(epochs)

            eeg_data["t4"].append(t)
                
    return eeg_data