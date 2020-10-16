#imports
import pandas as pd
import numpy as np

from inferenceFunctions import HRVIndicesAvg,evaluate,MeanValues,interpretability
from model import Unet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models, datasets
from tqdm import tqdm_notebook as tqdm
from torch import optim
import torch.nn.functional as F
import torchvision
from tqdm.notebook import tqdm 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.transforms as transforms

class CEBSTestDataset(Dataset):
    def __init__(self,patient, test_data_path, fs = 5000, wlen = 10, overlap = 5):
        
        self.wlen = wlen
        self.fs = fs
        self.overlap = overlap
        
        test_data_path = "/hdd/bci/CEBS/processed_data/test/"
        input_tensor_str = test_data_path + "inputSig_"+ str(patient) + ".pt"
        groundTruth_tensor_str = test_data_path+ "groundTruth1_" + str(patient) + ".pt"
        ecg_tensor = test_data_path+ "ecg12_" + str(patient) + ".pt"
        self.mer_input = torch.load(input_tensor_str)[:,0,:].unsqueeze(1)
        self.mer_ground = torch.load(groundTruth_tensor_str)
        self.ecg = torch.load(ecg_tensor)[:,0,:].unsqueeze(1)
        
    def __len__(self):
        return len(self.mer_input)
    
    def __getitem__(self,idx):
        
        label = self.mer_ground[idx]
        input_tensor = self.mer_input[idx]
        ecg = self.ecg[idx]
        return input_tensor,label,ecg


def testingloop():
    PATH = '/hdd/bci/CEBS/Best_model/best_model_onlyScg_newtraintestsplit_5ov.pt' #Original Model
    checkpoint = torch.load(PATH)
    model = Unet((None,1,50000))
    model.load_state_dict(checkpoint['model'])
    correlation = []
    all_patients = []
    patientsHRV = {}
    tolerance = 100
    cm = []
    ds_array = [1]
    set_no = 0
    for ds in ds_array:
        set_no+=1
        for patient in tqdm(range(18)):
                data = CEBSTestDataset(patient)
                loader = DataLoader(data,batch_size=1,shuffle=False)
                final_rpeaks =[]
                val_waveform = []
                scg_waveform = []
                ecg_waveform = []
                output_waveform = []
                actual_promi = []
                countr = 0
                for i,(inp, gt, ecg) in enumerate(loader):
                    countr+=1
                    output, detected_peaks,xxx = infer(model,inp, 0.3, distance = 625, smoothen=True, downsampling_factor=ds)
                    xxx = xxx['prominences']
                    gt = gt.view(gt.shape[-1]).numpy()
                    start = i*(1- (data.overlap/data.wlen))*data.fs*data.wlen
                    end = start + data.wlen*data.fs
                    lower, upper = (data.wlen*data.fs)//4, 3*(data.wlen*data.fs)//4

                    if i==0:
                        xxxx =np.nonzero(detected_peaks < lower)
                        actual_promi.append(xxx[xxxx])
                        final_rpeaks.append(detected_peaks[detected_peaks < lower])
                        val_waveform.append(gt[:lower])
                        ecg_waveform.append(ecg[:,:,:lower].cpu().numpy().reshape((-1,)))
                        scg_waveform.append(inp[:,:,:lower].cpu().numpy().reshape((-1,)))
                        output_waveform.append(output[:lower])

                    xxxx =np.nonzero((detected_peaks >= lower)&(detected_peaks < upper))
                    actual_promi.append(xxx[xxxx])
                    final_rpeaks.append(start + detected_peaks[(detected_peaks >= lower) & (detected_peaks<upper)])
                    val_waveform.append(gt[lower:upper])
                    ecg_waveform.append(ecg[:,:,lower:upper].cpu().numpy().reshape((-1,)))
                    scg_waveform.append(inp[:,:,lower:upper].cpu().numpy().reshape((-1,)))
                    output_waveform.append(output[lower:upper])

                    if i == len(data) - 1:
                        xxxx =np.nonzero(detected_peaks >= upper)
                        actual_promi.append(xxx[xxxx])
                        final_rpeaks.append(detected_peaks[(detected_peaks >= upper)])
                        val_waveform.append(gt[upper:])
                        ecg_waveform.append(ecg[:,:,upper:].cpu().numpy().reshape((-1,)))
                        scg_waveform.append(inp[:,:,upper:].cpu().numpy().reshape((-1,)))
                        output_waveform.append(output[upper:])


                if(len(val_waveform)<1):
                    continue
                val_waveform_ = np.concatenate(val_waveform)
                ecg_waveform_ = np.concatenate(ecg_waveform)
                scg_waveform_ = np.concatenate(scg_waveform)
                output_waveform_ = np.concatenate(output_waveform)
                actual_promi_ = np.array(np.concatenate(actual_promi),dtype = np.float64)
                final_rpeaks_ = np.array(np.concatenate(final_rpeaks), dtype = np.int)
                actual_rpeaks_ = np.where(val_waveform_ == 0)[0]
                identification = str(patient) + "_" + str(ds) 
                patientsHRV[f'{identification}'],cm_user = evaluate(actual_rpeaks_, final_rpeaks_,ds ,tolerance)
                pg_corr_one = pg.corr(output_waveform_,val_waveform_)
                correlation_one_user ={
                    'mse' : mean_squared_error(output_waveform_,val_waveform_),
                    'rmse': np.sqrt(mean_squared_error(output_waveform_,val_waveform_)),
                    'r' : pg_corr_one['r'][0],
                    'mae' : mean_absolute_error(output_waveform_, val_waveform_),   
                }
                correlation.append(correlation_one_user)
           
    patientsHRV = pd.DataFrame(patientsHRV).T
    print("Total {} Detected {} ".format(patientsHRV['Total Positives'].sum(),patientsHRV['detected'].sum()))
    print("TP {} FP {} Missed-FN {}".format(patientsHRV['TP'].sum(),patientsHRV['FP'].sum(),patientsHRV['Missed-FN'].sum()))
    HRVIndicesAvg(patientsHRV)
    MeanValues(patientsHRV)
    interpretability(correlation)
    print(patientsHRV)   