import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.signal import find_peaks
import torch


def generateSignals(data,fs = 5000, wlen = 10, overlap = 5):
    wlen = wlen*fs
    overlap = (overlap*fs)/wlen
    totalLength = len(data["scg"])

    for start in range(0, totalLength, int((1-overlap)*wlen)):
        yield data["scg"][start:start+wlen], data["ecg1"][start:start+wlen], data["rpeak1"][(data["rpeak1"] >=start) & (data["rpeak1"] <=start + wlen )] - start, data["ecg2"][start:start+wlen], data["rpeak2"][(data["rpeak2"] >=start) & (data["rpeak2"] <=start + wlen )] - start

def distanceTransform(signal, rpeaks):
    length = len(signal)
    transform = []
    lower = rpeaks[0]
    for j in range(0, lower):
        transform.append(abs(lower - j))
    for i in range(1,len(rpeaks)):
        upper = rpeaks[i]
        lower = rpeaks[i-1]
        middle = (upper + lower)/2
        for k in range(lower, upper):
            transform.append(abs(k-lower)) if k < middle else transform.append(abs(k-upper))
    for i in range(upper,length):
        transform.append(abs(i-upper))
    transform = np.array(transform) 
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaledTransform = scaler.fit_transform(transform.reshape((-1,1)))
    
    return scaledTransform

def create_loaders(data_path, inp_file = "data.pt", label_file = "labels.pt", test_size = 0.2, val_size = 0.2, train_batch_size = 64, val_batch_size = 64):
    data = torch.load(os.path.join(data_path,inp_file))
    target = torch.load(os.path.join(data_path,label_file))
    x_train, x_val, y_train, y_val = train_test_split(data, target, random_state = 42, test_size = val_size + test_size)
    x_val,x_test, y_val,y_test = train_test_split(x_val,y_val, random_state = 32, test_size = (test_size/(test_size + val_size)))
    train, val, test = TensorDataset(x_train, y_train), TensorDataset(x_val, y_val), TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train, batch_size=train_batch_size, shuffle =False, num_workers = 4, pin_memory = True)
    val_loader = DataLoader(val, batch_size = val_batch_size, shuffle = False,num_workers = 4, pin_memory = True)
    test_loader = DataLoader(test, batch_size = 1 , shuffle = False)
    
    return train_loader, val_loader, test_loader

def get_shape(loader):
    for x,y in loader:
        return x.shape
    
    
def infer(model, inp, prominence = 0.3, distance = 625,smoothen = True, downsampling_factor = 10):
    model.cuda()
    model.eval()
    inp = inp[:,0,:].view(1, 1, inp.shape[-1]).cuda()
    with torch.no_grad():
        pred = model(inp)
    if smoothen:
        out=smooth(pred.cpu().detach().view(pred.shape[-1]).numpy())
    else:
        out = pred.cpu().detach().view(pred.shape[-1]).numpy()
    if (downsampling_factor!=1):
        downsampled = out.flatten()[0::downsampling_factor]
    else:
        downsampled = out.flatten()
    valley_loc_downsampled,_ = getValleys(downsampled, prominence = prominence,distance = max(1,distance//downsampling_factor))
    return out,valley_loc_downsampled*downsampling_factor

def getValleys(signal, prominence, distance ):
    signal = signal*-1
    valley_loc, _ = find_peaks(signal, prominence = prominence,distance = distance)
    return valley_loc,_

def smooth(signal,window_len=50):
    y = pd.DataFrame(signal).rolling(window_len,center = True, min_periods = 1).mean().values.reshape((-1,))
    return y

def evaluate_window(actual, detected, fs = 5000, tolerance = 75):

    tolerance = (tolerance/1000)*fs
    grouped_missed = []
    FP= 0
    matched_beats = []    
    correct = 0
    for correctPeak in actual:
        matched = detected[np.where(abs(correctPeak - detected) < tolerance)[0]]
        try:
            assert len(matched) == 1
            correct+=1
            matched_beats.append(matched[0])                        
        except AssertionError:
            if len(matched) > 1:
                FP+= len(matched) - 1
            else:
                matched = [np.NaN]
                matched_beats.append(np.NaN)
        temp = np.asarray([correctPeak, matched[0]])
        grouped_missed.append(temp)

    grouped_missed = np.asarray(grouped_missed)
    matched_beats = np.asarray(matched_beats)
    matched_interbeat_intervals = np.diff(matched_beats)
    matched_interbeat_intervals = matched_interbeat_intervals[~np.isnan(matched_interbeat_intervals)]
    matched_IBI_SD = np.diff(matched_interbeat_intervals*1000/fs)
    matched_RMSSD = rms = np.sqrt(np.mean(matched_IBI_SD**2))
    matched_NN50 = len(np.where(matched_IBI_SD>50)[0])
    matched_pNN50 = matched_NN50/ len(matched_interbeat_intervals)
    matched_mIBI = matched_interbeat_intervals.mean()*1000/fs 
    matched_SDNN = matched_interbeat_intervals.std()*1000/fs
    actual_interbeat_intervals = np.diff(actual)
    actual_IBI_SD = np.diff(actual_interbeat_intervals*1000/fs)
    actual_RMSSD = rms = np.sqrt(np.mean(actual_IBI_SD**2))
    actual_NN50 = len(np.where(actual_IBI_SD>50)[0])
    actual_pNN50 = actual_NN50/ len(actual_interbeat_intervals)
    actual_mIBI = actual_interbeat_intervals.mean()*1000/fs
    actual_SDNN = actual_interbeat_intervals.std()*1000/fs

    metrics = {
        "Total Positives": len(actual),
        "Total Detected": len(detected),
        "True Positives": correct,
        "False Positivies": len(detected) - correct,
        "Missed": len(actual) - correct,
        "Actual Mean Inter Beat Interval" : actual_mIBI,
        "Detected Mean Inter Beat Interval": matched_mIBI,
        "Actual Standard Deviation of Intervals": actual_SDNN,
        "Detected Standard Deviation of Intervals": matched_SDNN,
        "Actual pNN50" : actual_pNN50, 
        'Detected pNN50': matched_pNN50,
        'Actual RMSSD' :  actual_RMSSD, 
        'Detected RMSSD': matched_RMSSD,
        'Sensitivity' : correct/(correct + (len(actual) - correct)),
        'PPV' : correct/(correct + (len(detected) - correct))
    }

    return metrics