import numpy as np
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def generateSignals(data,fs = 5000, wlen = 10, overlap = 5):
    wlen = wlen*fs
    overlap = (overlap*fs)/wlen
    totalLength = len(data["scg"])

    for start in range(0, totalLength, int((1-overlap)*wlen)):
        yield data["resp"][start:start+wlen],data["scg"][start:start+wlen], data["ecg1"][start:start+wlen], data["rpeak1"][(data["rpeak1"] >=start) & (data["rpeak1"] <=start + wlen )] - start, data["ecg2"][start:start+wlen], data["rpeak2"][(data["rpeak2"] >=start) & (data["rpeak2"] <=start + wlen )] - start

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