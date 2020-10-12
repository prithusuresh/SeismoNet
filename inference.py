import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models, datasets
from tqdm import tqdm_notebook as tqdm
from torch import optim
import torch.nn.functional as F
import torchvision
import wfdb 
from tqdm.notebook import tqdm
from scipy.signal import find_peaks,peak_prominences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.quantization import QuantStub, DeQuantStub
from scipy.stats import pearsonr
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.transforms as transforms


def infer(model, inp, prominence = 0.5, distance = 1,smoothen = True, downsampling_factor = 10 ):
	model.cuda()
	model.eval()
	inp = inp[:,0,:].view(1, 1, inp.shape[-1]).cuda()
#     inp = inp[:,0,:].view(1, 1, inp.shape[-1])
	with torch.no_grad():
		pred = model(inp)
	if smoothen:
		out=smooth(pred.cpu().detach().view(pred.shape[-1]).numpy())
#         out=smooth(pred.view(pred.shape[-1]).numpy())
	else:
		out = pred.cpu().detach().view(pred.shape[-1]).numpy()
#         out = pred.view(pred.shape[-1]).numpy()
	
	if (downsampling_factor!=1):
		downsampled = out.flatten()[0::downsampling_factor]
	else:
		downsampled = out.flatten()
#     print(out.shape, downsampled.shape)
	
#     valley_loc_og,xxx = getValleys(out, prominence = prominence,distance = distance)

	valley_loc_downsampled,xxx = getValleys(downsampled, prominence = prominence,distance = distance//downsampling_factor)
#     torch.cuda.empty_cache()
	
#     plt.figure(figsize = [20,10])
#     plt.subplot(1,2,1)
#     plt.scatter(valley_loc_og,out[valley_loc_og],c = "g")
#     plt.plot(out)
# #     plt.plot(smooth(out))
#     plt.subplot(1,2,2)
#     plt.scatter(valley_loc_downsampled, downsampled[valley_loc_downsampled],c = "b")
#     plt.plot(downsampled);
#     plt.show()
	return out,valley_loc_downsampled*downsampling_factor,xxx

def getValleys(signal, prominence, distance ):
	signal = signal*-1
#     print("signal shape {} TYPE {}".format(signal.shape,type(signal)))
	valley_loc,xxx = find_peaks(signal, prominence = prominence,distance = distance)
	
	
	return valley_loc,xxx

def smooth(signal,window_len=50):
	y = pd.DataFrame(signal).rolling(window_len,center = True, min_periods = 1).mean().values.reshape((-1,))
	return y




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