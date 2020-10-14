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
from scipy.stats import pearsonr
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.transforms as transforms
from sklearn.preprocessing import MinMaxScaler



def infer(model, inp, prominence = 0.5, distance = 1,smoothen = True, downsampling_factor = 10 ):
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
	valley_loc_downsampled,xxx = getValleys(downsampled, prominence = prominence,distance = distance//downsampling_factor)
	return out,valley_loc_downsampled*downsampling_factor,xxx

def getValleys(signal, prominence, distance ):
	signal = signal*-1
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
	scaler = MinMaxScaler()
	scaledTransform = scaler.fit_transform(transform.reshape((-1,1)))
	return scaledTransform

def evaluate(actual, detected,set_no ,tolerance = 75):
	fs = 5000
	tolerance = (tolerance/1000)*5000
	grouped_missed = []
	FP= 0
	matched_beats = []    
	correct=0
	for correctPeak in actual:
		matched = detected[np.where(abs(correctPeak - detected) < tolerance)[0]]
		try:
			assert len(matched) == 1
			correct+=1
			matched_beats.append(matched[0])                        
		except AssertionError:
			if len(matched) > 1:
				FP+= len(matched) - 1
				matched_beats.extend(matched.tolist()) 
			else:
				matched = [np.NaN]
				matched_beats.append(np.NaN)
		temp = np.asarray([correctPeak, matched[0]])
		grouped_missed.append(temp) 
	grouped_missed = np.asarray(grouped_missed)
	matched_beats = np.asarray(matched_beats)
	matched_interbeat_intervals = np.diff(matched_beats)
	matched_interbeat_intervals = matched_interbeat_intervals[~np.isnan(matched_interbeat_intervals)]
	matched_IBI_SD = np.diff(matched_interbeat_intervals/5)
	matched_RMSSD = rms = np.sqrt(np.mean(matched_IBI_SD**2))
	matched_NN50 = len(np.where(matched_IBI_SD>50)[0])
	if(len(matched_interbeat_intervals)==0):
		matched_pNN50 = -1000
	else:    
		matched_pNN50 = matched_NN50/ len(matched_interbeat_intervals)
	matched_mIBI = matched_interbeat_intervals.mean()/5 
	matched_SDNN = matched_interbeat_intervals.std()/5
	actual_interbeat_intervals = np.diff(actual)
	actual_IBI_SD = np.diff(actual_interbeat_intervals/5)
	actual_RMSSD = rms = np.sqrt(np.mean(actual_IBI_SD**2))
	actual_NN50 = len(np.where(actual_IBI_SD>50)[0])
	actual_pNN50 = actual_NN50/ len(actual_interbeat_intervals)
	actual_mIBI = actual_interbeat_intervals.mean()/5
	actual_SDNN = actual_interbeat_intervals.std()/5
	metrics = {
		"Total Positives": len(actual),
		"detected": len(detected),
		"TP": correct,
		"FP": len(detected) - correct,
		"Missed-FN": len(actual) - correct,
		"Mean Inter Beat Interval" : {'actual': actual_mIBI, 'detected': matched_mIBI},
		"Standard Deviation of Intervals": {'actual': actual_SDNN, 'detected': matched_SDNN},
		"pNN50" : {'actual': actual_pNN50, 'detected': matched_pNN50},
		'RMSSD' : {'actual': actual_RMSSD, 'detected': matched_RMSSD},
		'set':set_no,
		'Se' : correct/(correct + (len(actual) - correct)),
		'PPV' : correct/(correct + (len(detected) - correct))
	}
	cm_user = [[correct,(len(detected) - correct)],[(len(actual) - correct),0]] 
	return metrics,cm_user

def HRVIndicesAvg(patientsHRV):
	MIBI = patientsHRV['Mean Inter Beat Interval']
	SDNN = patientsHRV['Standard Deviation of Intervals']
	pNN50 = patientsHRV['pNN50']
	RMSSD = patientsHRV['RMSSD']
	
	actual = []
	detected = []
	for i in RMSSD:
		actual.append(i['actual'])
		detected.append(i['detected'])
	actual = np.asarray(actual)
	detected = np.asarray(detected)
	RMSSD_ecg_mean = actual.mean()
	RMSSD_ecg_std = actual.std()
	RMSSD_scg_mean = detected.mean()
	RMSSD_scg_std = detected.std()
	corr, _ = pearsonr(actual, detected)
	r2_corr = r2_score(actual, detected)
	print("RMSSD corr {} r2_score {}".format(corr,r2_corr))
	print("RMSSD ecg : {} +- {} scg : {} +- {}".format(RMSSD_ecg_mean,RMSSD_ecg_std,RMSSD_scg_mean,RMSSD_scg_std))
	
	actual = []
	detected = []
	for i in SDNN:
		actual.append(i['actual'])
		detected.append(i['detected'])
	actual = np.asarray(actual)
	detected = np.asarray(detected)
	SDNN_ecg_mean = actual.mean()
	SDNN_ecg_std = actual.std()
	SDNN_scg_mean = detected.mean()
	SDNN_scg_std = detected.std()
	corr, _ = pearsonr(actual, detected)
	r2_corr = r2_score(actual, detected)
	print("SDNN corr {} r2 score {} ".format(corr,r2_corr))
	print("SDNN ecg : {} +- {} scg : {} +- {}".format(SDNN_ecg_mean,SDNN_ecg_std,SDNN_scg_mean,SDNN_scg_std))

	actual = []
	detected = []
	for i in pNN50:
		actual.append(i['actual'])
		detected.append(i['detected'])
	actual = np.asarray(actual)
	detected = np.asarray(detected)
	pNN50_ecg_mean = actual.mean()
	pNN50_ecg_std = actual.std()
	pNN50_scg_mean = detected.mean()
	pNN50_scg_std = detected.std()
	corr, _ = pearsonr(actual, detected)
	r2_corr = r2_score(actual, detected)
	print("pNN50 corr {} r2 score {} ".format(corr,r2_corr))
	print("pNN50 ecg : {} +- {} scg : {} +- {}".format(pNN50_ecg_mean,pNN50_ecg_std,pNN50_scg_mean,pNN50_scg_std))

	actual = []
	detected = []
	for i in MIBI:
		actual.append(i['actual'])
		detected.append(i['detected'])
	actual = np.asarray(actual)
	detected = np.asarray(detected)
	MIBI_ecg_mean = actual.mean()
	MIBI_ecg_std = actual.std()
	MIBI_scg_mean = detected.mean()
	MIBI_scg_std = detected.std()
	corr, _ = pearsonr(actual, detected)
	r2_corr = r2_score(actual, detected)
	print("MIBI corr {} r2 score {} ".format(corr,r2_corr))
	print("MIBI ecg : {} +- {} scg : {} +- {}".format(MIBI_ecg_mean,MIBI_ecg_std,MIBI_scg_mean,MIBI_scg_std))

def MeanValues(patientsHRV):
	Se_allData = patientsHRV['Se'].mean()
	PPV_allData = patientsHRV['PPV'].mean()
	Se_std  = patientsHRV['Se'].std()
	PPV_std = patientsHRV['PPV'].std()
	print("Se mean ",Se_allData)
	print("Se std ",Se_std)
	print("PPV mean ",PPV_allData)
	print("PPV std ",PPV_std)

def interpretability(correlation):
	df = pd.DataFrame(correlation)
	r_mean= df['r'].mean()
	rmse_mean = df['rmse'].mean()
	mse_mean = df['mse'].mean()
	mae_mean = df['mae'].mean()
	r_mean_sd= df['r'].std()
	rmse_mean_sd = df['rmse'].std()
	mse_mean_sd = df['mse'].std()
	mae_mean_sd = df['mae'].std()
	print("corr_mean_waveforms {} sd: {}".format(r_mean,r_mean_sd))
	print("rmse_mean {} sd {}".format(rmse_mean,rmse_mean_sd))
	print("mse_mean {} sd {}".format(mse_mean,mse_mean_sd))
	print("mae_mean {} sd {}".format(mae_mean,mae_mean_sd))


def plot_blandaltman(x, y, hrv,agreement=1.96, confidence=.95, figsize=(5, 4),dpi=100, ax=None):
    # Safety check
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == 1 and y.ndim == 1
    assert x.size == y.size
    n = x.size
    mean = np.vstack((x, y)).mean(0)
    diff = x - y
    md = diff.mean()
    sd = diff.std(axis=0, ddof=1)

    # Confidence intervals
    if confidence is not None:
        assert 0 < confidence < 1
        ci = dict()
        ci['mean'] = stats.norm.interval(confidence, loc=md,
                                         scale=sd / np.sqrt(n))
        seLoA = ((1 / n) + (agreement**2 / (2 * (n - 1)))) * (sd**2)
        loARange = np.sqrt(seLoA) * stats.t.ppf((1 - confidence) / 2, n - 1)
        ci['upperLoA'] = ((md + agreement * sd) + loARange,
                          (md + agreement * sd) - loARange)
        ci['lowerLoA'] = ((md - agreement * sd) + loARange,
                          (md - agreement * sd) - loARange)

    # Start the plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot the mean diff, limits of agreement and scatter
    ax.axhline(md, color='#6495ED', linestyle='--')
    ax.axhline(md + agreement * sd, color='coral', linestyle='--')
    ax.axhline(md - agreement * sd, color='coral', linestyle='--')
    ax.scatter(mean, diff, alpha=0.7)

    loa_range = (md + (agreement * sd)) - (md - agreement * sd)
#     print (md + (agreement * sd), md - (agreement * sd))
    print (loa_range)
    offset = (loa_range / 100.0) * 1.5

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    ax.text(0.98, md - offset, 'Mean', ha="right", va="top",
            transform=trans)
#     ax.text(0.98, md - offset, '%.2f' % md, ha="right", va="top",
#             transform=trans)

    ax.text(0.98, md + (agreement * sd) + offset, '+%.2f SD' % agreement,
            ha="right", va="bottom", transform=trans)
#     ax.text(0.98, md + (agreement * sd) - offset,
#             '%.2f' % (md + agreement * sd), ha="right", va="top",
#             transform=trans)

    ax.text(0.98, md - (agreement * sd) + offset, '-%.2f SD' % agreement,
            ha="right", va="bottom", transform=trans)
#     ax.text(0.98, md - (agreement * sd) + offset,
#             '%.2f' % (md - agreement * sd), ha="right", va="bottom",
#             transform=trans)

    if confidence is not None:
        ax.axhspan(ci['mean'][0], ci['mean'][1],
                   facecolor='#6495ED', alpha=0.0)

        ax.axhspan(ci['upperLoA'][0], ci['upperLoA'][1],
                   facecolor='coral', alpha=0.0)

        ax.axhspan(ci['lowerLoA'][0], ci['lowerLoA'][1],
                   facecolor='coral', alpha=0.0)
        print (ci['upperLoA'][0], ci['upperLoA'][1], ci['lowerLoA'][0], ci['lowerLoA'][1])
    # Labels and title
    diff_line = "Difference "+hrv
    mean_line="Mean "+ hrv
    ax.set_ylabel(diff_line)
    ax.set_xlabel(mean_line)
#     ax.set_title('Bland-Altman plot')

    # Despine and trim
    sns.despine(trim=False, ax=ax)

    return ax