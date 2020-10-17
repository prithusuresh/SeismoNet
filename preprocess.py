import os

import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import argparse
from glob import glob

from utils import *
from dataloader import CEBSDataset


def main(args):

	file_type = args.file_type
	data_path = args.data_path
	
	if not(os.path.exists("saved_data")):
		print ("Creating Saved Data Path")
		os.mkdir("saved_data")
	if not(os.path.exists("saved_data/{}".format(file_type))):
		os.mkdir("saved_data/{}".format(file_type))
	if not(os.path.exists("saved_data/{}/pickle_files".format(file_type))):
		os.mkdir("saved_data/{}/pickle_files".format(file_type))
		
	files = sorted(glob(os.path.join(data_path,file_type+"*.dat")))
	
	for i in tqdm(files, total = len(files)):
		
		i = i.rstrip(".dat")
		[x,info] = wfdb.rdsamp(i)
		ann = wfdb.io.rdann(i,'atr')
		all_peaks = ann.sample

		subjectWise_dict ={"rpeak1": all_peaks[::2],
						   "rpeak2": all_peaks[1::2],
						   "resp": x[:,2].flatten(),
						   "scg": x[:,3].flatten(),
						   "ecg1":x[:,0].flatten(),
						   "ecg2":x[:,1].flatten(),
						}
		with open("saved_data/{}/pickle_files/{}.pkl".format(file_type,i.split("/")[-1]), "wb") as f:
			pickle.dump(subjectWise_dict,f)
			
		wlen = args.wlen
		overlap = args.overlap
		fs = args.fs
		
		
		generator = generateSignals(subjectWise_dict, fs, wlen, overlap)
		
		scgSig = []
		ecg1Sig = []
		ecg2Sig = []
	
		groundTruth1 = []
		groundTruth2 = []
		for scg,ecg1,rpeak1,ecg2,rpeak2 in generator:
			if ecg1.shape[0] != wlen*fs or ecg2.shape[0] != wlen*fs or scg.shape[0] != wlen*fs or rpeak1 is None or rpeak2 is None:
				continue
			transform1 = distanceTransform(ecg1, rpeak1)

			transform2 = distanceTransform(ecg2, rpeak2)


			scgSig.append(scg.reshape((1,-1)))

			ecg1Sig.append(ecg1.reshape((1,-1)))
			ecg2Sig.append(ecg2.reshape((1,-1)))


			groundTruth1.append(transform1.reshape((1,-1)))
			groundTruth2.append(transform2.reshape((1,-1)))

		inputSig_t = torch.tensor(scgSig).type(torch.float)
		ecg1Sig_t = torch.tensor(ecg1Sig).type(torch.float)
		ecg2Sig_t = torch.tensor(ecg2Sig).type(torch.float)

		
		ecg12Sig_t = torch.cat((ecg1Sig_t, ecg2Sig_t),1)

		groundTruth1_t = torch.tensor(groundTruth1).type(torch.float)
		groundTruth2_t = torch.tensor(groundTruth2).type(torch.float)
		saving_path = 'saved_data/{}/preprocessed_data/'.format(file_type) 
		
		if not(os.path.exists(saving_path)):
				os.mkdir(saving_path)

			
		p_no = int(i.split("/")[2].split(".")[0].lstrip(file_type))
		torch.save(inputSig_t, saving_path+"inputSig_{}.pt".format(p_no))
		torch.save(groundTruth1_t, saving_path+"groundTruth1_{}.pt".format(p_no))
		torch.save(groundTruth2_t, saving_path+"groundTruth2_{}.pt".format(p_no))
		torch.save(ecg12Sig_t,saving_path+"ecg12_{}.pt".format(p_no))
		
	print("--Saving Data--")
	data = CEBSDataset(os.path.join("saved_data/", file_type))
	torch.save(data.input, os.path.join("saved_data/", file_type, "data.pt"))
	torch.save(data.ground, os.path.join("saved_data/", file_type, "labels.pt"))
		
		
		
		

if __name__ =="__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_type',  nargs='?',type = str, default= "b", help = 'm, p or b')
	parser.add_argument('--data_path', nargs = '?', type = str, default = "../files/", help= "path to data files")
	parser.add_argument('--wlen', nargs = '?', type = int, default = 10, help= "window length in seconds")
	parser.add_argument('--overlap', nargs = '?', type = int, default = 5, help= "overlap length in seconds")
	parser.add_argument('--fs', nargs = '?', type = int, default = 5000, help= "sampling frequency")

					
	args = parser.parse_args()
	main(args)