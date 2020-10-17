'''infer'''
import os
import sys
import signal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from glob import glob
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import argparse

from model import SeismoNet
from utils import *

def main(args):
	
	
	if args.create_test_file:
		print ("Creating Test File... ")
		data_path = os.path.join(args.data_path, args.file_type)
		__, _, test_loader = create_loaders(data_path, "data.pt","labels.pt")
	
	else:
		print ("Loading Test File... ")
		test_tensor = torch.load(args.test_tensor_file)
		test_dataset = TensorDataset(test_tensor)
		test_loader = DataLoader(test_dataset, batch_size = 1, pin_memory = True)
	
	print ("Loading Model... ")
	model = SeismoNet(get_shape(test_loader))
	model.load_state_dict(torch.load(args.best_model)["model"])
	window_info  = []
	metrics = []
	if not(os.path.exists("results/")):
		os.mkdir("results/")
	
	for i,x in enumerate(test_loader):
		
		if len(x) > 1:
			pred_distance_transform, pred_peak_locations = infer(model, x[0] ,downsampling_factor = 1)
			print (pred_distance_transform, pred_peak_locations)

		else:
			pred_distance_transform, peak_locations = infer(model, x, downsampling_factor = 1)
			
		if args.evaluate:
			assert len(x)>1
			actual_peak_locations = np.where(x[1] == 0.0)[0]  #provide actual rpeak locations as array 
			metrics.append(evaluate_window(actual_peak_locations, pred_peak_locations))
		
		if args.save_figures:
			if not(os.path.exits("results/figures")):
				os.mkdir("results/figures")
			plt.figure(figsize = [10,5])
			plt.subplot(1,2,1)
			plt.plot(x[0].cpu().numpy().flatten())
			plt.subplot(1,2,2)
			plt.plot(pred_distance_transform.flatten())
			plt.plot(x[1].cpu().numpy().flatten())
			plt.scatter(pred_peak_locations, pred_distance_transform.flatten()[pred_peak_locations])
			plt.savefig("results/figures/{}.png".format(i+1))
		
		window_info.append(pred_peak_locations)
	metrics = pd.DataFrame(metrics)
	metrics.to_csv("results/results.csv")
		

		
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--create_test_file", action ="store_true", help = "Create test file if not already present")
	parser.add_argument('--test_tensor_file',nargs="?" , help = 'Path to saved files directory')
	parser.add_argument('--data_path',nargs="?", const = "saved_data/", default = "saved_data/", help = 'Path to saved files directory')
	parser.add_argument('--file_type',nargs="?", const = "b", default = "b", help = "file type")
	parser.add_argument('--best_model',nargs="?", const = "best_model/best_model_pretrained.pt", default = "best_model/best_model_pretrained.pt", help = "Best Model File")
	parser.add_argument('--evaluate', action = "store_true", help = "Compare against label or not")
	parser.add_argument('--save_figures', action = "store_true", help = "save figure along with results")
	args = parser.parse_args()

	main(args)
	
	