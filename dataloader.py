import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from glob import glob

class CEBSDataset(Dataset):
	def __init__(self, data_path, ecg_channel = 1):
		
		
		
		self.data_path = data_path #---"saved_data/b" 
		self.input = []
		self.ground = []
		
		gt_file_suffix = "groundTruth{}_".format(ecg_channel)
		p_files = sorted(glob(os.path.join(data_path, "preprocessed_data","inputSig_*.pt")))
		
		for inp_file in p_files:
			p_no = inp_file.split(".")[-2].split("_")[-1]
			self.input.append(torch.load(inp_file))
			gt_file_name = '/'.join(inp_file.split("/")[:-1]) +"/"+ gt_file_suffix + str(p_no) + ".pt"
			self.ground.append(torch.load(gt_file_name))

		self.input = torch.cat(self.input).type(torch.float)
		self.ground = torch.cat(self.ground).type(torch.float)
		
	def __len__(self):
		return len(self.mer_input)
	
	def __getitem__(self,idx):
		
		label = self.ground[idx]
		input_tensor = self.input[idx]
		return input_tensor,label