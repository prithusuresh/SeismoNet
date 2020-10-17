import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class IncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size = 15, stride = 1, padding = 7):
	super(IncBlock,self).__init__()

	self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias = False)

	self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = size, stride = stride, padding = padding ),
				   nn.BatchNorm1d(out_channels//4))

	self.conv2 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
				   nn.BatchNorm1d(out_channels//4),
				   nn.LeakyReLU(0.2),
				   nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size +2 , stride = stride, padding = padding + 1),
				   nn.BatchNorm1d(out_channels//4))

	self.conv3 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
				   nn.BatchNorm1d(out_channels//4),
				   nn.LeakyReLU(0.2),
				   nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size + 4 , stride = stride, padding = padding + 2),
				   nn.BatchNorm1d(out_channels//4))

	self.conv4 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
				   nn.BatchNorm1d(out_channels//4),
				   nn.LeakyReLU(0.2),
				   nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size + 6 , stride = stride, padding = padding + 3),
				   nn.BatchNorm1d(out_channels//4))
	
	self.relu = nn.ReLU()

    def forward(self,x):
	
	res = self.conv1x1(x)
	
	c1 = self.conv1(x)
	
	c2 = self.conv2(x)
	
	c3 = self.conv3(x)
	
	c4 = self.conv4(x)
	
	concat = torch.cat((c1,c2,c3,c4),dim = 1)
	
	concat+=res
	
	return self.relu(concat)

class AveragingBlock(nn.Module):

    def __init__(self,in_channels = 1, out_channels = 1):
	
	super(AveragingBlock, self).__init__()
	
	self.conv1 = nn.Conv1d(in_channels,8,3)
	self.bn1 = nn.BatchNorm1d(8)
	
	self.conv2 = nn.Conv1d(8,16,3)
	self.bn2 =nn.BatchNorm1d(16)  
	
	self.conv3 = nn.Conv2d(1,1,(3,3), 2)
	self.bn3 = nn.BatchNorm2d(1)
	
	self.conv4 = nn.Conv2d(1, 1, (3,15), padding = (0,7))
	self.bn4 = nn.BatchNorm2d(1)
	
	self.conv5 = nn.Conv1d(1,out_channels,3, padding = 1)
	self.bn5 = nn.BatchNorm1d(out_channels)
	
	self.relu1 = nn.LeakyReLU(0.2)
	
	self.mp1 = nn.MaxPool1d(2)
	self.mp2 = nn.MaxPool2d((2,2))

    def forward(self, x):
	
	x = self.relu1(self.bn1(self.conv1(x)))
	
	x = self.relu1(self.bn2(self.conv2(x)))
	
	x = x.view(x.shape[0],1,x.shape[1],x.shape[2])
	
	x = self.relu1(self.bn3(self.conv3(x)))
	
	x = self.mp2(x)
	
	x = self.relu1(self.bn4(self.conv4(x)))
	
	x = torch.squeeze(x, dim = 1)
	
	x = self.relu1(self.bn5(self.conv5(x)))
	
	return x

class SeismoNet(nn.Module):
    def __init__(self, shape):
	super(SeismoNet, self).__init__()
	in_channels = 1
	self.cea = nn.Sequential(AveragingBlock())

	self.en1 = nn.Sequential(nn.Conv1d(in_channels, 32, 3, padding = 1),
				 nn.BatchNorm1d(32),
				 nn.LeakyReLU(0.2),
				 nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
				 IncBlock(32,32))

	self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
				 nn.BatchNorm1d(64),
				 nn.LeakyReLU(0.2),
				 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
				 IncBlock(64,64))

	self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
				 nn.BatchNorm1d(128),
				 nn.LeakyReLU(0.2),
				 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
				 IncBlock(128,128))

	self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
				 nn.BatchNorm1d(256),
				 nn.LeakyReLU(0.2),
				 nn.Conv1d(256, 256, 5, stride = 2, padding = 1),
				 IncBlock(256,256))

	self.en5 = nn.Sequential(nn.Conv1d(256,512, 3, padding = 1),
				 nn.BatchNorm1d(512),
				 nn.LeakyReLU(0.2),
				 IncBlock(512,512))


	self.de1 = nn.Sequential(nn.ConvTranspose1d(512,256,1),
				 nn.BatchNorm1d(256),
				 nn.LeakyReLU(0.2),
				 IncBlock(256,256))

	self.de2 =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
				  nn.BatchNorm1d(256),
				  nn.LeakyReLU(0.2),
				  nn.ConvTranspose1d(256,128,3, stride = 2),
				  IncBlock(128,128))

	self.de3 =  nn.Sequential(nn.Conv1d(256,128,3, stride = 1, padding = 1),
				  nn.BatchNorm1d(128),
				  nn.LeakyReLU(0.2),
				  nn.ConvTranspose1d(128,64,3, stride = 2),
				  IncBlock(64,64))

	self.de4 =  nn.Sequential(nn.Conv1d(128,64,3, stride = 1, padding = 1),
				  nn.BatchNorm1d(64),
				  nn.LeakyReLU(0.2),
				  nn.ConvTranspose1d(64,32,3, stride = 2),
				  IncBlock(32,32))
	
	self.de5 = nn.Sequential(nn.Conv1d(64,32,3, stride = 1, padding = 1),
				 nn.BatchNorm1d(32),
				 nn.LeakyReLU(0.2),
				 nn.ConvTranspose1d(32,16,3, stride = 2),
				 IncBlock(16,16))

	self.de6 = nn.Sequential(nn.ConvTranspose1d(16,8,2,stride =2),
				 nn.BatchNorm1d(8),
				 nn.LeakyReLU(0.2))
	
	self.de7 = nn.Sequential(nn.ConvTranspose1d(8,4,2,stride =2),
				 nn.BatchNorm1d(4),
				 nn.LeakyReLU(0.2))
	
	self.de8 = nn.Sequential(nn.ConvTranspose1d(4,2,1,stride =1),
				 nn.BatchNorm1d(2),
				 nn.LeakyReLU(0.2))
	
	self.de9 = nn.Sequential(nn.ConvTranspose1d(2,1,1,stride =1),
				 nn.BatchNorm1d(1),
				 nn.LeakyReLU(0.2))


    def forward(self,x):

	x = self.cea(x)                          #-Convolutional Ensemble Averaging--
	
	x = nn.ConstantPad1d((1,1),0)(x)
	
	e1 = self.en1(x)                         #-----------------------------------
	e2 = self.en2(e1)                        #-----------------------------------
	e3 = self.en3(e2)                        #---------Contracting Path----------
	e4 = self.en4(e3)                        #-----------------------------------
	e5 = self.en5(e4)                        #-----------------------------------
	
	d1 = self.de1(e5)                        #-----------------------------------
	cat = torch.cat([d1,e4],1)               #-----------------------------------
	d2 = self.de2(cat)                       #-----------------------------------
	cat = torch.cat([d2,e3],1)               #-----------------------------------
	d3 = self.de3(cat)                       #----------Expanding Path-----------
	cat = torch.cat([d3[:,:,:-2],e2],1)      #-----------------------------------  
	d4 = self.de4(cat)                       #-----------------------------------
	cat = torch.cat([d4[:,:,:-1],e1],1)      #-----------------------------------  
	d5 = self.de5(cat)[:,:,:-1]              #-----------------------------------
	d6 = self.de6(d5)                        #-----------------------------------
	
	d7 = self.de7(d6)                        #-----------------------------------
	d8 = self.de8(d7)                        #---------Denoising Block-----------
	d9 = self.de9(d8)                        #-----------------------------------
	
	return d9
	