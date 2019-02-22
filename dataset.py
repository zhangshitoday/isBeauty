'''
Function:
	load the train data.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize


'''load data'''
class ImageFolder(Dataset):
	def __init__(self, imagespath, labpath, shape=(350, 350), is_shuffle=True, mode='train'):
		self.img_shape = shape
		self.imagepaths = sorted(glob.glob(os.path.join(imagespath, '*.*')))
		if mode == 'train':
			self.imagepaths = self.imagepaths[:int(len(self.imagepaths) * 0.8)]
		elif mode == 'test':
			self.imagepaths = self.imagepaths[int(len(self.imagepaths) * 0.8):]
		else:
			raise ValueError('ImageFolder --> mode should be <train> or <test>, not %s...' % mode)
		if is_shuffle:
			random.shuffle(self.imagepaths)
		ratings = pd.read_excel(labpath)
		filenames = ratings.groupby('Filename').size().index.tolist()
		self.labels = []
		for filename in filenames:
			score = round(ratings[ratings['Filename'] == filename]['Rating'].mean(), 2)
			self.labels.append({'Filename': filename, 'score': score})
		self.labels = pd.DataFrame(self.labels)
	def __getitem__(self, index):
		# Image
		img_path = self.imagepaths[index % len(self.imagepaths)]
		img = np.array(Image.open(img_path)) / 255.
		input_img = resize(img, (*self.img_shape, 3), mode='reflect')
		input_img = np.transpose(input_img, (2, 0, 1))
		input_img = torch.from_numpy(input_img).float()
		# Label
		filename = img_path.split('/')[-1]
		label = self.labels[self.labels.Filename == filename].score.values
		return img_path, input_img, label
	def __len__(self):
		return len(self.imagepaths)
