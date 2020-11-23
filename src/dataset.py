import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms

import cv2
import numpy as np

import os
from config import *

class CassavaDataset(Dataset):

	def __init__(self, dataframe, transform=None, dataset=False, device=None):
		self.df = dataframe
		self.transform = transform
		self.dataset = dataset
		self.device = device

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):

		img_name = self.df.image_id.values[idx]	
		img_path = os.path.join(TRAIN_PATH, img_name)

		img = cv2.imread(img_path)
		img = cv2.resize(img, (256, 256))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img / 255
		img = torch.from_numpy(img)
		img = img.permute(2, 0, 1)


		labels = self.df.loc[idx, ['label_0', 'label_1', 'label_2', 'label_3', 'label_4']].values
		labels = torch.from_numpy(labels.astype(np.int8))
		labels = labels.unsqueeze(-1)

		return img, labels

if __name__ == '__main__':

	fold = 0
	df = pd.read_csv('./input/train_folds.csv')

	train_df = df[df.kfold != fold][:5].reset_index(drop=True)
	valid_df = df[df.kfold == fold].reset_index(drop=True)

	train_dataset = CassavaDataset(train_df)
	valid_dataset = CassavaDataset(valid_df)

	train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
	vaid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)