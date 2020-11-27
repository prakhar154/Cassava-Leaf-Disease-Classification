import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from model import CassavaModel
from loss import DenseCrossEntropy
import dataset
from config import *


def train_one_fold(fold, model, optimizer):
	df = pd.read_csv('./input/train_ohe.csv')

	train_df = df[df.kfold != fold].reset_index(drop=True)
	valid_df = df[df.kfold == fold].reset_index(drop=True)

	train_dataset = dataset.CassavaDataset(train_df, device=DEVICE)
	valid_dataset = dataset.CassavaDataset(valid_df, device=DEVICE)

	train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
	vaid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)

	device = torch.device(DEVICE)

	

	criterion = DenseCrossEntropy()

	train_fold_results = []

	for epoch in range(EPOCHS):
		model.train()
		t_loss = 0

		for step, batch in enumerate(train_dataloader):
			img = batch[0]
			label = batch[1]

			img = img.to(DEVICE, dtype=torch.float)
			label = label.to(DEVICE, dtype=torch.float)

			outputs = model(img)
			# print(f'outputs \n {outputs}')
			loss = criterion(outputs, label.squeeze(-1))
			loss.backward()

			t_loss += loss.item()

			optimizer.step()
			optimizer.zero_grad()

		model.eval()
		val_loss = 0
		val_preds = None
		val_labels = None


		for step, batch in enumerate(vaid_dataloader):
			img = batch[0]
			label = batch[1]

			if val_labels is None:
				val_labels = label.clone().squeeze(-1)
			else:
				val_labels = torch.cat((val_labels, label.squeeze(-1)), dim=0)

			img = img.to(DEVICE, dtype=torch.float)
			label = label.to(DEVICE, dtype=torch.float)

			with torch.no_grad():
				outputs = model(img)

				loss = criterion(outputs, label.squeeze(-1))
				val_loss += loss.item()

				preds = torch.softmax(outputs, dim=1).data.cuda()

				if val_preds is None:
					val_preds = preds
				else:
					val_preds = torch.cat((val_preds, preds), dim=0)
		val_preds = torch.argmax(val_preds, dim=1)

		print(f'EPOCH : {epoch}, train_loss: {t_loss}, valid_loss: {val_loss}')

		train_fold_results.append({
			'fold': fold,
			'epoch': epoch,
			'train_loss': t_loss / len(train_dataloader),
			'valid_loss': val_loss / len(vaid_dataloader)
			})

	return val_preds, train_fold_results



def k_fold_train(folds):

	model = CassavaModel()
	model.to(DEVICE)

	plist = [{'params':model.parameters(), 'lr':5e-5}]
	optimizer = optim.Adam(plist)

	df = pd.read_csv('./input/train_ohe.csv')
	oof_preds = np.zeros((df.shape[0]))
	train_results = []

	for i in range(folds):
		valid_idx = df[df.kfold == i].index

		val_preds, train_fold_results = train_one_fold(i, model, optimizer)
		oof_preds[valid_idx] = val_preds.numpy()

		train_results += train_fold_results

		torch.save({
			'fold': i,
			'lr': optimizer.state_dict()['params_groups'][0]['lr'],
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict()
			}, f'./model/baseline/val_loss {train_results[i].val_loss}.pth')


if __name__ == '__main__':
	k_fold_train(5)

	


