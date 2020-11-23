import torch
import torch.nn as nn


class CassavaModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.conv = nn.Conv2d(3, 128, kernel_size=5, stride=4)
		self.norm = nn.BatchNorm2d(128)
		self.pool = nn.MaxPool2d(4)
		self.hidden1 = nn.Linear(128*15*15, 64)
		self.hidden2 = nn.Linear(64, 64)
		self.hidden3 = nn.Linear(64, 5)
		self.dropout = nn.Dropout(0.3)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()

	def forward(self, x):
		x = self.conv(x)
		x = self.relu(x)
		x = self.pool(self.norm(x))
		x = x.view(x.shape[0], -1)
		x = self.relu(self.hidden1(x))
		x = self.relu(self.hidden2(x))
		x = self.dropout(x)
		x = self.softmax(self.hidden3(x))
		return x
		# print(x)

if __name__ == '__main__':
	model = CassavaModel()
	# print(model)

	x = torch.ones((1, 3, 256, 256))
	# print(x.shape)
	print(model(x))