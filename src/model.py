import torch
import torch.nn as nn
import torchvision.models as models


class CassavaModel(nn.Module):

	def __init__(self):
		super().__init__()
		model = models.resnet18()
		model.fc = nn.Sequential(
			nn.Linear(512, 5),
			nn.Softmax())
		self.model = model

	def forward(self, x):
		x = self.model(x)
		return x

if __name__ == '__main__':
	model = CassavaModel()
	# print(model)

	x = torch.ones((1, 3, 256, 256))
	# print(x.shape)
	print(model(x))