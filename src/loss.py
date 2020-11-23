import torch
import torch.nn.functional as F
import torch.nn as nn

class DenseCrossEntropy(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, logits, labels):
		logits = logits.float()
		labels = labels.float()

		logprobs = F.log_softmax(logits, dim=-1)
		loss = labels * logprobs
		loss = loss.sum(-1)

		return loss.mean()


if __name__ == '__main__':
	logits = torch.tensor((1, 2, 3, 4, 5))
	labels = torch.tensor((1, 1, 1, 1, 1))

	criterion = DenseCrossEntropy()
	print(criterion.forward(logits, labels))