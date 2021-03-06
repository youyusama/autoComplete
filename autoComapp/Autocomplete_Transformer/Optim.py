import torch
import torch.nn as nn
from torch.autograd import Variable

# =============================================================================
#
# Optimizer : 优化器 使用Adam优化器
#
# =============================================================================

class NoamOpt:
	"实现学习率的优化包装器"

	def __init__(self, model_size, factor, warmup, optimizer):
		self.optimizer = optimizer
		self._step = 0
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		self._rate = 0

	def step(self):
		"Update parameters and rate"
		self._step += 1
		rate = self.rate()
		for p in self.optimizer.param_groups:
			p['lr'] = rate
		self._rate = rate
		self.optimizer.step()

	def rate(self, step=None):
		"实现学习率 lrate"
		if step is None:
			step = self._step
		return self.factor * \
		       (self.model_size ** (-0.5) *
		        min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
	return NoamOpt(model.src_embed[0].d_model, 2, 4000,
	               torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# =============================================================================
#
# Regularization : 正则化
#
# =============================================================================
class LabelSmoothing(nn.Module):
	"实现平滑标签"
	#即将one-hot编码中的0改成很小的数，1改成接近1的数
	def __init__(self, size, padding_idx, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.criterion = nn.KLDivLoss(reduction='sum')
		self.padding_idx = padding_idx
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.size = size
		self.true_dist = None

	def forward(self, x, target):
		assert x.size(1) == self.size
		true_dist = x.data.clone()
		true_dist.fill_(self.smoothing / (self.size - 2))
		true_dist.scatter_(1, target.long().data.unsqueeze(1), self.confidence)
		true_dist[:, self.padding_idx] = 0
		mask = torch.nonzero(target.data == self.padding_idx)
		if mask.dim() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		self.true_dist = true_dist
		return self.criterion(x, Variable(true_dist, requires_grad=False))