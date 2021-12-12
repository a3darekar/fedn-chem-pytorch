import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
#import util
import yaml
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def create_seed_model(exp_config):
	net_params = exp_config['net_params']
	net_name = exp_config['net_name']

	net_params['g_feature_n'] = 38
	net_params['GS'] = 4

	loss_params = exp_config['loss_params']
	loss = create_loss(loss_params, False)
	
	model = eval(net_name)(**net_params)
	
	opt_params = exp_config['opt_params']
	optimizer = create_optimizer(opt_params, model.parameters())
	return model, loss, optimizer





class GraphVertConfigBootstrapWithMultiMax(nn.Module):
	def __init__(self, g_feature_n=-1, g_feature_out_n=None, 
				 int_d = None, layer_n = None, 
				 mixture_n = 5,
				 mixture_num_obs_per=1,
				 resnet=True, 
				 gml_class = 'GraphMatLayers',
				 gml_config = {}, 
				 init_noise=1e-5,
				 init_bias = 0.0, agg_func=None, GS=1, OUT_DIM=1, 
				 input_norm='batch', out_std= False, 
				 resnet_out = False, resnet_blocks = (3, ), 
				 resnet_d = 128,
				 resnet_norm = 'layer',
				 resnet_dropout = 0.0, 
				 inner_norm=None, 
				 out_std_exp = False, 
				 force_lin_init=False, 
				 use_random_subsets=True):
		
		"""
		GraphVertConfigBootstrap with multiple max outs
		"""
		if layer_n is not None:
			g_feature_out_n = [int_d] * layer_n

		super( GraphVertConfigBootstrapWithMultiMax, self).__init__()
		self.gml = eval(gml_class)(g_feature_n, g_feature_out_n, 
								   resnet=resnet, noise=init_noise,
								   agg_func=parse_agg_func(agg_func), 
								   norm = inner_norm, 
								   GS=GS,
								   **gml_config)

		if input_norm == 'batch':
			self.input_norm = MaskedBatchNorm1d(g_feature_n)
		elif input_norm == 'layer':
			self.input_norm = MaskedLayerNorm1d(g_feature_n)
		else:
			self.input_norm = None

		self.resnet_out = resnet_out 

		if not resnet_out:
			self.mix_out = nn.ModuleList([nn.Linear(g_feature_out_n[-1], OUT_DIM) for _ in range(mixture_n)])
		else:
			self.mix_out = nn.ModuleList([ResNetRegressionMaskedBN(g_feature_out_n[-1], 
																   block_sizes = resnet_blocks, 
																   INT_D = resnet_d, 
																   FINAL_D=resnet_d,
																   norm = resnet_norm,
																   dropout = resnet_dropout, 
																   OUT_DIM=OUT_DIM) for _ in range(mixture_n)])

		self.out_std = out_std
		self.out_std_exp = False

		self.use_random_subsets = use_random_subsets
		self.mixture_num_obs_per = mixture_num_obs_per
		
		if force_lin_init:
			for m in self.modules():
				if isinstance(m, nn.Linear):
					if init_noise > 0:
						nn.init.normal_(m.weight, 0, init_noise)
					else:
						nn.init.xavier_uniform_(m.weight)
					if m.bias is not None:
						if init_bias > 0:
							nn.init.normal_(m.bias, 0, init_bias)
						else:
							nn.init.constant_(m.bias, 0)

	def forward(self, x):
		batch_size = x.shape[0]
		vect_feat = x[:, 1:4865].reshape(batch_size, 128, 38)
		adj = x[:, 4865:-1].reshape(batch_size, 4, 128, 128)
		input_mask = torch.zeros((batch_size, 128))
		for indx, row in enumerate(input_mask):
			m_id = int(x[indx, -1])
			row[m_id] = 1
		G = adj
		
		BATCH_N, MAX_N, F_N = vect_feat.shape

		if self.input_norm is not None:
			vect_feat = apply_masked_1d_norm(self.input_norm, 
											 vect_feat, 
											 input_mask)
		
		G_features = self.gml(G, vect_feat, input_mask)

		g_squeeze = G_features.squeeze(1)
		g_squeeze_flat =g_squeeze.reshape(-1, G_features.shape[-1])
		
		if self.resnet_out:
			x_1 = [m(g_squeeze_flat, torch.FloatTensor(np.array([1])).reshape(-1)).reshape(BATCH_N, MAX_N, -1) for m in self.mix_out]
		else:
			x_1 = [m(g_squeeze) for m in self.mix_out]
		x_1=x_1[0]
		ret = {'shift_mu' : x_1}
		return ret

def apply_masked_1d_norm(norm, x, mask):
	"""
	Apply one of these norms and do the reshaping
	"""
	F_N = x.shape[-1]
	x_flat = x.reshape(-1, F_N)
	mask_flat = mask.reshape(-1)
	out_flat = norm(x_flat, mask_flat)
	out = out_flat.reshape(*x.shape)
	return out


class NoUncertainLoss(nn.Module):
	"""
	"""
	def __init__(self, norm='l2', scale=1.0, **kwargs):
		super(NoUncertainLoss, self).__init__()
		if norm == 'l2':
			self.loss = nn.MSELoss()
		elif norm == 'huber':
			self.loss = nn.SmoothL1Loss()
		elif 'tukeybw' in norm:
			c = float(norm.split('-')[1])
			self.loss = TukeyBiweight(c)
			
		self.scale = scale

	def __call__(self, res, vert_pred, vert_pred_mask):

		mu = res['shift_mu']
		mask = vert_pred_mask
		assert torch.sum(mask) > 0
		y_masked = vert_pred[mask>0].reshape(-1, 1) * self.scale
		mu_masked = mu[mask > 0].reshape(-1, 1) * self.scale

		return self.loss(y_masked, mu_masked)


def create_optimizer(opt_params, net_params):
	opt_direct_params = {}
	optimizer_name = opt_params.get('optimizer', 'adam') 
	if optimizer_name == 'adam':
		for p in ['lr', 'amsgrad', 'eps', 'weight_decay', 'momentum']:
			if p in opt_params:
				opt_direct_params[p] = opt_params[p]

		optimizer = torch.optim.Adam(net_params, **opt_direct_params)
	elif optimizer_name == 'adamax':
		for p in ['lr', 'eps', 'weight_decay', 'momentum']:
			if p in opt_params:
				opt_direct_params[p] = opt_params[p]

		optimizer = torch.optim.Adamax(net_params, **opt_direct_params)
		
	elif optimizer_name == 'adagrad':
		for p in ['lr', 'eps', 'weight_decay', 'momentum']:
			if p in opt_params:
				opt_direct_params[p] = opt_params[p]

		optimizer = torch.optim.Adagrad(net_params, **opt_direct_params)
		
	elif optimizer_name == 'rmsprop':
		for p in ['lr', 'eps', 'weight_decay', 'momentum']:
			if p in opt_params:
				opt_direct_params[p] = opt_params[p]

		optimizer = torch.optim.RMSprop(net_params, **opt_direct_params)
		
	elif optimizer_name == 'sgd':
		for p in ['lr', 'momentum']:
			if p in opt_params:
				opt_direct_params[p] = opt_params[p]

		optimizer = torch.optim.SGD(net_params, **opt_direct_params)

	return optimizer


def create_loss(loss_params, USE_CUDA):
	loss_name = loss_params['loss_name']

	std_regularize = loss_params.get('std_regularize', 0.01)
	mu_scale = move(torch.Tensor(loss_params.get('mu_scale', [1.0])), USE_CUDA)
	std_scale = move(torch.Tensor(loss_params.get('std_scale', [1.0])), USE_CUDA)

	if loss_name == 'NormUncertainLoss':
		criterion = NormUncertainLoss(mu_scale, 
										   std_scale,
										   std_regularize = std_regularize)
	elif loss_name == 'UncertainLoss':
		criterion = UncertainLoss(mu_scale, 
									   std_scale,
									   norm = loss_params['norm'], 
									   std_regularize = std_regularize, 
									   std_pow = loss_params['std_pow'], 
									   use_reg_log = loss_params['use_reg_log'],
									   std_weight = loss_params['std_weight'])

	elif loss_name == "NoUncertainLoss":
		
		criterion = NoUncertainLoss(**loss_params)
	elif loss_name == "SimpleLoss":
		
		criterion = SimpleLoss(**loss_params)
	elif "EnsembleLoss" in loss_name:
		subloss = loss_name.split("-")[1]
		
		criterion = EnsembleLoss(subloss, **loss_params)
	elif loss_name == "PermMinLoss":
		
		criterion = PermMinLoss(**loss_params)
	elif loss_name == "ReconLoss":
		
		criterion = ReconLoss(**loss_params)
	elif loss_name == "CouplingLoss":
		
		criterion = coupling.CouplingLoss(**loss_params)
	elif loss_name == "DistReconLoss":
		
		criterion = DistReconLoss(**loss_params)
	else:
		raise ValueError(loss_name)


	return criterion


def move(tensor, cuda=False):
	if cuda:
		if isinstance(tensor, nn.Module):
			return tensor.cuda()
		else:
			return tensor.cuda(non_blocking=True)
	else:
		return tensor.cpu()


class GraphMatLayer(nn.Module):
	def __init__(self, C, P , GS=1,  
				 noise=1e-6, agg_func=None, 
				 dropout=0.0, use_bias=True):
		"""
		Pairwise layer -- takes a N x M x M x C matrix
		and turns it into a N x M x M x P matrix after
		multiplying with a graph matrix N x M x M
		
		if GS != 1 then there will be a per-graph-channel 
		linear layer
		"""
		super(GraphMatLayer, self).__init__()

		self.GS = GS
		self.noise=noise

		self.linlayers = nn.ModuleList()
		self.dropout = dropout
		self.dropout_layers = nn.ModuleList()
		for ll in range(GS):
			l = nn.Linear(C, P, bias=use_bias)
			if use_bias:
				l.bias.data.normal_(0.0, self.noise)
			l.weight.data.normal_(0.0, self.noise) #?!
			self.linlayers.append(l)
			if dropout > 0.0:
				self.dropout_layers.append(nn.Dropout(p=dropout))
			
		#self.r = nn.PReLU()
		self.r = nn.ReLU()
		self.agg_func = agg_func
 
	def forward(self, G, x):
		def apply_ll(i, x):
			y = self.linlayers[i](x)
			if self.dropout > 0:
				y = self.dropout_layers[i](y)
			return y

		multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)])
		# this is per-batch-element
		xout = torch.stack([torch.matmul(G[i], multi_x[:, i]) for i in range(x.shape[0])])

		x = self.r(xout)
		if self.agg_func is not None:
			x = self.agg_func(x, dim=1)
		return x

class GraphMatLayerFast(nn.Module):
	def __init__(self, C, P , GS=1,  
				 noise=1e-6, agg_func=None, 
				 dropout=False, use_bias=False, 
				 ):
		"""
		Pairwise layer -- takes a N x M x M x C matrix
		and turns it into a N x M x M x P matrix after
		multiplying with a graph matrix N x M x M
		
		if GS != 1 then there will be a per-graph-channel 
		linear layer
		"""
		super(GraphMatLayerFast, self).__init__()

		self.GS = GS
		self.noise=noise

		self.linlayers = nn.ModuleList()
		for ll in range(GS):
			l = nn.Linear(C, P, bias=use_bias)
			if self.noise == 0:
				if use_bias:
					l.bias.data.normal_(0.0, 1e-4)
				torch.nn.init.xavier_uniform_(l.weight)
			else:
				if use_bias:
					l.bias.data.normal_(0.0, self.noise)
				l.weight.data.normal_(0.0, self.noise) #?!
			self.linlayers.append(l)
			
		#self.r = nn.PReLU()
		self.r = nn.LeakyReLU()
		self.agg_func = agg_func
 
	def forward(self, G, x):
		BATCH_N, CHAN_N,  MAX_N, _ = G.shape
		def apply_ll(i, x):
			y = self.linlayers[i](x)
			return y

		multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
		xout = torch.einsum("ijkl,jilm->jikm", [G, multi_x])
		xout = self.r(xout)
		if self.agg_func is not None:
			xout = self.agg_func(xout, dim=0)
		return xout

class GraphMatLayers(nn.Module):
	def __init__(self, input_feature_n, 
				 output_features_n, resnet=False, GS=1, 
				 norm=None,
				 force_use_bias = False, 
				 noise=1e-5, agg_func=None,
				 layer_class = 'GraphMatLayerFast', 
				 layer_config = {}):
		super(GraphMatLayers, self).__init__()
		
		self.gl = nn.ModuleList()
		self.resnet = resnet

		LayerClass = eval(layer_class)
		for li in range(len(output_features_n)):
			if li == 0:
				gl = LayerClass(input_feature_n, output_features_n[0],
								noise=noise, agg_func=agg_func, GS=GS, 
								use_bias=not norm or force_use_bias, 
								**layer_config)
			else:
				gl = LayerClass(output_features_n[li-1], 
								output_features_n[li], 
								noise=noise, agg_func=agg_func, GS=GS, 
								use_bias=not norm or force_use_bias, 
								**layer_config)
			
			self.gl.append(gl)

		self.norm = norm
		if self.norm is not None:
			if self.norm == 'batch':
				Nlayer = MaskedBatchNorm1d
			elif self.norm == 'layer':
				Nlayer = MaskedLayerNorm1d
			self.bn = nn.ModuleList([Nlayer(f) for f in output_features_n])
			
		
	def forward(self, G, x, input_mask=None):
		for gi, gl in enumerate(self.gl):
			x2 = gl(G, x)
			if self.norm:
				x2 = self.bn[gi](x2.reshape(-1, x2.shape[-1]), 
								 input_mask.reshape(-1)).reshape(x2.shape)

			if self.resnet:
				if x.shape == x2.shape:
					x3 = x2 + x
				else:
					x3 = x2
			else:
				x3 = x2
			x = x3
		

		return x

def parse_agg_func(agg_func):
	if isinstance(agg_func, str):
		if agg_func == 'goodmax':
			return goodmax
		elif agg_func == 'sum':
			return torch.sum
		elif agg_func == 'mean':
			return torch.mean
		else:
			raise NotImplementedError()
	return agg_func

def goodmax(x, dim):
	return torch.max(x, dim=dim)[0]

class GraphMatLayerExpressionWNorm2(nn.Module):
	def __init__(self, C, P, GS=1,  
				 terms = [{'power': 1, 'diag' : False}], 
				 noise=1e-6, agg_func=None, 
				 use_bias=False, 
				 post_agg_nonlin = None,
				 post_agg_norm = None,
				 per_nonlin = None,
				 dropout = 0.0,
				 cross_term_agg_func = 'sum', 
				 norm_by_neighbors=False, 
				 ):
		"""
		"""
	
		super(GraphMatLayerExpressionWNorm2, self).__init__()

		self.pow_ops = nn.ModuleList()
		for t in terms:
			l = GraphMatLayerFastPow2(C, P, GS, 
									 mat_pow = t.get('power', 1), 
									 mat_diag = t.get('diag', False), 
									 noise = noise, 
									 use_bias = use_bias, 
									 nonlin = t.get('nonlin', per_nonlin), 
									 norm_by_neighbors=norm_by_neighbors, 
									 dropout = dropout)
			self.pow_ops.append(l)
			
		self.post_agg_nonlin = post_agg_nonlin
		if self.post_agg_nonlin == 'leakyrelu':
			self.r = nn.LeakyReLU()
		elif self.post_agg_nonlin == 'relu':
			self.r = nn.ReLU()
		elif self.post_agg_nonlin == 'sigmoid':
			self.r = nn.Sigmoid()
		elif self.post_agg_nonlin == 'tanh':
			self.r = nn.Tanh()
			
		self.agg_func = agg_func
		self.cross_term_agg_func = cross_term_agg_func
		self.norm_by_neighbors = norm_by_neighbors
		self.post_agg_norm = post_agg_norm
		if post_agg_norm == 'layer':
			self.pa_norm = nn.LayerNorm(P)

		elif post_agg_norm == 'batch':
			self.pa_norm = nn.BatchNorm1d(P)
			
	def forward(self, G, x):
		BATCH_N, CHAN_N,  MAX_N, _ = G.shape    
		
		terms_stack = torch.stack([l(G, x) for l in self.pow_ops], dim=-1)

		if self.cross_term_agg_func == 'sum':
			xout = torch.sum(terms_stack, dim=-1)
		elif self.cross_term_agg_func == 'max':
			xout = torch.max(terms_stack, dim=-1)[0]
		elif self.cross_term_agg_func == 'prod':
			xout = torch.prod(terms_stack, dim=-1)
		else:
			raise ValueError(f"unknown cross term agg func {self.cross_term_agg_func}")
		
		if self.agg_func is not None:
			xout = self.agg_func(xout, dim=0)

		if self.post_agg_nonlin is not None:
			xout = self.r(xout)
		if self.post_agg_norm is not None:
			xout = self.pa_norm(xout.reshape(-1, xout.shape[-1])).reshape(xout.shape)
			
		return xout

class GraphMatLayerFastPow2(nn.Module):
	def __init__(self, C, P, GS=1,  
				 mat_pow = 1, 
				 mat_diag = False,
				 noise=1e-6, agg_func=None, 
				 use_bias=False, 
				 nonlin = None, 
				 dropout = 0.0, 
				 norm_by_neighbors=False, 
				 ):
		"""
		Two layer MLP 

		"""
		super(GraphMatLayerFastPow2, self).__init__()

		self.GS = GS
		self.noise=noise

		self.linlayers1 = nn.ModuleList()
		self.linlayers2 = nn.ModuleList()
		
		for ll in range(GS):
			l = nn.Linear(C, P)
			self.linlayers1.append(l)
			l = nn.Linear(P, P)
			self.linlayers2.append(l)
		self.dropout_rate = dropout

		if self.dropout_rate > 0:
			self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(GS)])

		#self.r = nn.PReLU()
		self.nonlin = nonlin
		if self.nonlin == 'leakyrelu':
			self.r = nn.LeakyReLU()
		elif self.nonlin == 'sigmoid':
			self.r = nn.Sigmoid()
		elif self.nonlin == 'tanh':
			self.r = nn.Tanh()
		elif self.nonlin is None:
			pass
		else:
			raise ValueError(f'unknown nonlin {nonlin}')
			
		self.agg_func = agg_func
		self.mat_pow = mat_pow
		self.mat_diag = mat_diag

		self.norm_by_neighbors = norm_by_neighbors
 
	def forward(self, G, x):
		BATCH_N, CHAN_N,  MAX_N, _ = G.shape
		def apply_ll(i, x):
			y = F.relu(self.linlayers1[i](x))
			y = self.linlayers2[i](y)
			
			if self.dropout_rate > 0.0:
				y = self.dropout_layers[i](y)
			return y
		Gprod = G
		for mp in range(self.mat_pow -1):
			Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
		if self.mat_diag:
			Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod
		multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
		xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])

		if self.norm_by_neighbors != False:
			G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
			if self.norm_by_neighbors == 'sqrt':
				xout = xout / torch.sqrt(G_neighbors.unsqueeze(-1))
				
			else:
				xout = xout / G_neighbors.unsqueeze(-1)

		if self.nonlin is not None:
			xout = self.r(xout)
		if self.agg_func is not None:
			xout = self.agg_func(xout, dim=0)
		return xout

class MaskedBatchNorm1d(nn.Module):
	def __init__(self, feature_n):
		"""
		Batchnorm1d that skips some rows in the batch 
		"""

		super(MaskedBatchNorm1d, self).__init__()
		self.feature_n = feature_n
		self.bn = nn.BatchNorm1d(feature_n)

	def forward(self, x, mask):
		assert x.shape[0] == mask.shape[0]
		assert mask.dim() == 1
		
		bin_mask = mask > 0
		y_i = self.bn(x[bin_mask])
		y = torch.zeros(x.shape, device=x.device)
		y[bin_mask] = y_i
		return y

class ResNetRegressionMaskedBN(nn.Module):
	def __init__(self, D, block_sizes, INT_D, FINAL_D, 
				 OUT_DIM=1, norm='batch', dropout=0.0):
		super(ResNetRegressionMaskedBN, self).__init__()

		layers = [nn.Linear(D, INT_D)]
		usemask = [False]
		for block_size in block_sizes:
			layers.append(ResNet(INT_D, INT_D, block_size))
			usemask.append(False)

			if dropout > 0.0:
				layers.append(nn.Dropout(dropout))
				usemask.append(False)
			if norm == 'layer':
				layers.append(MaskedLayerNorm1d(INT_D))
				usemask.append(True)
			elif norm == 'batch':
				layers.append(MaskedBatchNorm1d(INT_D))
				usemask.append(True)
		layers.append(nn.Linear(INT_D, OUT_DIM))
		usemask.append(False)
			
		self.layers = nn.ModuleList(layers)
		self.usemask = usemask

	def forward(self, x, mask):
		for l, use_mask in zip(self.layers, self.usemask):
			if use_mask:
				x = l(x, mask)
			else:
				x = l(x)
		return x

class MaskedLayerNorm1d(nn.Module):
	def __init__(self, feature_n):
		"""
		LayerNorm that skips some rows in the batch 
		"""

		super(MaskedLayerNorm1d, self).__init__()
		self.feature_n = feature_n
		self.bn = nn.LayerNorm(feature_n)

	def forward(self, x, mask):
		assert x.shape[0] == mask.shape[0]
		assert mask.dim() == 1
		
		bin_mask = mask > 0
		y_i = self.bn(x[bin_mask])
		y = torch.zeros(x.shape, device=x.device)
		y[bin_mask] = y_i
		return y


class MyDataset(Dataset):
	def __init__(self, root, n_inp):
		self.df = pd.read_csv(root)
		self.data = self.df.to_numpy()
		self.x , self.y = (torch.from_numpy(self.data[:,:n_inp]),
						   torch.from_numpy(self.data[:,n_inp:]))
	def __getitem__(self, idx):
		return self.x[idx, :], self.y[idx,:]
	def __len__(self):
		return len(self.data)

