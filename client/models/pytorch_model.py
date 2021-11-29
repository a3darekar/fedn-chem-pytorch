from torch import nn
import torch
import torch.nn.functional as F


# Create an initial CNN Model
def create_seed_model():
	exp_config = yaml.load(open("bootstrap_13C.yaml", 'r'), Loader=yaml.FullLoader)
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
		index =1#jump molid
		vect_feat = torch.zeros([1,128, 38], dtype=torch.float32)
		for n in range(128):
			for o in range(38):
				vect_feat[0,n,o]=x[index]
				index=index+1
		adj=torch.zeros([1,4,128,128], dtype=torch.float32)
		for first in range(4):
			for second in range(128):
				for third in range(128):
					adj[0,first,second,third]=x[index]
					index=index+1
		input_mask=torch.zeros([1,128], dtype=torch.int32)
		input_mask[0,int(x[index].item())]=1
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