from fedn.utils.pytorchhelper import PytorchHelper
from models.pytorch_model import create_seed_model
import collections

def weights_to_np(weights):

	weights_np = collections.OrderedDict()
	for w in weights:
		weights_np[w] = weights[w].cpu().detach().numpy()
	return weights_np

if __name__ == '__main__':
	import yaml

	with open('settings.yaml', 'r') as fh:
		try:
			settings = dict(yaml.safe_load(fh))
		except yaml.YAMLError as e:
			raise(e)

	# Create a seed model and push to Minio
	model, _, _ = create_seed_model(settings)
	outfile_name = "../initial_model/initial_model.npz"
	helper = PytorchHelper()
	helper.save_model(weights_to_np(model.state_dict()), outfile_name)

