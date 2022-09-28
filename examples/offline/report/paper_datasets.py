"""

"""
import pandas as pd

# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
from examples.offline._constants import *  # should in the top.
from kjl.utils.tool import timer, check_path

# DATASETS = ['DUMMY']
FEATURES = ['IAT+SIZE']
HEADER = [False]
TUNING = [False, True]


@timer
def get_dataset_details(gather_file='.txt', out_dir='', delimeter='&'):
	""" collect all individual results together

	Parameters
	----------
	in_dir:
		search results from the given directory
	out_dir:
		save the gathered results to the given directory
	Returns
	-------
	out_file:
		the short csv for a quick overview
	"""

	df = pd.read_csv(gather_file, sep=',', header=None)
	res = []
	for dataset in DATASETS:
		for line in df.values:
			if dataset == line[0]:
				shape = str(line[9])
				dim = str(line[10])
				res.append([dataset, shape, dim])
				break

	# Save all results to
	out_file = os.path.join(out_dir, 'dataset_detail.txt')
	check_path(out_file)
	with open(out_file, 'w') as f:
		f.write('\\toprule\n')
		# dataset
		f.write('Dataset & ' + f' {delimeter} '.join([vs[0] for vs in res]) + '\\\\ \n')
		f.write('\\midrule\n')
		# shape
		f.write('Train Set & ' + f' {delimeter} '.join([vs[1].split('|')[0] for vs in res]) + '\\\\ \n')
		f.write('\\midrule\n')
		f.write('Val Set & ' + f' {delimeter} '.join([vs[1].split('|')[1] for vs in res]) + '\\\\ \n')
		f.write('\\midrule\n')
		f.write('Test Set & ' + f' {delimeter} '.join([vs[1].split('|')[2] for vs in res]) + '\\\\ \n')
		f.write('\\midrule\n')
		# dim
		f.write('Dim &' + f' {delimeter} '.join([vs[-1] for vs in res]) + '\\\\ \n')
		f.write('\\bottomrule\n')

	return out_file


if __name__ == '__main__':
	out_dir = r'examples/offline/report/out/'
	in_file = f'examples/offline/out/src_dst/results/2021-09-24/gather.csv'
	out_file = get_dataset_details(in_file, out_dir)
	lg.info(f'{out_file} exists: {os.path.exists(out_file)}')
