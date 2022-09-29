import os
import sys
import traceback

import numpy as np
import sklearn.preprocessing
from loguru import logger as lg
from sklearn.decomposition import PCA

# from examples.offline.report import _speedup
from kjl.models.gmm import GMM
from kjl.utils.tool import load

lg.remove()
lg.add(sys.stdout, level='DEBUG')
print(flush=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Args():

	def __init__(self, params):
		for k, v in params.items():
			# self.args.k = v
			setattr(self, k, v)

def correlation(X, title=''):
	df = pd.DataFrame(X)
	corr = df.corr()

	kot = corr[abs(corr) >= .8]
	print(np.nanmin(kot.values), np.nanmax(kot.values))
	plt.figure(figsize=(12, 8))
	sns.heatmap(kot, cmap="Greens")
	plt.title(title)
	plt.show()

np.set_printoptions(precision=2)

def parse_result(n_normal_max_train=1000, verbose = 1):
	res = []
	in_dir = 'examples/offline/out/src_dst'
	in_dir = f'examples/offline/out/train_size_{n_normal_max_train}/src_dst'
	# dataset = 'CTU1'
	# dataset = 'MACCDC1'
	dataset = 'SFRIG1_2021'
	# dataset = 'AECHO1_2020'
	# dataset = 'DWSHR_AECHO_2020'

	full_file = f'{in_dir}/{dataset}/IAT+SIZE/header_False/GMM(full)/tuning_True/res.dat'
	# diag_file = f'{in_dir}/{dataset}/IAT+SIZE/header_False/GMM(diag)/tuning_True/res.dat'

	print(full_file)
	full_data = load(full_file)
	# diag_data = load(diag_file)
	diag_data = full_data
	# print(full_data.items())
	aucs = []
	val_aucs = []
	for k in full_data.keys():
		full = full_data[k]
		diag = diag_data[k]
		lg.debug(k)
		if verbose>=5: print('full', full['best']['model']['model'].model.n_components,
		      full['best']['model']['model'].params['GMM_n_components'])
		res.append(full['best']['model']['model'].model.n_components)
		if verbose >=5: print('diag', diag['best']['model']['model'].model.n_components,
		      diag['best']['model']['model'].params['GMM_n_components'])
		for v in ['train', 'val', 'test']:
			if verbose >=5:  print(np.alltrue(full[v][f'X_{v}'] == diag[v][f'X_{v}']),
			      np.alltrue(full[v][f'y_{v}'] == diag[v][f'y_{v}']))

		if verbose >=5: print(f'full_best_auc:', full['best']['score'], f'diag_best_auc:', diag['best']['score'])
		if verbose >= 5: print(f'full_val_auc:', full['val']['val_auc'], f'diag_val_auc:', diag['val']['val_auc'])
		if verbose >=5: print(f'full_test_auc:', full['test']['test_auc'], f'diag_test_auc:', diag['test']['test_auc'])
		aucs.append(full['test']['test_auc'])
		val_aucs.append(full['val']['val_auc'])
		continue
		n_components = full['best']['model']['model'].model.n_components
		for flg in [False, True]:
			print(f'\nReducing features: {flg}')
			for data, cov in [(full, 'full'), (diag, 'diag')]:
				lg.debug(f'***{cov}')
				X_train = data['train']['X_train']
				X_test = data['test']['X_test']
				# print('before std:')
				correlation(X_train, title=f'{cov}, n_components: {n_components}')
				std = sklearn.preprocessing.StandardScaler()
				std.fit(X_train)
				X_train = std.transform(X_train)
				X_test = std.transform(X_test)
				if flg:
					# PCA requires standardized data
					pca = PCA(n_components=0.95)
					pca.fit(X_train)
					print(pca.explained_variance_ratio_)
					X_train = pca.transform(X_train)
					X_test = pca.transform(X_test)
					# print('After PCA:')
					# print(np.corrcoef(X_train))
					correlation(X_train, title=f'{cov}, n_components: {n_components} after PCA')
				lg.debug(f'{cov}, {X_train.shape}, {X_test.shape}')
				params = {
					'is_kjl': False, 'kjl_d': 5, 'kjl_n': 100, 'kjl_q': 0.3,
					'is_nystrom': False, 'nystrom_d': 5, 'nystrom_n': 100, 'nystrom_q': 0.3,
					'is_quickshift': False, 'quickshift_k': 100, 'quickshift_beta': 0.9,
					'random_state': 42,
					'GMM_n_components': n_components,
					'GMM_covariance_type': cov
				}

				gmm = GMM(params)
				gmm.fit(X_train)
				res = gmm.test(X_test, data['test']['y_test'])
				lg.debug(f'res: {res}')

	print(f'n_components: {res}, means: {np.mean(res)}, std: {np.std(res)}')
	print(f'testing aucs: {aucs}, means: {np.mean(aucs)}, std: {np.std(aucs)}')
	print(f'val aucs: {val_aucs}, means: {np.mean(val_aucs)}, std: {np.std(val_aucs)}')
	return res, aucs, val_aucs

def dat2gather_csv(in_file, out_file):
	data = load(in_file)
	print(data)

def _get_line(results):
	lines = []
	for vs in results:
		tmp = []
		try:
			size_ = []
			for set_key in ['train', 'val', 'test']:
				shape_ = vs[5]['0th_repeat'][set_key][f'{set_key}_shape']
				size_.append(shape_[0])
				dim_ = shape_[1]
				auc_ = []
				time_ = []
				model_size_ = []
				for i_repeat in sorted(vs[5].keys()):
					# auc_ = '-'.join([str(vs[5][i_repeat][set_key][f'{set_key}_auc'])  for i_repeat in sorted(vs[5].keys())])
					# time_ = '-'.join([str(vs[5][i_repeat][set_key][f'{set_key}_time'])  for i_repeat in sorted(vs[5].keys())])
					# model_size_ = '-'.join()
					auc_.append(str(vs[5][i_repeat][set_key][f'{set_key}_auc']))
					time_.append(str(vs[5][i_repeat][set_key][f'{set_key}_time']))
					model_size_.append(str(0))
				tmp.append('-'.join(time_) + ',' + '-'.join(auc_))
			size_ = '|'.join([str(v) for v in size_])
			tmp = [size_, str(dim_)] + tmp + ['-'.join(model_size_)]
		except Exception as e:
			traceback.print_exc()
		line = ','.join(vs[:5] + tmp)
		lines.append(line)
	return lines

def parse_kde_result():
	# format all results without kde
	try:
		root_dir = 'examples/offline/report/out/src_dst/results-20220910'
		for device in ['MacOS', 'NANO', 'RASP']:
			lg.info(f'\n\n***{device}, {root_dir}')
			results = []
			in_file = os.path.join(root_dir, f'{device}/training_results/KDE-2022-09-10 22:35:52.173766/res.dat')
			kde_results = load(in_file)
			for dataset, dataset_name in [('CTU-2022-09-03 20:08:46.601527', 'CTU1'), ('MAWI1_2020-2022-09-04 06:47:44.771062', 'MAWI1_2020'),
			                ('MACCDC1-2022-09-04 14:52:54.447067', 'MACCDC1'), ('AECHO1_2020-2022-09-04 06:48:54.703572', 'AECHO1_2020'),
			                ('DWSHR_AECHO_2020-2022-09-04 13:09:53.571742', 'DWSHR_AECHO_2020')]:
				in_file = os.path.join(root_dir, f'{device}/training_results/{dataset}/res.dat')
				data = load(in_file)
				tmp = [vs for vs in kde_results if vs[0] == dataset_name and vs[4] == 'tuning_True'][0]
				data.insert(3, tmp)
				results.extend(data)

			out_file = os.path.join(root_dir, f'{device}/training_results/IAT+SIZE-header_False/gather_detail.csv')
			with open(out_file, 'w') as f:
				for vs in results:
					tmp = []
					try:
						size_ = []
						for set_key in ['train', 'val', 'test']:
							shape_ = vs[5]['0th_repeat'][set_key][f'{set_key}_shape']
							size_.append(shape_[0])
							dim_ = shape_[1]
							auc_ = []
							time_ = []
							model_size_ = []
							for i_repeat in sorted(vs[5].keys()):
								# auc_ = '-'.join([str(vs[5][i_repeat][set_key][f'{set_key}_auc'])  for i_repeat in sorted(vs[5].keys())])
								# time_ = '-'.join([str(vs[5][i_repeat][set_key][f'{set_key}_time'])  for i_repeat in sorted(vs[5].keys())])
								# model_size_ = '-'.join()
								auc_.append(str(vs[5][i_repeat][set_key][f'{set_key}_auc']))
								time_.append(str(vs[5][i_repeat][set_key][f'{set_key}_time']))
								model_size_.append(str(0))
							tmp.append('-'.join(time_) + ',' + '-'.join(auc_))
						size_ = '|'.join([str(v) for v in size_])
						tmp = [size_, str(dim_)] + tmp + ['-'.join(model_size_)]
					except Exception as e:
						traceback.print_exc()
					line = ','.join(vs[:5] + tmp)
					print(line)
					f.write(line + '\n')
			print(out_file)

			in_file = out_file
			lg.debug(in_file)
			_speedup.main(in_file, FEATURES=['IAT+SIZE'], HEADERS=[False])
			# in_file = os.path.join(root_dir, f'{device}/STATS-header_True/gather.csv')
			# lg.debug(in_file)
			# _speedup.main(in_file, FEATURES=['STATS'], HEADERS=[True])

			in_file = os.path.splitext(in_file)[0] + '-speedup.csv'
			df = pd.read_csv(in_file)
			# only contains the full covariance
			df_full = df[(df.iloc[:, 4] == 'tuning_True') & (~df.iloc[:, 3].str.contains('diag'))].iloc[:,
			          [0, 3, 5, 7, 8, 9]].T
			df_full.columns = sum([[v] * 9 for v in ['UNB', 'CTU', 'MAWI', 'MACCDC', 'SFRIG', 'AECHO', 'DSHWR']],
			                      [])  # flatten a nested list
			df_full.iloc[1, :] = sum([[v2 for v, v2 in [('OCSVM(rbf)', 'OCSVM'),
			                                            ('KJL-OCSVM(linear)', 'OC-KJL-SVM(linear)'),
			                                            ('Nystrom-OCSVM(linear)', 'OC-Nystrom-SVM(linear)'),
			                                            ('KDE', 'KDE'),
			                                            ('GMM(full)', 'GMM(full)'),
			                                            ('KJL-GMM(full)', 'OC-KJL'),
			                                            ('Nystrom-GMM(full)', 'OC-Nystrom'),
			                                            ('KJL-QS-GMM(full)', 'OC-KJL-QS'),
			                                            ('Nystrom-QS-GMM(full)', 'OC-Nystrom-QS')]] * 7], [])
			df_diag = df[(df.iloc[:, 4] == 'tuning_True') & (~df.iloc[:, 3].str.contains('full'))].iloc[:,
			          [0, 3, 5, 7, 8, 9]].T
			df_diag.columns = sum([[v] * 9 for v in ['UNB', 'CTU', 'MAWI', 'MACCDC', 'SFRIG', 'AECHO', 'DSHWR']],
			                      [])  # flatten a nested list
			df = pd.concat([df_full, df_diag], axis=0)
			gmm_file = os.path.splitext(in_file)[0] + '-speedup-gmm.csv'
			df.to_csv(gmm_file, sep=',', encoding='utf-8', index=False)
			print(gmm_file)


	except Exception as e:
		lg.error(f'Error: {e}.')

def format_(vs, n_precision=2):
	return [float(f'{v:.{n_precision}f}') for v in vs]

if __name__ == '__main__':

	res = []
	for n_normal_max_train in [1000, 2000, 3000, 4000, 5000]:
		print(f'n_normal_max_train: {n_normal_max_train}')
		n_comps, test_aucs, val_aucs= parse_result(n_normal_max_train)
		res.append((n_comps, test_aucs, val_aucs))

	for i, n_normal_max_train in enumerate([1000, 2000, 3000, 4000, 5000]):
		n_comps, test_aucs, _ = res[i]
		print(n_normal_max_train, n_comps, format_(test_aucs), np.mean(test_aucs), np.std(test_aucs))

	# print('n_comp', [res[i][0] for i in range(len(res))])
	# print('test_auc', [format_(res[i][1]) for i in range(len(res))])
	# print('val_auc', [format_(res[i][2]) for i in range(len(res))])
	# parse_kde_result()
	exit()

	# format all results without kde
	try:
		root_dir = 'examples/offline/report/out/src_dst/results-20220910'
		for device in ['MacOS', 'NANO', 'RASP']:
			lg.info(f'\n\n***{device}, {root_dir}')
			results = []
			for dataset in ['CTU-2022-09-03 20:08:46.601527', 'MAWI1_2020-2022-09-04 06:47:44.771062',
			                'MACCDC1-2022-09-04 14:52:54.447067', 'AECHO1_2020-2022-09-04 06:48:54.703572',
			                'DWSHR_AECHO_2020-2022-09-04 13:09:53.571742']:
				in_file = os.path.join(root_dir, f'{device}/training_results/{dataset}/res.dat')
				data = load(in_file)
				results.extend(data)

			out_file = os.path.join(root_dir, f'{device}/training_results/IAT+SIZE-header_False/gather_detail.csv')
			with open(out_file, 'w') as f:
				for vs in results:
					tmp = []
					try:
						size_ = []
						for set_key in ['train', 'val', 'test']:
							shape_ = vs[5]['0th_repeat'][set_key][f'{set_key}_shape']
							size_.append(shape_[0])
							dim_ = shape_[1]
							auc_ = []
							time_ = []
							model_size_ = []
							for i_repeat in sorted(vs[5].keys()):
								# auc_ = '-'.join([str(vs[5][i_repeat][set_key][f'{set_key}_auc'])  for i_repeat in sorted(vs[5].keys())])
								# time_ = '-'.join([str(vs[5][i_repeat][set_key][f'{set_key}_time'])  for i_repeat in sorted(vs[5].keys())])
								# model_size_ = '-'.join()
								auc_.append(str(vs[5][i_repeat][set_key][f'{set_key}_auc']))
								time_.append(str(vs[5][i_repeat][set_key][f'{set_key}_time']))
								model_size_.append(str(0))
							tmp.append('-'.join(time_) + ',' + '-'.join(auc_))
						size_ = '|'.join([str(v) for v in size_])
						tmp = [size_, str(dim_)] + tmp + ['-'.join(model_size_)]
					except Exception as e:
						traceback.print_exc()
					line = ','.join(vs[:5] + tmp)
					print(line)
					f.write(line + '\n')
			print(out_file)

			in_file = out_file
			lg.debug(in_file)
			_speedup.main(in_file, FEATURES=['IAT+SIZE'], HEADERS=[False])
			# in_file = os.path.join(root_dir, f'{device}/STATS-header_True/gather.csv')
			# lg.debug(in_file)
			# _speedup.main(in_file, FEATURES=['STATS'], HEADERS=[True])

			in_file = os.path.splitext(in_file)[0] + '-speedup.csv'
			df = pd.read_csv(in_file)
			# only contains the full covariance
			df_full = df[(df.iloc[:, 4] == 'tuning_True') & (~df.iloc[:, 3].str.contains('diag'))].iloc[:, [0, 3, 5, 7, 8, 9]].T
			df_full.columns = sum([[v]*8 for v in ['UNB', 'CTU', 'MAWI', 'MACCDC', 'SFRIG', 'AECHO','DSHWR']], []) # flatten a nested list
			df_full.iloc[1, :] = sum([[v2 for v, v2 in [('OCSVM(rbf)', 'OCSVM'),
			                                                 ('KJL-OCSVM(linear)', 'OC-KJL-SVM(linear)'),
			                                                 ('Nystrom-OCSVM(linear)', 'OC-Nystrom-SVM(linear)'),
			                                                 ('GMM(full)', 'GMM(full)'),
			                                                 ('KJL-GMM(full)', 'OC-KJL'),('Nystrom-GMM(full)', 'OC-Nystrom'),
			                                                 ('KJL-QS-GMM(full)', 'OC-KJL-QS'),
			                                                 ('Nystrom-QS-GMM(full)', 'OC-Nystrom-QS')]]*7], [])
			df_diag = df[(df.iloc[:, 4] == 'tuning_True') & (~df.iloc[:, 3].str.contains('full'))].iloc[:, [0, 3, 5, 7, 8, 9]].T
			df_diag.columns = sum([[v]*8 for v in ['UNB', 'CTU', 'MAWI', 'MACCDC', 'SFRIG', 'AECHO','DSHWR']], []) # flatten a nested list
			df = pd.concat([df_full, df_diag], axis=0)
			gmm_file =os.path.splitext(in_file)[0] + '-speedup-gmm.csv'
			df.to_csv(gmm_file, sep=',', encoding='utf-8', index=False)
			print(gmm_file)
		# paper_speedup.main(root_dir, feature='IAT+SIZE', header=False)
		# main(root_dir, feature='STATS', header=True)
		# feature = 'IAT+SIZE'
		# header = False
		# neon_file = rspi_file = nano_file = os.path.join(root_dir, f'{device}/{feature}-header_{header}/gather_detail-speedup.csv')
		# baseline_file = os.path.join(root_dir, f'{device}/{feature}-header_{header}/baseline.txt')
		# baseline_table(neon_file, rspi_file, nano_file, baseline_file, header, feature)
		# lg.debug(f'baseline: {baseline_file}, {os.path.exists(baseline_file)}')
		#
		# speedup_file = os.path.join(root_dir, f'{device}/{feature}-header_{header}/speedup.txt')
		# speedup_table(neon_file, rspi_file, nano_file, speedup_file, header, feature)
		# lg.debug(f'speedup: {speedup_file}, {os.path.exists(speedup_file)}')

	except Exception as e:
		lg.error(f'Error: {e}.')
