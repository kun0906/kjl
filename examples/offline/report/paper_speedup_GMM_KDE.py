"""

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import traceback

import pandas as pd

from examples.offline._constants import *
from examples.offline.report import _speedup
from examples.offline.report._speedup import get_one_result


def baseline_table(neon_file, rspi_file, nano_file, out_file, header=False, feature='IAT+SIZE'):
	neon_df = pd.read_csv(neon_file, header=None)
	rspi_df = pd.read_csv(rspi_file, header=None)
	nano_df = pd.read_csv(nano_file, header=None)
	model = 'OCSVM(rbf)'
	res = {}
	for tuning in TUNINGS:
		tmp = []
		for dataset in DATASETS:
			neon_result = get_one_result(neon_df, dataset, feature, f'header_{header}', model, f'tuning_{tuning}')
			rspi_result = get_one_result(rspi_df, dataset, feature, f'header_{header}', model, f'tuning_{tuning}')
			nano_result = get_one_result(nano_df, dataset, feature, f'header_{header}', model, f'tuning_{tuning}')
			tmp.append([neon_result, rspi_result, nano_result])
		res[f'tuning_{tuning}'] = tmp

	with open(out_file, 'w') as f:
		for tuning, tmp in res.items():
			f.write(f'\n\n{tuning}, {feature}, header_{header}\n')
			f.write('\\toprule\n')
			f.write('\multicolumn{2}{|c|}{Dataset}    & UNB & CTU  & MAWI & MACCDC & SFRIG  & AECHO   &DWSHR  \\\\ \n')
			f.write(f'\midrule\n')
			# AUC
			f.write('\multicolumn{2}{|c|}{AUC} & ' + ' & '.join([vs[8] for vs, _, _ in tmp]).
			        replace('+/-', '$\pm$') + '\\\\ \n')
			f.write(f'\midrule\n')
			# Train time
			f.write(r'\multicolumn{2}{|c|}{\C{Server Train \\Time (ms)}} & ' + ' & '.
			        join([vs[5] for vs, _, _ in tmp]).replace('+/-', '$\pm$') + '\\\\ \n')
			f.write(f'\midrule\n')
			# Test time (RSPI)
			f.write('\multirow{3}{*}{\C{\hspace{-0.15cm}Test Time\\\\ (ms)}} & \C{RSPI} & ' +
			        ' & '.join([vs[7] for _, vs, _ in tmp]).replace('+/-', '$\pm$') + '\\\\ \n')
			f.write('\cmidrule{2-9}\n')
			f.write('&\C{NANO} & ' + ' & '.join([vs[7] for _, _, vs in tmp]).replace('+/-', '$\pm$') + '\\\\ \n')
			f.write('\cmidrule{2-9}\n')
			f.write('&\C{Server}  & ' + ' & '.join([vs[7] for vs, _, _ in tmp]).replace('+/-', '$\pm$') + '\\\\ \n')
			f.write(f'\midrule\n')
			# Space
			f.write('\multicolumn{2}{|c|}{\C{Space (kB)}} & ' + ' & '.
			        join([vs[9] for vs, _, _ in tmp]).replace('+/-', '$\pm$') + '\\\\ \n')
			f.write(f'\\bottomrule\n')


def speedup_table(neon_file, rspi_file, nano_file, out_file, header=False, feature='IAT+SIZE'):
	neon_df = pd.read_csv(neon_file, header=None)
	rspi_df = pd.read_csv(rspi_file, header=None)
	nano_df = pd.read_csv(nano_file, header=None)
	res = {}
	for tuning in TUNINGS:
		tmp = []
		for dataset in DATASETS:
			for model in MODELS:
				neon_result = get_one_result(neon_df, dataset, feature, f'header_{header}', model, f'tuning_{tuning}')
				rspi_result = get_one_result(rspi_df, dataset, feature, f'header_{header}', model, f'tuning_{tuning}')
				nano_result = get_one_result(nano_df, dataset, feature, f'header_{header}', model, f'tuning_{tuning}')
				tmp.append([neon_result, rspi_result, nano_result])
		res[f'tuning_{tuning}'] = tmp

	with open(out_file, 'w') as f:
		for tuning, tmp in res.items():
			for covariance_type in ['full']:  # paper only use the 'full' result. ['full', 'diag']:
				f.write(f'\n\n\n***{tuning}, {covariance_type}, {feature}, header_{header}\n')
				# OC-KJL Table (i.e., KJL-GMM(''))
				f.write(f'\n\n***********************************************************************************\n')
				f.write(f'{tuning}, {covariance_type}, {feature}, header_{header},  OC-KJL Table\n')
				# Test time speedup (Neon)
				f.write('\\toprule\n')
				f.write(r'\multicolumn{2}{|c|}{Dataset}    & UNB      & CTU  & MAWI         & MACCDC    '
				        r'   & SFRIG  & AECHO   &DWSHR \\' + '\n')
				f.write(f'\midrule\n')
				f.write(r'\multirow{3}{*}{\C{\hspace{-0.15cm}Test Time\\Speedup}} &  \C{RSPI} & ' + ' & '.
				        join([vs[7] for _, vs, _ in tmp
				              if f'KJL-GMM({covariance_type})' == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\cmidrule{2-9}\n')
				f.write('& \C{NANO} & ' + ' & '.join([vs[7] for _, _, vs in tmp if f'KJL-GMM({covariance_type})'
				                                      == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\cmidrule{2-9}\n')
				f.write('&\C{Server} & ' + ' & '.join([vs[7] for vs, _, _ in tmp if f'KJL-GMM({covariance_type})'
				                                       == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write(f'\midrule\n')
				f.write('\multicolumn{2}{|c|}{\C{Space Reduction}} & ' + ' & '.join(
					[vs[9] for vs, _, _ in tmp if f'KJL-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\bottomrule\n')

				# OC-Nystrom Table (i.e., Nystrom-GMM('full'))
				f.write(f'\n\n***********************************************************************************\n')
				f.write(f'{tuning}, {covariance_type}, {feature}, header_{header},  OC-Nystrom Table\n')
				# Test time speedup (Neon)
				f.write('\\toprule\n')
				f.write(
					r'\multicolumn{2}{|c|}{Dataset}    & UNB      & CTU  & MAWI         & MACCDC    '
					r'   & SFRIG  & AECHO   &DWSHR \\' + '\n')
				f.write(f'\midrule\n')
				f.write(r'\multirow{3}{*}{\C{\hspace{-0.15cm}Test Time\\Speedup}} &  \C{RSPI} & ' + ' & '.join(
					[vs[7] for _, vs, _ in tmp if f'Nystrom-GMM({covariance_type})'
					 == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\cmidrule{2-9}\n')
				f.write('& \C{NANO} & ' + ' & '.join([vs[7] for _, _, vs in tmp if f'Nystrom-GMM({covariance_type})'
				                                      == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\cmidrule{2-9}\n')
				f.write('&\C{Server} & ' + ' & '.join([vs[7] for vs, _, _ in tmp if f'Nystrom-GMM({covariance_type})'
				                                       == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write(f'\midrule\n')
				f.write('\multicolumn{2}{|c|}{\C{Space Reduction}} & ' + ' & '.join(
					[vs[9] for vs, _, _ in tmp if f'Nystrom-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\bottomrule\n')

				# OC-KJL-QS Table (i.e., KJL-GMM(''))
				f.write(f'\n\n***********************************************************************************\n')
				f.write(f'{tuning}, {covariance_type}, {feature}, header_{header},  OC-KJL-QS Table\n')
				# Test time speedup (Neon)
				f.write('\\toprule\n')
				f.write(r'\multicolumn{2}{|c|}{Dataset}    & UNB      & CTU  & MAWI         & MACCDC      '
				        r' & SFRIG  & AECHO   &DWSHR \\' + '\n')
				f.write(f'\midrule\n')
				f.write(r'\multirow{3}{*}{\C{\hspace{-0.15cm}Test Time\\Speedup}} &\C{RSPI} & ' + ' & '.join(
					[vs[7] for _, vs, _ in tmp if f'KJL-QS-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\cmidrule{2-9}\n')
				f.write('&\C{NANO} & ' + ' & '.join([vs[7] for _, _, vs in tmp if f'KJL-QS-GMM({covariance_type})'
				                                     == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\cmidrule{2-9}\n')
				f.write('&\C{Server} & ' + ' & '.join([vs[7] for vs, _, _ in tmp if f'KJL-QS-GMM({covariance_type})'
				                                       == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write(f'\midrule\n')
				f.write('\multicolumn{2}{|c|}{\C{Space Reduction}} & ' + ' & '.
				        join([vs[9] for vs, _, _ in tmp if f'KJL-QS-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\bottomrule\n')

				# OC-Nystrom-QS Table
				f.write(f'\n\n***********************************************************************************\n')
				f.write(f'{tuning}, {covariance_type}, {feature}, header_{header}, OC-Nystrom-QS Table\n')
				# Test time speedup (Neon)
				f.write('\\toprule\n')
				f.write(
					r'\multicolumn{2}{|c|}{Dataset}    & UNB      & CTU  & MAWI         & MACCDC   '
					r'    & SFRIG  & AECHO   &DWSHR \\' + '\n')
				f.write(f'\midrule\n')
				f.write(r'\multirow{3}{*}{\C{\hspace{-0.15cm}Test Time\\Speedup}} &\C{RSPI} & ' + ' & '.join(
					[vs[7] for _, vs, _ in tmp if f'Nystrom-QS-GMM({covariance_type})' == vs[3]])
				        .replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\cmidrule{2-9}\n')
				f.write('&\C{NANO} & ' + ' & '.join([vs[7] for _, _, vs in tmp if f'Nystrom-QS-GMM({covariance_type})'
				                                     == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\cmidrule{2-9}\n')
				f.write('&\C{Server} & ' + ' & '.join([vs[7] for vs, _, _ in tmp if f'Nystrom-QS-GMM({covariance_type})'
				                                       == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write(f'\midrule\n')
				# Space reduction
				f.write('\multicolumn{2}{|c|}{\C{Space Reduction}} & ' + ' & '.join(
					[vs[9] for vs, _, _ in tmp if f'Nystrom-QS-GMM({covariance_type})' == vs[3]]).
				        replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\bottomrule\n')

				f.write(f'\n\n***********************************************************************************\n')
				# OC-KJL, OC-KJL-QS vs OC-Nystrom, OC-Nystrom-QS
				f.write(f'retrained AUC and Train time speedup: {tuning}, {covariance_type}, {feature}, '
				        f'header_{header}, OC-KJL, OC-KJL-QS vs OC-Nystrom, OC-Nystrom-QS\n')

				f.write('\\toprule\n')
				f.write(
					r'\diagbox[width=7em,height=2.5em]{\vspace{0.cm}\hspace{-0.15cm}Method}{\vspace{8.cm}\hspace{8.cm}Dataset}  '
					r'  & UNB   & CTU  '
					r'& MAWI & MACCDC & SFRIG  & AECHO   &DWSHR   \\' + '\n')
				f.write(f'\midrule\n')
				# OC-KJL
				# Retrained AUC (Neon)
				f.write(r'\C{\hspace{-1.2cm}OC-KJL:\\\hspace{0.0cm}\uline{AUC Retained}}  & ' + ' & '.
				        join([vs[8] for vs, _, _ in tmp if f'KJL-GMM({covariance_type})'
				              == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\cmidrule{2-8}\n')
				# Retrained AUC (Neon)
				f.write('\C{Train Speedup} & ' + ' & '.join([vs[5] for vs, _, _ in tmp if f'KJL-GMM({covariance_type})'
				                                             == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\midrule\n')
				# OC-KJL-QS
				f.write(r'\C{\hspace{-.7cm}OC-KJL-QS:\\\hspace{-0.1cm}\uline{AUC Retained}} & ' + ' & '.
				        join([vs[8] for vs, _, _ in tmp if f'KJL-QS-GMM({covariance_type})'
				              == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\cmidrule{2-8}\n')
				f.write(
					'\C{Train Speedup} & ' + ' & '.join([vs[5] for vs, _, _ in tmp if f'KJL-QS-GMM({covariance_type})'
					                                     == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\midrule\n')
				f.write('\\hline\n')
				f.write('\\midrule\n')
				# OC-Nystrom
				f.write(r'\C{\hspace{-.55cm}OC-Nystr\"om:\\\hspace{-0.1cm}\uline{AUC Retained}} & ' + ' & '.
				        join([vs[8] for vs, _, _ in tmp if f'Nystrom-GMM({covariance_type})'
				              == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\cmidrule{2-8}\n')
				f.write('\C{Train Speedup} & ' + ' & '.join([vs[5] for vs, _, _ in tmp
				                                             if f'Nystrom-GMM({covariance_type})'
				                                             == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\midrule\n')
				# OC-Nystrom-QS
				f.write(r'\C{\hspace{-0.05cm}OC-Nystr\"om-QS:\\\hspace{-0.1cm}\uline{AUC Retained}} & '
				        + ' & '.join([vs[8] for vs, _, _ in tmp if f'Nystrom-QS-GMM({covariance_type})'
				                      == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\cmidrule{2-8}\n')
				f.write('\C{Train Speedup} & ' + ' & '.join([vs[5] for vs, _, _ in tmp if
				                                             f'Nystrom-QS-GMM({covariance_type})'
				                                             == vs[3]]).replace('+/-', '$\pm$') + '\\\\ \n')
				f.write('\\bottomrule\n')

		# # Test time speedup (Neon)
		# f.write('Neon Test time & ' + ' & '.join([vs[7] for vs, _, _ in tmp if f'KJL-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		# f.write('RSPI Test time & ' + ' & '.join([vs[7] for _, vs, _ in tmp if f'KJL-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		# f.write('Nano Test time & ' + ' & '.join([vs[7] for _, _, vs in tmp if f'KJL-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		# # Space reduction
		# f.write(
		# 	'Neon space & ' + ' & '.join([vs[9] for vs, _, _ in tmp if f'KJL-GMM({covariance_type})' == vs[3]]).
		# 	replace('+/-', '$\pm$') + '\n')
		#
		# f.write('Neon Test time & ' + ' & '.join([vs[7] for vs, _, _ in tmp if f'KJL-QS-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		# f.write('RSPI Test time & ' + ' & '.join([vs[7] for _, vs, _ in tmp if f'KJL-QS-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		# f.write('Nano Test time & ' + ' & '.join([vs[7] for _, _, vs in tmp if f'KJL-QS-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		# # Space reduction
		# f.write('Neon space & ' + ' & '.join(
		# 	[vs[9] for vs, _, _ in tmp if f'KJL-QS-GMM({covariance_type})' == vs[3]]).
		#         replace('+/-', '$\pm$') + '\n')
		#
		# f.write('Neon Time time & ' + ' & '.join([vs[7] for vs, _, _ in tmp if f'Nystrom-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		# f.write('RSPI Time time & ' + ' & '.join([vs[7] for _, vs, _ in tmp if f'Nystrom-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		# f.write('Nano Time time & ' + ' & '.join([vs[7] for _, _, vs in tmp if f'Nystrom-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		# # Space reduction
		# f.write('Neon space & ' + ' & '.join(
		# 	[vs[9] for vs, _, _ in tmp if f'Nystrom-GMM({covariance_type})' == vs[3]]).
		#         replace('+/-', '$\pm$') + '\n')
		#
		# f.write('Neon Test time & ' + ' & '.join([vs[7] for vs, _, _ in tmp if
		#                                           f'Nystrom-QS-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		# f.write('RSPI Test time & ' + ' & '.join([vs[7] for _, vs, _ in tmp if
		#                                           f'Nystrom-QS-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		# f.write('Nano Test time & ' + ' & '.join([vs[7] for _, _, vs in tmp if
		#                                           f'Nystrom-QS-GMM({covariance_type})'
		#                                           == vs[3]]).replace('+/-', '$\pm$') + '\n')
		#
		# # Space reduction
		# f.write('Neon space & ' + ' & '.join([vs[9] for vs, _, _ in tmp if f'Nystrom-QS-GMM({covariance_type})'
		#                                       == vs[3]]).replace('+/-', '$\pm$') + '\n')


def main(root_dir, feature='IAT+SIZE', header=False):
	###################################################################################################
	# 2. format results
	neon_file = os.path.join(root_dir, f'Neon/{feature}-header_{header}/gather-speedup.csv')
	rspi_file = os.path.join(root_dir, f'RSPI/{feature}-header_{header}/gather-speedup.csv')
	nano_file = os.path.join(root_dir, f'Nano/{feature}-header_{header}/gather-speedup.csv')

	baseline_file = os.path.join(root_dir, f'{feature}-header_{header}-baseline.txt')
	baseline_table(neon_file, rspi_file, nano_file, baseline_file, header, feature)
	lg.debug(f'baseline: {baseline_file}, {os.path.exists(baseline_file)}')

	speedup_file = os.path.join(root_dir, f'{feature}-header_{header}-speedup.txt')
	speedup_table(neon_file, rspi_file, nano_file, speedup_file, header, feature)
	lg.debug(f'speedup: {speedup_file}, {os.path.exists(speedup_file)}')


if __name__ == '__main__':
	is_iot_device = False
	if is_iot_device:
		root_dir = 'examples/offline/report/out/src_dst/results-20210928'
		root_dir = 'examples/offline/report/out/src_dst/results-20220905'
		root_dir = 'examples/offline/report/out/src_dst/results-KDE_GMM-20220916'
		date_str = '2022-09-16 12:32:22.738044'
		for device in ['RSPI', 'Nano']:
			lg.info(f'\n\n***{device}, {root_dir}')
			for n_normal_max_train in [1000, 3000, 5000, 8000, 10000]:  # 1000, 3000, 5000, 10000
				try:
					lg.info(f'\n\n***{device}, {root_dir}')
					in_file = os.path.join(root_dir, f'{device}/train_size_{n_normal_max_train}/src_dst/results/{date_str}/IAT+SIZE-header_False/gather-all.csv')
					lg.debug(in_file)
					_speedup.main(in_file, FEATURES=['IAT+SIZE'], HEADERS=[False])
					# in_file = os.path.join(root_dir, f'{device}/STATS-header_True/gather.csv')
					# lg.debug(in_file)
					# _speedup.main(in_file, FEATURES=['STATS'], HEADERS=[True])


				except Exception as e:
					traceback.print_exc()

		# main(root_dir, feature='IAT+SIZE', header=False)
		# main(root_dir, feature='STATS', header=True)

	else:
		root_dir = 'examples/offline/report/out/src_dst/results-20220905'
		root_dir = 'examples/offline/report/out/src_dst/results-KDE_GMM-20220916'
		# After deployment and copy the result ('examples/offline/deployment/out/src_dst/results') to 'examples/offline/report/out/src_dst/'
		for device in ['RSPI', 'MacOS', 'NANO']:
			if device == 'RSPI':
				date_str = '2022-09-17 08:48:25.295151'
			elif device =='NANO':
				date_str = '2022-09-16 12:34:44.607093'
			else:
				continue
			df_all = []
			for n_normal_max_train in [1000, 3000, 5000, 8000, 10000]:  # 1000, 3000, 5000, 10000
				lg.info(f'\n\n***{device}, {root_dir}')
				# in_file = os.path.join(root_dir, f'{device}/deployed_results/IAT+SIZE-header_False/gather-all.csv')
				if device == 'MacOS':
					in_file = os.path.join(root_dir, f'{device}/deployed_results2/IAT+SIZE-header_False/gather-all.csv')
					continue
				else:
					in_file = os.path.join(root_dir, f'{device}/train_size_{n_normal_max_train}/src_dst/results/{date_str}/IAT+SIZE-header_False/gather-all.csv')
				lg.debug(in_file)
				_speedup.main(in_file, FEATURES=['IAT+SIZE'], HEADERS=[False])
				# in_file = os.path.join(root_dir, f'{device}/deployed_results/STATS-header_True/gather.csv')
				# lg.debug(in_file)
				# _speedup.main(in_file, FEATURES=['STATS'], HEADERS=[True])

				in_file = os.path.splitext(in_file)[0] + '-speedup.csv'
				df = pd.read_csv(in_file)
				# only contains the full covariance
				df_full = df[(df.iloc[:, 4] == 'tuning_True') & (~df.iloc[:, 3].str.contains('diag'))].iloc[:,
				          [0, 3, 5, 7, 8, 9]].T     # get needed columns
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

				# df_diag = df[(df.iloc[:, 4] == 'tuning_True') & (~df.iloc[:, 3].str.contains('full'))].iloc[:,
				#           [0, 3, 5, 7, 8, 9]].T
				# df_diag.columns = sum([[v] * 9 for v in ['UNB', 'CTU', 'MAWI', 'MACCDC', 'SFRIG', 'AECHO', 'DSHWR']],
				#                       [])  # flatten a nested list
				# df = pd.concat([df_full, df_diag], axis=0)
				df = df_full
				gmm_file = os.path.splitext(in_file)[0] + '-speedup-gmm.csv'
				df.to_csv(gmm_file, sep=',', encoding='utf-8', index=False)
				print(gmm_file)

				one_row = [''] * df_full.shape[1]
				one_row[45] = f'train_size_{n_normal_max_train}'
				# one_row = np.asarray(one_row)
				# one_row.reshape((1, -1))
				if len(df_all) == 0:
					df_all = pd.DataFrame(columns=df_full.columns)

				df_all.loc[len(df_all)] = one_row
				df_all = pd.concat([df_all, df_full], axis=0)


			# save all in one
			gmm_file = os.path.join(root_dir, f'{device}/speedup-gmm-all_train_sizes.csv')
			df_all.to_csv(gmm_file, sep=',', encoding='utf-8', index=False)
			print(gmm_file)
			# main(root_dir, feature='IAT+SIZE', header=False)
			# main(root_dir, feature='STATS', header=True)
