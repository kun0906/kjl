"""

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx

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
	root_dir = 'examples/offline/report/out/src_dst/results-20210928'

	for device in ['Neon', 'RSPI', 'Nano']:
		lg.info(f'\n\n***{device}, {root_dir}')
		in_file = os.path.join(root_dir, f'{device}/IAT+SIZE-header_False/gather.csv')
		lg.debug(in_file)
		_speedup.main(in_file, FEATURES=['IAT+SIZE'], HEADERS=[False])
		in_file = os.path.join(root_dir, f'{device}/STATS-header_True/gather.csv')
		lg.debug(in_file)
		_speedup.main(in_file, FEATURES=['STATS'], HEADERS=[True])

	main(root_dir, feature='IAT+SIZE', header=False)
	main(root_dir, feature='STATS', header=True)
