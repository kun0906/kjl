"""

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import itertools
import os.path
import traceback

import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from examples.offline._constants import *
from examples.offline.report import _speedup
from examples.offline.report._speedup import get_one_result
from kjl.utils.tool import check_path


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



def show_diff_train_sizes(ocsvm_gmm, out_file='', title='auc', n_repeats=5):
	import matplotlib.pyplot as plt
	font_size = 15

	colors = ['blue', 'green', 'orange', 'c', 'm', 'b', 'r', 'tab:brown', 'tab:green']
	labels = ['OCSVM', 'GMM', 'KDE']

	sub_dataset = []
	yerrs = []
	data_name = ''
	j = 0
	for i, (train_size, res_) in enumerate(ocsvm_gmm):
		for vs in res_:
			try:
				# vs = vs.split(',')
				data_name = vs[0]
				# tmp = [float(v) for v in str(vs[-3]).split('-')]    # for testing time
				tmp =  [float(v) for v in str(vs[-2]).split('-')]   # for testing AUC
				m_auc, std_auc = np.mean(tmp), np.std(tmp)
				# m_auc = f'{m_auc:.2f}'
				# std_auc = f'{std_auc:.2f}'
				vs = [data_name, vs[3], int(train_size), float(m_auc), float(std_auc)]
			except Exception as e:
				print(e)
				# traceback.print_exc()
				vs = ['0', '0', 0, 0, 0]
				# yerrs.append(0)
			sub_dataset.append(vs)

	df = pd.DataFrame(sub_dataset, columns=['datasets', 'model_name', 'train_size', 'auc', 'std'])
	# g = sns.barplot(y="diff", x='datasets', data=df, hue='repres', ax=axes[t, ind % fig_cols], palette=new_colors)
	print(sub_dataset, 'yerrs:', yerrs)

	mask = df['model_name'] == 'OCSVM(rbf)'
	ocsvm = df[mask]
	j = 0
	plt.errorbar(ocsvm['train_size'].values, ocsvm['auc'].values, ocsvm['std'].values, fmt='*-',
	             capsize=3, color=colors[j], ecolor='tab:red',
	             markersize=8, markerfacecolor='black',
	             label=labels[j])

	mask = df['model_name'] == 'GMM(full)'
	gmm = df[mask]
	j = 1
	plt.errorbar(gmm['train_size'].values, gmm['auc'].values, gmm['std'].values,
	             markersize=8, markerfacecolor='black',
	             fmt='o-', capsize=3, color=colors[j],
	             ecolor='tab:red',
	             label=labels[j])

	try:
		mask = df['model_name'] == 'KDE'
		kde = df[mask]
		j = 2
		plt.errorbar(kde['train_size'].values, kde['auc'].values, kde['std'].values,
		             markersize = 8, markerfacecolor ='black',
		             fmt='^-', capsize=3, color=colors[j],
		             ecolor='tab:red',
		             label=labels[j])
	except Exception as e:
		print(e)

	plt.xticks(ocsvm['train_size'].values, fontsize=font_size - 1)
	plt.yticks(fontsize=font_size - 1)
	plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	# fig.suptitle(f'{data_name}', y=0.98, fontsize=font_size)
	plt.xlabel('Train size ($n$)', fontsize=font_size)
	plt.ylabel('AUC', fontsize=font_size)
	# plt.title(f'{data_name}', fontsize = font_size-2)
	if data_name == 'DWSHR_AECHO_2020':
		# bbox_to_anchor: (x, y, width, height)
		plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.3, 0.5, 0.5), fontsize=font_size - 2)
	elif data_name == 'AECHO1_2020':
		plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.068, 0.5, 0.5), fontsize=font_size - 2)
	else:
		plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.25, 0.5, 0.5), fontsize=font_size - 2)
	# plt.ylim([0.86, 1])
	# n_groups = len(show_datasets)
	# index = np.arange(n_groups)
	# # print(index)
	# # plt.xlim(xlim[0], n_groups)
	# plt.xticks(index + len(show_repres) // 2 * bar_width, labels=[v for v in show_datasets])
	plt.tight_layout()

	# plt.legend(show_repres, loc='lower right')
	# # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
	print(out_file)
	plt.savefig(out_file, dpi = 600)  # should use before plt.show()
	plt.show()
	plt.close()

	return out_file

def get_results_(in_file, DATASETS = [], FEATURES=['IAT+SIZE'],
                 MODELS = ['OCSVM(rbf)', 'GMM(full)', 'KDE'],
                 HEADERS=[False], TUNINGS=[True]):
	try:
		df = pd.read_csv(in_file, header=None) # doesn't work warn_bad_lines=True, error_bad_lines=False
	except Exception as e:
		data = []
		with open(in_file, 'r') as f:
			line = f.readline()
			while line:
				if (not line.startswith(',0_0|0_0|0_0,')):
					data.append(line.split(','))
				line = f.readline()
		df = pd.DataFrame(data)

	res = []
	for dataset, feature, header, model, tuning in list(
			itertools.product(DATASETS, FEATURES, HEADERS, MODELS, TUNINGS)):
		try:
			# baseline
			baseline = get_one_result(df, dataset, feature, f'header_{header}', 'OCSVM(rbf)', f'tuning_{tuning}')
			# another model result
			model_result = get_one_result(df, dataset, feature, f'header_{header}', model, f'tuning_{tuning}')

			# data = get_speedup(baseline, model_result)
			line = model_result[:]
		except Exception as e:
			lg.error(e)
			traceback.print_exc()
			data = (0, 0, 0, 0, 0)
		# line = ','.join([dataset, feature, f'header_{header}', model, f'tuning_{tuning}'] + [str(v) for v in data])
		lg.debug(line)
		res.append(line)

	return res

if __name__ == '__main__':
	is_iot_device = False
	if is_iot_device:
		root_dir = 'examples/offline/report/out/src_dst/results-20210928'
		root_dir = 'examples/offline/report/out/src_dst/results-20220905'
		root_dir = 'examples/offline/report/out/src_dst/results-KDE_GMM-20220916'
		date_str = '2022-09-16 12:32:22.738044'
		for device in ['RSPI', 'Nano']:
			lg.info(f'\n\n***{device}, {root_dir}')
			for n_normal_max_train in [1000, 2000, 3000, 4000, 5000]:  # 1000, 3000, 5000, 10000
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
		root_dir = 'examples/offline/deployment/out'
		# After deployment and copy the result ('examples/offline/deployment/out/src_dst/results') to 'examples/offline/report/out/src_dst/'
		for device in ['MacOS', 'RSPI', 'NANO']:
			if device == 'RSPI':
				date_str = '2022-09-17 08:48:25.295151'
			elif device =='NANO':
				date_str = '2022-09-16 12:34:44.607093'
			else:
				date_str = '2022-09-20 14:47:11.703436'
				data_str = '2022-09-26 20:51:34.988580'
				data_str = '2022-09-27 21:03:06.474680'
				data_str = '2022-09-28 08:16:08.248261'
			df_all = []
			ocsvm_gmm = []
			for n_normal_max_train in [1000, 2000, 3000, 4000, 5000]:  # 1000, 3000, 5000, 10000, # 1000, 2000, 3000, 4000, 5000
				lg.info(f'\n\n***{device}, {root_dir}')
				# in_file = os.path.join(root_dir, f'{device}/deployed_results/IAT+SIZE-header_False/gather-all.csv')
				if device == 'MacOS':
					# in_file = os.path.join(root_dir, f'{device}/deployed_results2/IAT+SIZE-header_False/gather-all.csv')
					in_file = os.path.join(root_dir,  f'{device}/train_size_{n_normal_max_train}/src_dst/results/{date_str}/IAT+SIZE-header_False/gather-all.csv')
					in_file = os.path.join(root_dir,
				                       f'train_size_{n_normal_max_train}/src_dst/results/{data_str}/IAT+SIZE-header_False/gather-all.csv')

				# continue
				else:
					in_file = os.path.join(root_dir, f'{device}/train_size_{n_normal_max_train}/src_dst/results/{date_str}/IAT+SIZE-header_False/gather-all.csv')
				lg.debug(in_file)

				#### for different training sizes
				# 'DWSHR_AECHO_2020', 'CTU1', 'MAWI1_2020' ,'MACCDC1',  'AECHO1_2020', 'AECHO1_2020, UNB3_345, SFRIG1_2021
				ocsvm_gmm_data_name = 'AECHO1_2020'
				ocsvm_gmm.append((n_normal_max_train, get_results_(in_file, DATASETS = [ocsvm_gmm_data_name],
				                                                   FEATURES=['IAT+SIZE'], HEADERS=[False],
				                                                   TUNINGS = [True])))
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
			check_path(gmm_file)
			df_all.to_csv(gmm_file, sep=',', encoding='utf-8', index=False)
			print(gmm_file)

			show_diff_train_sizes(ocsvm_gmm, out_file= gmm_file + f'-{ocsvm_gmm_data_name}.png')
			# main(root_dir, feature='IAT+SIZE', header=False)
			# main(root_dir, feature='STATS', header=True)
