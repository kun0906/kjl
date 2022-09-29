"""

"""
# Email: kun.bj@outlook.com
# Author: kun
# License: xxx
import itertools
import os.path
import traceback

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from examples.offline._constants import *
from examples.offline.report import _speedup
from examples.offline.report._speedup import get_one_result
from kjl.utils.tool import check_path

import matplotlib.pyplot as plt

def show_diff_train_sizes(ax, idx, ocsvm_gmm, font_size = 18):

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
	ax.errorbar(ocsvm['train_size'].values, ocsvm['auc'].values, ocsvm['std'].values, fmt='*-',
	             capsize=3, color=colors[j], ecolor='tab:red',
	             markersize=8, markerfacecolor='black',
	             label=labels[j])

	mask = df['model_name'] == 'GMM(full)'
	gmm = df[mask]
	j = 1
	ax.errorbar(gmm['train_size'].values, gmm['auc'].values, gmm['std'].values,
	             markersize=8, markerfacecolor='black',
	             fmt='o-', capsize=3, color=colors[j],
	             ecolor='tab:red',
	             label=labels[j])


	mask = df['model_name'] == 'KDE'
	kde = df[mask]
	j = 2
	ax.errorbar(kde['train_size'].values, kde['auc'].values, kde['std'].values,
	             markersize = 8, markerfacecolor ='black',
	             fmt='^-', capsize=3, color=colors[j],
	             ecolor='tab:red',
	             label=labels[j])


	ax.set_xticks(ocsvm['train_size'].values, fontsize=font_size - 1)
	# ax.set_yticks()
	# ax.set_yticks(ax.ticks(), fontsize=font_size - 1)
	ax.tick_params(axis='both', which='major', labelsize=font_size)
	if idx == 2:
		# ax.set_yticks(fontsize=font_size - 1)
		# ax.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		# fig.suptitle(f'{data_name}', y=0.98, fontsize=font_size)
		# ax.set_xlabel('Train size ($n$)', fontsize=font_size)
		# ax.set_ylabel('AUC', fontsize=font_size)
		ax.legend(loc='upper right', bbox_to_anchor=(0.5, 0.01, 0.5, 0.5), fontsize=font_size - 2)
		# pass
	# # plt.title(f'{data_name}', fontsize = font_size-2)
	# if data_name == 'DWSHR_AECHO_2020':
	# 	# bbox_to_anchor: (x, y, width, height)
	# 	plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.3, 0.5, 0.5), fontsize=font_size - 2)
	# elif data_name == 'AECHO1_2020':
	# 	plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.068, 0.5, 0.5), fontsize=font_size - 2)
	# else:
	# 	plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.25, 0.5, 0.5), fontsize=font_size - 2)
	# # plt.ylim([0.86, 1])
	# # n_groups = len(show_datasets)
	# # index = np.arange(n_groups)
	# # # print(index)
	# # # plt.xlim(xlim[0], n_groups)
	# # plt.xticks(index + len(show_repres) // 2 * bar_width, labels=[v for v in show_datasets])
	# plt.tight_layout()
	#
	# # plt.legend(show_repres, loc='lower right')
	# # # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
	# print(out_file)
	# plt.savefig(out_file, dpi = 600)  # should use before plt.show()
	# plt.show()
	# plt.close()

	# return out_file

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
	root_dir = 'examples/offline/report/out/src_dst/results-20220905'
	root_dir = 'examples/offline/report/out/src_dst/results-KDE_GMM-20220916'
	root_dir = 'examples/offline/deployment/out'



	fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
	                                    figsize=(18, 5))

	#### for different training sizes
	# 'DWSHR_AECHO_2020', 'CTU1', 'MAWI1_2020' ,'MACCDC1',  'AECHO1_2020', 'AECHO1_2020, UNB3_345, SFRIG1_2021
	# After deployment and copy the result ('examples/offline/deployment/out/src_dst/results') to 'examples/offline/report/out/src_dst/'
	titles = ['a) AECHO', 'b) SFRIG', 'c) DWSHR']
	for idx, ocsvm_gmm_data_name in enumerate(['AECHO1_2020', 'SFRIG1_2021', 'DWSHR_AECHO_2020']):
		date_str = '2022-09-20 14:47:11.703436'
		data_str = '2022-09-26 20:51:34.988580'
		data_str = '2022-09-27 21:03:06.474680'
		data_str = '2022-09-28 08:16:08.248261'
		df_all = []
		ocsvm_gmm = []
		for n_normal_max_train in [1000, 2000, 3000, 4000, 5000]:  # 1000, 3000, 5000, 10000, # 1000, 2000, 3000, 4000, 5000
			in_file = os.path.join(root_dir,
			                       f'train_size_{n_normal_max_train}/src_dst/results/{data_str}/IAT+SIZE-header_False/gather-all.csv')
			ocsvm_gmm.append((n_normal_max_train, get_results_(in_file, DATASETS = [ocsvm_gmm_data_name],
			                                                   FEATURES=['IAT+SIZE'], HEADERS=[False],
			                                                   TUNINGS = [True])))

		# axes[idx].set_title(titles[idx], fontsize=20)
		axes[idx].set_title(titles[idx], y=0, pad=-60, verticalalignment="top", fontsize=20)
		show_diff_train_sizes(axes[idx], idx, ocsvm_gmm)

	# fig.suptitle('Errorbar subsampling')
	font_size =20
	# fig.supxlabel('Train size ($n$)', fontsize=font_size, y=0.003)
	# fig.supylabel('AUC', fontsize=font_size)
	#(x, y)
	fig.text(0.5, 0.113, 'Train size ($n$)', ha='center', fontsize=font_size)
	fig.text(0.015, 0.6, 'AUC', va='center', rotation='vertical', fontsize=font_size)
	plt.yticks(fontsize=font_size - 1)
	plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

	device = 'MacOS'
	out_file = os.path.join(root_dir, f'{device}/different_training_size_AECHO_SFRIG_DWSHR.png')
	check_path(out_file)

	plt.subplots_adjust(left=0.07, bottom=0.25, right=0.98, top=0.95, wspace=0.05, hspace=0)
	# plt.tight_layout()
	# plt.legend(show_repres, loc='lower right')
	# # plt.savefig("DT_CNN_F1"+".jpg", dpi = 400)
	print(out_file)
	plt.savefig(out_file, dpi = 600)  # should use before plt.show()
	plt.show()
	plt.close()
