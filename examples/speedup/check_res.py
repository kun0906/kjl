from kjl.utils.data import load_data
import numpy as np
import copy

def check(in_file = '.dat'):

    res = load_data(in_file)
    print(res)
    best_res, mid_res = res
    best1, outs1 = mid_res

    best_avg_auc = -1
    for out in outs1[0]:
        if np.mean(out['auc']) > best_avg_auc:
            best_avg_auc = np.mean(out['auc'])
            best_results = copy.deepcopy(out)

    print('\n', best_avg_auc, best_results)

    return out


def main():

    in_file = 'speedup/out-qs/src_dst/iat_size-header_False/MAWI1_2020/before_proj_False-gs_True/KJL-QS-GMM(full)-std_False_center_False-d_5-full/res.dat'
    check(in_file)

if __name__ == '__main__':
    main()