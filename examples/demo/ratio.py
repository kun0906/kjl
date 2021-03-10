



import numpy as np



def get_auc_ratio(ocsvm_aucs, gmm_aucs, same=True):
    if same:
        mean_ratio = np.mean(ocsvm_aucs)
        std_ratio = np.std(ocsvm_aucs)
    else:
        mean_ocsvm = np.mean(ocsvm_aucs)
        mean_gmm = np.mean(gmm_aucs)
        print(f'para_1. {mean_ocsvm}, para_2: {mean_gmm}')
        mean_ratio = mean_gmm / mean_ocsvm

        ratios = [v / mean_ocsvm for v in gmm_aucs]
        print(ratios)
        std_ratio = np.std(ratios)

    diff = f"{mean_ratio:.2f} +/- {std_ratio:.2f}"

    return diff

# Nystrom-QS-GMM/OCSVM
ocsvm_aucs = [0.9921444444444444,0.9938555555555555,0.9887111111111111,0.9864444444444445,0.9934666666666666]
gmm_aucs = [0.9556,0.9544999999999999,0.9670555555555556,0.9631333333333334,0.9104666666666666]
diff = get_auc_ratio(ocsvm_aucs, gmm_aucs, same=False)
print(diff)


# OCSVM/Nystrom-QS-GMM
ocsvm_test_times = [0.658808,0.396533,0.737079,0.553822,10.01747]
gmm_test_times = [0.035903000000000004,0.020409,0.025368,0.01683,0.038575]
diff = get_auc_ratio(gmm_test_times, ocsvm_test_times, same=False)
print(diff)

