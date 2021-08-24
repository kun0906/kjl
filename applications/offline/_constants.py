# print('test2')
# import sys
# from os.path import dirname
# sys.path.append(dirname(__file__))
# sys.path.append('applications')
# print(sys.path)
#


RANDOM_STATE = 42
OVERWRITE = False
VERBOSE= 10

FEATURE = 'IAT+SIZE'
ORIGINAL_DIR =  r'../Datasets'
IN_DIR = r'applications/offline/data'
OUT_DIR = r'applications/offline/out'

DATASETS = {#'UNB345_3': 'UNB',
            'CTU1': 'CTU',
            # 'MAWI1_2020': 'MAWI',
            # 'MACCDC1': 'MACCDC',
            # 'SFRIG1_2020': 'SFRIG',
            # 'AECHO1_2020': 'AECHO',
            # 'DWSHR_WSHR_2020': 'DWSHR'
            }

MODELS = {#'OCSVM(rbf)': 'OCSVM',
          'KJL-OCSVM(linear)': 'OC-KJL-SVM',
          # 'Nystrom-OCSVM(linear)': 'OC-Nystrom-SVM',
          #
          # 'KJL-GMM(full)': 'OC-KJL',
          # 'KJL-GMM(diag)': 'OC-KJL(diag)',
          # # 'Nystrom-GMM(full)': 'OC-Nystrom',
          # # 'Nystrom-GMM(diag)': 'OC-Nystrom(diag)',
          #
          # 'KJL-QS-GMM(full)': 'OC-KJL-QS_0',
          # 'KJL-QS-GMM(diag)': 'OC-KJL-QS(diag)_0',
          # # 'Nystrom-QS-GMM(full)': 'OC-Nystrom-QS_0',
          # # 'Nystrom-QS-GMM(diag)': 'OC-Nystrom-QS(diag)_0',
          #
          # 'KJL-QS-init_GMM(full)': 'OC-KJL-QS',
          # 'KJL-QS-init_GMM(diag)': 'OC-KJL-QS(diag)',
          # 'Nystrom-QS-init_GMM(full)': 'OC-Nystrom-QS',
          # 'Nystrom-QS-init_GMM(diag)': 'OC-Nystrom-QS(diag)',
          }

data_orig2name = {'UNB345_3': 'UNB',
                  'CTU1': 'CTU',
                  'MAWI1_2020': 'MAWI',
                  'MACCDC1': 'MACCDC',
                  'SFRIG1_2020': 'SFRIG',
                  'AECHO1_2020': 'AECHO',
                  'DWSHR_WSHR_2020': 'DWSHR'
                  }

data_name2orig = {'UNB': 'UNB345_3',
                  'CTU': 'CTU1',
                  'MAWI': 'MAWI1_2020',
                  'MACCDC': 'MACCDC1',
                  'SFRIG': 'SFRIG1_2020',
                  'AECHO': 'AECHO1_2020',
                  'DWSHR': 'DWSHR_WSHR_2020',
                  }

model_orig2name = {'OCSVM(rbf)': 'OCSVM',
                   'KJL-OCSVM(linear)': 'OC-KJL-SVM',
                   'Nystrom-OCSVM(linear)': 'OC-Nystrom-SVM',

                   'KJL-GMM(full)': 'OC-KJL',
                   'KJL-GMM(diag)': 'OC-KJL(diag)',
                   'Nystrom-GMM(full)': 'OC-Nystrom',
                   'Nystrom-GMM(diag)': 'OC-Nystrom(diag)',

                   'KJL-QS-GMM(full)': 'OC-KJL-QS_0',
                   'KJL-QS-GMM(diag)': 'OC-KJL-QS(diag)_0',
                   'Nystrom-QS-GMM(full)': 'OC-Nystrom-QS_0',
                   'Nystrom-QS-GMM(diag)': 'OC-Nystrom-QS(diag)_0',

                   'KJL-QS-init_GMM(full)': 'OC-KJL-QS',
                   'KJL-QS-init_GMM(diag)': 'OC-KJL-QS(diag)',
                   'Nystrom-QS-init_GMM(full)': 'OC-Nystrom-QS',
                   'Nystrom-QS-init_GMM(diag)': 'OC-Nystrom-QS(diag)',
                   }

model_name2orig = {'OCSVM': 'OCSVM(rbf)',
                   'OC-KJL-SVM': 'KJL-OCSVM(linear)',
                   'OC-Nystrom-SVM': 'Nystrom-OCSVM(linear)',

                   'OC-KJL': 'KJL-GMM(full)',
                   'OC-KJL(diag)': 'KJL-GMM(diag)',
                   'OC-Nystrom': 'Nystrom-GMM(full)',
                   'OC-Nystrom(diag)': 'Nystrom-GMM(diag)',

                   'OC-KJL-QS_0': 'KJL-QS-GMM(full)',
                   'OC-KJL-QS(diag)_0': 'KJL-QS-GMM(diag)',
                   'OC-Nystrom-QS_0': 'Nystrom-QS-GMM(full)',
                   'OC-Nystrom-QS(diag)_0': 'Nystrom-QS-GMM(diag)',

                   'OC-KJL-QS': 'KJL-QS-init_GMM(full)',
                   'OC-KJL-QS(diag)': 'KJL-QS-init_GMM(diag)',
                   'OC-Nystrom-QS': 'Nystrom-QS-init_GMM(full)',
                   'OC-Nystrom-QS(diag)': 'Nystrom-QS-init_GMM(diag)',
                   }
