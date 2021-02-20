
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#
# penguins = sns.load_dataset("penguins")
# sns.histplot(data=penguins, x="flipper_length_mm")
# plt.show()

import numpy as np

data = [2803, 594, 560, 286, 244, 59, 37, 13, 9, 8, 7, 7, 5, 5, 4, 3, 3, 3, 2, 2]
# # np.histogram(data, bins=10)
# plt.hist(data, bins=[0, 10, 50, 100, 500, 2000, 2500], width=2)

"""
https://stackoverflow.com/questions/33497559/display-a-histogram-with-very-non-uniform-bin-widths
display a histogram with very non-uniform bin widths
"""
# fig,ax = plt.subplots()
bins=[0, 5, 10, 50, 100, 500, 2000, max(X) if max(X) > 3000 else 3000]

hist, bin_edges = np.histogram(data,bins)
data = [[v1, v2] for v1, v2 in zip(hist, bin_edges)]
df = pd.DataFrame(data, columns=['height', 'x-interval'])
ax = sns.barplot(y='height', x ='x-interval', hue=None, data=df)
# ax.bar(range(len(hist)),hist,width=1,align='center',tick_label=
#         ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])

pre_v = bin_edges[0]
labels = []
for v in bin_edges[1:]:
    labels.append(f'{pre_v}-{v}')
    pre_v = v
ax.set_xticklabels(labels)
plt.show()
