import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np

data1 = pd.read_csv('./logs/stars_7_feat_sort/cls_5.mbs_20.ubs_10.numstep1.updatelr0.001nonorm/test_ubs10_stepsize0.001.csv', sep=',')
data2 = pd.read_csv('./logs/stars_7_feat_sort_20_20/cls_5.mbs_20.ubs_10.numstep1.updatelr0.001nonorm/test_ubs10_stepsize0.001.csv', sep=',')
data3 = pd.read_csv('./logs/stars_7_feat_sort_20_20_20/cls_5.mbs_20.ubs_10.numstep1.updatelr0.001nonorm/test_ubs10_stepsize0.001.csv', sep=',')
data4 = pd.read_csv('./logs/stars_7_feat_sort_40/cls_5.mbs_20.ubs_10.numstep1.updatelr0.001nonorm/test_ubs10_stepsize0.001.csv', sep=',')

row1 = data1.iloc[0]
row2 = data2.iloc[0]
row3 = data3.iloc[0]
row4 = data4.iloc[0]

plt.figure(figsize=(10, 5))

plt.plot(range(0, len(row1)), row1, 'bo-', label="MAML [40,40]")
plt.plot(range(0, len(row2)), row2, 'go-', label="MAML [20,20]")
plt.plot(range(0, len(row3)), row3, 'yo-', label="MAML [20,20,20]")
plt.plot(range(0, len(row4)), row4, 'ro-', label="MAML [40]")
plt.plot(range(0, len(row1)), np.zeros(11), 'ko-', label="Oracle")
plt.xlabel("Number of gradient steps")
plt.ylabel("MSE")
plt.legend(loc='upper right')
#plt.show()
plt.savefig('./MSE.png')