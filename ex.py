import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILES = ["gat_acc_Cora", "gatv2_acc_Cora", "gat_acc_CiteSeer", "gatv2_acc_CiteSeer"]
K = 16

for f in FILES:
    df = pd.read_csv(f, index_col=0)
    acc_mean = df.mean().to_numpy()

    print("Min Acc: {}".format(np.min(acc_mean)))
    print("Max Acc: {}".format(np.max(acc_mean)))
    print("Best K: {}".format(np.argmax(acc_mean) + 1))
    print("Range Acc: {}".format(np.max(acc_mean) - np.min(acc_mean)))
    acc_std = df.std().to_numpy()
    df = df.append(df.mean(), ignore_index=True)
    df = df.append(df.std(), ignore_index=True)

    _, ax = plt.subplots()
    ax.errorbar(np.array(list(range(1, K + 1))), acc_mean, yerr=acc_std, fmt="-o", linewidth=0.5)
    ax.set_xlabel(r'$\Phi$')
    ax.set_ylabel("Accuracy")
    plt.show()