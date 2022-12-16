import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_plots(K, files):
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv("../"+f, index_col=0)

            acc_mean = df.mean().to_numpy()
            acc_std = df.std().to_numpy()
            _, ax = plt.subplots()
            ax.errorbar(np.array(list(range(1, K + 1))), acc_mean, yerr=acc_std, fmt="-o", linewidth=0.5)
            ax.set_xlabel(r'$K$')
            ax.set_ylabel("Accuracy")
            plt.show()
