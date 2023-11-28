import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

colors = [(0.0, 0., 1),(1., 0., 0.0)]
cmap = LinearSegmentedColormap.from_list('my_cmap', colors)

stds = ["0", "001", "005", "01", "02", "05", "1"]

losses = []
colors = []
if __name__ == '__main__':
    for i, std in enumerate(stds):
        data = pd.read_csv(f'/root/zmw/icl/models/noisy_linear_regression/NL_{std}/curve.csv')["0"]
        data = data.to_numpy()

        plt.plot(range(0,len(data),10), data[::10], color=cmap(i/len(stds)), linewidth=0.5)
    plt.legend([0,0.01,0.05,0.1,0.2,0.5,1])
    plt.title(r"Training loss with different label noise $\sigma$")
    plt.savefig('loss_curve.png', dpi=200)