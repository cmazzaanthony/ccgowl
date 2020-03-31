import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_multiple_theta_matrices_2d(thetas, titles):
    # Generate a mask for the upper triangle
    mask = np.zeros_like(thetas[0], dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    n = len(thetas)
    assert n <= 4

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    f.tight_layout()
    axarr = [ax1, ax2, ax3, ax4]
    for i in range(n):
        ax = axarr[i]
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.set_yticks([])
        ax.get_yaxis().set_ticklabels([])

        # Set up the matplotlib figure
        ax.set_title(titles[i])

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(thetas[i], mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax,
                    xticklabels=[], yticklabels=[])

    plt.show()
