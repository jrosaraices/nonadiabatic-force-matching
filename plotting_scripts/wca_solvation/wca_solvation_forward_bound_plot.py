#!/usr/bin/env python3

import re
import pathlib
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib import colors, colormaps

# set matplotlib global font settings
# (change depending on font and LaTeX availability)
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=(
    r'\usepackage{amsmath}'
    r'\usepackage{amssymb}'
    r'\usepackage[helvratio=0.9,trueslanted]{newtx}'
))

# set default fontsize for legends, axis labels, and axis ticklabels
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.labelsize'] = 12

# set default linestyle and linewidth
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['axes.linewidth'] = 1.0

# set colormap to plot lines with
cmap = colormaps["magma_r"]
norm = colors.Normalize(vmin=0, vmax=1)


def get_noneq_label(filepath):
    label = r'(?<=τ_)[0-9]+\.[0-9]+(?=\.log$)'
    try:
        match = re.search(label, str(filepath)).group()
    except AttributeError:
        return ''
    else:
        return match


parser = argparse.ArgumentParser(
    prog=pathlib.Path(__file__).name, usage='%(prog)s [options]',
    description='''Plot forward free-energy estimates''')

parser.add_argument('-data_dir', type=str, required=True, nargs='+',
    help='paths to files containing post-processed forward bound data')
parser.add_argument('-reversed', action='store_true',
    help='assume that estimates were generated for the B → A transition')
parser.add_argument('-ylim', type=float, default=[None, None],
    help='y-axis limits')


if __name__ == '__main__':

    args = parser.parse_args()

    data_files = sorted(pathlib.Path(file) for file in args.data_files)
    data_labels = []
    data_arrays = []

    for file in data_files:
        label = get_noneq_label(file)
        if label:
            data_labels.append(label)
            data_arrays.append(pd.read_csv(str(file), dtype=np.float64, comment='#', sep=r'\s+'))

    num_files = len(data_labels) - 1
    data_colors = cmap((np.arange(num_files) + 1 / 2) / (num_files + 1)).tolist()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.0, 2.0), layout='none')
    plt.subplots_adjust(left=0.20, right=0.95, bottom=0.25, top=0.925)

    for (label, color, array) in zip(data_labels, data_colors, data_arrays):
        x, y, yerr = array['t'].to_numpy(), array['wt_avg'].to_numpy(), array['wt_std'].to_numpy()
        line, = ax.plot(x / x.max(), y, color=color,
                        linewidth=2.0, label=r'$\tau = {:.1f}$'.format(float(label)))
        fill = ax.fill_between(x / x.max(), y - yerr, y + yerr,
                               facecolor=line.get_color(), alpha=1.0)

    label = r'$\tau \to \infty$'
    color = cmap((num_files + 1 / 2) / (num_files + 1))
    array = data_arrays[-1]

    x, y, yerr = array['t'].to_numpy(), array['wt_avg'].to_numpy(), array['wt_std'].to_numpy()
    line, = ax.plot(x / x.max(), y, color=color,
                    linewidth=2.0, label=label)
    fill = ax.fill_between(x / x.max(), y - yerr, y + yerr,
                           facecolor=line.get_color(), alpha=1.0, zorder=-1)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim(args.ylim)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5.0))

    ax.set_xlabel(r'$t / \tau$')
    ax.set_ylabel(r'$\beta \, \langle \mathcal{W}_t \rangle$')
    ax.legend(loc='best', frameon=False, draggable=True, fancybox='off')

    plt.show()
