#!/usr/bin/env python3

import pathlib
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

parser = argparse.ArgumentParser(
    prog=pathlib.Path(__file__).name, usage='%(prog)s [options]',
    description='''Compare free-energy estimates from noising
                   and denoising variational bounds to a ground truth''')

parser.add_argument('-noising_filenames', required=True, nargs='+', type=str,
    help='data files contaning noising free-energy estimates', metavar='filename')
parser.add_argument('-denoising_filenames', required=True, nargs='+', type=str,
    help='data files contaning denoising free-energy estimates', metavar='filename')
parser.add_argument('-reference_filename', required=True, type=str,
    help='data file contaning reference free-energy estimates', metavar='filename')
parser.add_argument('-noising_key', required=False, type=str, metavar='column_header',
    default='wt', help='data column header for noising free-energy estimates')
parser.add_argument('-denoising_key', required=False, type=str, metavar='column_header',
    default='ft', help='data column header for denoising free-energy estimates')
parser.add_argument('-reference_key', required=False, type=str, metavar='column_header',
    default='wt', help='data column header for reference free-energy estimates')
parser.add_argument('-xlim', required=False, nargs=2, type=float,
    default=[None, None], help='x-axis limits', metavar=' ')
parser.add_argument('-ylim', required=False, nargs=2, type=float,
    default=[None, None], help='y-axis limits', metavar=' ')
parser.add_argument('-savefig', type=str, default=None,
    help='saved location for plotted data (PDF)', metavar=' ')
parser.add_argument('-rescaling_factor', type=float, default=1,
    help='rescaling factor for ALL free-energy estimates', metavar=' ')


if __name__ == '__main__':

    args = parser.parse_args()
    assert len(args.noising_filenames) == len(args.denoising_filenames)

    noising_data = [
        pd.read_csv(str(file), dtype=np.float64, comment='#', sep=r'\s+')
        for file in args.noising_filenames]
    denoising_data = [
        pd.read_csv(str(file), dtype=np.float64, comment='#', sep=r'\s+')
        for file in args.denoising_filenames]

    reference_data = pd.read_csv(
        str(args.reference_filename), dtype=np.float64, comment='#', sep=r'\s+')
    reference_avg = reference_data[args.reference_key+'_avg'].to_numpy()
    reference_std = np.sqrt(reference_data[args.reference_key+'_std'].to_numpy())

    noising_estimates_x = []
    noising_estimates_y = []
    noising_estimates_yerr = []

    key_avg = args.noising_key + '_avg'
    key_std = args.noising_key + '_std'

    for data in noising_data:
        noising_estimates_x.append(
            1 / data['t'].to_numpy()[-1])
        noising_estimates_y.append(
            (data[key_avg].to_numpy()[-1] - reference_avg[-1]) -
            (data[key_avg].to_numpy()[+0] - reference_avg[+0]))
        noising_estimates_yerr.append(
            np.sqrt(data[key_std].to_numpy()[-2] ** 2 + reference_std[-2] ** 2 +
                    data[key_std].to_numpy()[+1] ** 2 + reference_std[+1] ** 2))

    denoising_estimates_x = []
    denoising_estimates_y = []
    denoising_estimates_yerr = []

    key_avg = args.denoising_key + '_avg'
    key_std = args.denoising_key + '_std'

    for data in denoising_data:
        denoising_estimates_x.append(
            1 / data['t'].to_numpy()[-1])
        denoising_estimates_y.append(
            (data[key_avg].to_numpy()[-1] - reference_avg[-1]) -
            (data[key_avg].to_numpy()[+0] - reference_avg[+0]))
        denoising_estimates_yerr.append(
            np.sqrt(data[key_std].to_numpy()[-2] ** 2 + reference_std[-2] ** 2 +
                    data[key_std].to_numpy()[+1] ** 2 + reference_std[+1] ** 2))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.0, 2.0), layout='none')
    plt.subplots_adjust(left=0.20, right=0.95, bottom=0.25, top=0.925)

    fmt_noising = 'r-o'
    label_noising = r'$f(\tau) \equiv \beta\, \langle \mathcal{W}_\tau \rangle$'

    ax.errorbar(
        np.asarray(noising_estimates_x),
        np.asarray(noising_estimates_y) / args.rescaling_factor,
        np.asarray(noising_estimates_yerr) / args.rescaling_factor, label=label_noising,
        linewidth=1.0, elinewidth=1.0, capsize=5.0, markersize=4.0,
        fmt=fmt_noising, zorder=-1)

    fmt_denoising = 'b-o'
    label_denoising = r'$f(\tau) \equiv \beta\, \widehat{\Delta F}_\tau$'

    ax.errorbar(
        np.asarray(denoising_estimates_x),
        np.asarray(denoising_estimates_y) / args.rescaling_factor,
        np.asarray(denoising_estimates_yerr) / args.rescaling_factor, label=label_denoising,
        linewidth=1.0, elinewidth=1.0, capsize=5.0, markersize=4.0,
        fmt=fmt_denoising, zorder=-1)

    ax.axhline(y=0.0, linewidth=0.5, color='k', zorder=0)
    ax.set_xticks(denoising_estimates_x)
    ax.set_ylim(args.ylim)
    ax.set_xlim(args.xlim)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

    ax.set_ylabel(r'$\beta\, [f(\tau) - \Delta F]$')
    ax.set_xlabel(r'$\tau^{-1}$')
    ax.legend(
        frameon=False, fancybox=False, draggable=True, loc='best',
        markerscale=1.0, labelspacing=0.5, borderpad=0.2,
        handlelength=0.5, handleheight=1.0, handletextpad=0.3)

    if args.savefig:
        plt.savefig(args.savefig, format='pdf')
    else:
        plt.show()
