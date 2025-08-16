#!/usr/bin/env python3
# coding: utf-8

import sys
import numpy as np
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser
from collections import OrderedDict

from scipy.integrate import cumulative_trapezoid
from lj_formation_noising_bound import get_parameter_from_dirname

parser = ArgumentParser(
    prog=Path(__file__).name, usage='%(prog)s [options]',
    description='''Process nonequilibrium trajectory data in hoomd.write.Table format
                   to calculate the thermodynamic work''')

parser.add_argument('-data_dirname', type=str, required=True,
    help='path to parent directory of mixing energy files', metavar=' ')
parser.add_argument('-data_baseglob', type=str, default='nonequilibrium_*.txt',
    help='glob string for file name patter of mixing energy files (CSV format)', metavar=' ')
parser.add_argument('-data_key', type=str, default='mixing.energy',
    help='key of the mixing energy column in the CSV data files', metavar=' ')


if __name__ == '__main__':

    args = parser.parse_args()
    PROJECT_DIR = Path().resolve()
    dirname = PROJECT_DIR.joinpath(args.data_dirname)
    assert dirname.is_dir()

    β = get_parameter_from_dirname(dirname, 'β')
    τ = get_parameter_from_dirname(dirname, 'τ')

    filenames = [
        filename for filename in
        dirname.glob(args.data_baseglob)
        if not filename.is_symlink()]

    _mixing_energies = []

    for filename in sorted(filenames):
        print(f'# working through file '
              f'"{filename.relative_to(PROJECT_DIR)}" ...',
              end=' ', file=sys.stderr)

        # open data file as Pandas CSV-like object
        file = pd.read_csv(filename, dtype=np.float64, sep=r'\s+')

        # read mixing_energy column
        mixing_energy = file[args.data_key].to_numpy()
        _mixing_energies.append(mixing_energy)

        print(f'\r', end='', flush=True, file=sys.stderr)

    mixing_energies = np.vstack(_mixing_energies)
    N_trajs, N_epochs = mixing_energies.shape

    t = np.linspace(0, τ, num=N_epochs, endpoint=True)
    ctrapz = lambda y: cumulative_trapezoid(y, x=t, initial=0)

    # --- WORK RATE --- #
    dwdt = -mixing_energies[:, ::-1] * β / τ
    dwdt_avg = dwdt.mean(0)
    dwdt_std = np.sqrt(dwdt.var(0) / (N_trajs - 1))

    # --- THERMODYNAMIC WORK --- #
    wt = ctrapz(dwdt)
    wt_avg = wt.mean(0)
    wt_std = np.sqrt(wt.var(0) / (N_trajs - 1))

    # print output as CSV file to stdout
    dataframe = pd.DataFrame(OrderedDict(
        t=t, dwdt_avg=dwdt_avg, dwdt_std=dwdt_std,
        wt_avg=wt_avg, wt_std=wt_std))
    print(f'#{args.data_dirname}')
    dataframe.to_string(
        sys.stdout, index=False, col_space=14, justify='right',
        float_format=lambda x: u'{0:9.6e}'.format(x))
