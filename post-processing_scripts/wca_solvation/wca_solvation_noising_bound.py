#!/usr/bin/env python3
# coding: utf-8

import re
import sys
import numpy as np
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser
from collections import OrderedDict

from scipy.integrate import cumulative_trapezoid

parser = ArgumentParser(
    prog=Path(__file__).name, usage='%(prog)s [options]',
    description='''Process noising trajectory data in hoomd.write.Table format
                   to estimate the upper bound on the free-energy difference''')

parser.add_argument('-data_dirname', type=str, required=True,
    help='path to parent directory of mixing energy files', metavar=' ')
parser.add_argument('-data_baseglob', type=str, default='noising_*.txt',
    help='glob string for file name patter of mixing energy files (CSV format)', metavar=' ')
parser.add_argument('-solvent_energy_key', required=False,
                    type=str, default='wca_monomer.U_BB.energy', metavar=' ')
parser.add_argument('-solution_energy_key', required=False,
                    type=str, default='wca_monomer.U_AB.energy', metavar=' ')
# parser.add_argument('-solvation_energy_key', required=False,
#                     type=str, default='wca_monomer.dUdλ_AB.energy', metavar=' ')
parser.add_argument('-solvation_energy_key', required=False,
                    type=str, default='wca_monomer.mixing.energy', metavar=' ')


def get_parameter_from_dirname(dirname, parameter_key):
    assert isinstance(dirname, Path)

    pattern = re.compile(r'(?<=\b{}_)[0-9]+[.0-9]*'.format(parameter_key))
    matches = pattern.findall(str(dirname))

    if not matches:
        raise ValueError(
            f'Parameter name "{parameter_key}" not in dirname "{dirname}".')
    try:
        onlymatch, = matches
    except:
        raise ValueError(
            f'Multiple matches to parameter name "{parameter_key}" in dirname "{dirname}".')
    else:
        return float(onlymatch)


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

    _solvation_energies = []

    for filename in sorted(filenames):
        print(f'# working through file '
              f'"{filename.relative_to(PROJECT_DIR)}" ...',
              end=' ', file=sys.stderr)

        # open data file as Pandas CSV-like object
        file = pd.read_csv(filename, dtype=np.float64, sep=r'\s+')

        # read solvation_energy column
        solvation_energy = file[args.solvation_energy_key].to_numpy()
        _solvation_energies.append(solvation_energy)

        print(f'\r', end='', flush=True, file=sys.stderr)

    solvation_energies = np.vstack(_solvation_energies)
    N_trajs, N_epochs = solvation_energies.shape

    t = np.linspace(0, τ, num=N_epochs, endpoint=True)
    ctrapz = lambda y: cumulative_trapezoid(y, x=t, initial=0)

    # --- WORK RATE --- #
    dwdt = mixing_energies * β / τ
    dwdt_avg = dwdt.mean(0)
    dwdt_std = np.sqrt(dwdt.var(0) / (N_trajs - 1))

    # --- THERMODYNAMIC WORK --- #
    wt = ctrapz(dwdt)
    wt_avg = wt.mean(0)
    wt_std = np.sqrt(wt.var(0) / (N_trajs - 1))
    # --- #

    # print output as CSV file to stdout
    dataframe = pd.DataFrame(OrderedDict(
        t=t,
        wt_avg=wt_avg, wt_std=wt_std,
        dwdt_avg=dwdt_avg, dwdt_std=dwdt_std)
    print(f'#{args.data_dirname}')
    dataframe.to_string(
        sys.stdout, index=False, col_space=14, justify='right',
        float_format=lambda x: u'{0:9.6e}'.format(x))
