#!/usr/bin/env python3
# coding: utf-8

import re
import sys
import numpy as np
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser
from collections import OrderedDict

from scipy.special import softmax
from scipy.integrate import cumulative_trapezoid
from wca_forward_bound import get_parameter_from_dirname


parser = ArgumentParser(
    prog=Path(__file__).name, usage='%(prog)s [options]',
    description='''Process backward trajectory data in hoomd.write.Table format
                   to estimate the lower bound on the free-energy difference''')

parser.add_argument('-data_dirname', type=str, required=True,
                    help='path to parent directory of mixing energy files', metavar=' ')
parser.add_argument('-data_baseglob', type=str, default='backward_*.txt',
                    help='glob string for file name patter of mixing energy files (CSV format)', metavar=' ')
parser.add_argument('-solvent_energy_key', required=False,
                    type=str, default='wca_monomer.U_BB.energy', metavar=' ')
parser.add_argument('-solution_energy_key', required=False,
                    type=str, default='wca_monomer.U_AB.energy', metavar=' ')
# parser.add_argument('-solvation_energy_key', required=False,
#                     type=str, default='wca_monomer.dUdλ_AB.energy', metavar=' ')
parser.add_argument('-solvation_energy_key', required=False,
                    type=str, default='wca_monomer.mixing.energy', metavar=' ')
parser.add_argument('-policy_energy_key', required=False,
                    type=str, default='wca_monomer.policy.energy', metavar=' ')
parser.add_argument('-policy_squared_force_key', required=False,
                    type=str, default='wca_monomer.policy.squared_force', metavar=' ')


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
    _policy_energies = []
    _policy_squared_forces = []

    for filename in sorted(filenames):
        print(f'# working through file '
              f'"{filename.relative_to(PROJECT_DIR)}" ...',
              end=' ', file=sys.stderr)

        # open data file as Pandas CSV-like object
        file = pd.read_csv(filename, dtype=np.float64, sep=r'\s+')

        # read solvation_energy column
        solvation_energy = file[args.solvation_energy_key].to_numpy()
        _solvation_energies.append(solvation_energy)

        # read policy_energy colum
        policy_energy = file[args.policy_energy_key].to_numpy()
        _policy_energies.append(policy_energy)

        # read policy_squared_force column
        policy_squared_force = file[args.policy_squared_force_key].to_numpy()
        _policy_squared_forces.append(policy_squared_force)

        print(f'\r', end='', flush=True, file=sys.stderr)

    solvation_energies = np.vstack(_solvation_energies)
    N_trajs, N_epochs = solvation_energies.shape

    policy_energies = np.vstack(_policy_energies)
    policy_squared_forces = np.vstack(_policy_squared_forces)
    assert policy_energies.shape == policy_squared_forces.shape == solvation_energies.shape

    t = np.linspace(0, τ, num=N_epochs, endpoint=True)
    ctrapz = lambda y: cumulative_trapezoid(y, x=t, initial=0)

    # --- WORK RATE --- #
    dwdt = -solvation_energies[:, ::-1] * β / τ
    dwdt_avg = dwdt.mean(0)
    dwdt_std = np.sqrt(dwdt.var(0) / (N_trajs - 1))

    # --- THERMODYNAMIC WORK --- #
    wt = ctrapz(dwdt)
    wt_avg = wt.mean(0)
    wt_std = np.sqrt(wt.var(0) / (N_trajs - 1))

    # --- INTEGRATED SQUARED NORM OF NONADIABATIC FORCE --- #
    dcdt = -policy_squared_forces[:, ::-1] * β / 4
    ct_avg = ctrapz(dcdt).mean(0)
    ct_std = np.sqrt((ctrapz(dcdt) - ct_avg[None]).__pow__(2).mean(0) / (N_trajs - 1))

    # --- TERMINAL NONADIABATIC POTENTIAL WITH CUMULANT SHIFT --- #
    vt = -np.log(softmax(β * policy_energies[:, ::-1], 0))
    vt -= vt[:, 0][..., None]
    vt_avg = vt.mean(0)
    vt_std = np.sqrt(vt.var(0) / (N_trajs - 1))
    # --- #

    # --- DENOISING FREE-ENERGY ESTIMATE --- #
    dsdt = dwdt - dcdt
    st_avg = ctrapz(dsdt).mean(0)
    st_std = np.sqrt((ctrapz(dsdt) - st_avg[None]).__pow__(2).mean(0) / (N_trajs - 1))
    ft_avg = st_avg - vt_avg
    ft_std = np.sqrt((ctrapz(dsdt) - vt - ft_avg[None]).__pow__(2).mean(0) / (N_trajs - 1))
    # --- #

    # print output as CSV file to stdout
    dataframe = pd.DataFrame(OrderedDict(
        t=t,
        wt_avg=wt_avg, wt_std=wt_std,
        ct_avg=ct_avg, ct_std=ct_std,
        st_avg=st_avg, st_std=st_std,
        ft_avg=ft_avg, ft_std=ft_std,
        vt_avg=vt_avg, vt_std=vt_std))
    dataframe.to_string(
        sys.stdout, index=False, col_space=14, justify='right',
        float_format=lambda x: u'{0:9.6e}'.format(x))
