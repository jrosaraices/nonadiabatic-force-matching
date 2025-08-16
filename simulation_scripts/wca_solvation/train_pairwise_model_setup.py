#!/usr/bin/env python3
# coding: utf-8

import time
import random
import pathlib
import argparse
import datetime

import numpy as np
import gsd.hoomd

from itertools import pairwise, islice
from typing import List, Tuple, Literal
from pairwise_model import PairData, periodic, get_pairs_with_type_mask


def pair_data_from_frame(frame: gsd.hoomd.Frame,
        i_type: str, j_type: str, num_pairs: int) -> PairData:

    return get_pairs_with_type_mask(
        frame.particles.position, frame.particles.typeid,
        (frame.particles.types.index(i_type), frame.particles.types.index(j_type)),
        frame.configuration.box[:3], pair_list=None, num_pairs=num_pairs)


parser = argparse.ArgumentParser(
    prog=pathlib.Path(__file__).name, usage='%(prog)s [options]',
    description='''Train a neural network model of the Fokker--Planck
                   velocity field sampled from trajectory data''')

parser.add_argument('-gsd_dirname', type=str, required=True,
                    help='directory containing training data (".gsd")', metavar=' ')
parser.add_argument('-gsd_basename_glob', type=str, default='noising_*.gsd',
                    help='training data basename glob', metavar=' ')
parser.add_argument('-output_basename', type=str, default='data.npz',
                    help='output data basename (".npz")', metavar= ' ')
parser.add_argument('-trajectory_duration', type=float, default=1.0,
                    help='trajectory duration', metavar=' ')
parser.add_argument('-frame_step', type=int, default=1,
                    help='skip every "N" frames', metavar=' ')
parser.add_argument('-i_type', type=str, default='A',
                    help='particle type `I` in the pair type `(I, J)`')
parser.add_argument('-j_type', type=str, default='B',
                    help='particle type `J` in the pair type `(I, J)`')
parser.add_argument('-num_neighs', type=int, default=12,
                    help='number of nearest neighboring pairs to store', metavar=' ')

# ------------------------------------
# PREPARE TRAJECTORY DATA TO FIT MODEL
# ------------------------------------

if __name__ == '__main__':

    args = parser.parse_args()

    # --- LOAD TRAJECTORY DATA --- #

    print(f'# loading trajectory data...')
    tic = time.monotonic()

    ObsData = np.ndarray[Tuple[int, Literal[1]], np.dtype[np.float64]]
    VecData = np.ndarray[Tuple[int, Literal[3]], np.dtype[np.float64]]
    IdxData = np.ndarray[Tuple[int, Literal[2]], np.dtype[np.float32]]

    timestep: List[ObsData] = []
    position: List[VecData] = []
    velocity: List[VecData] = []

    pair_vectors: List[VecData] = []
    pair_indices: List[IdxData] = []

    dirname = pathlib.Path(args.gsd_dirname)
    filenames = sorted(list(dirname.glob(args.gsd_basename_glob)))

    print(filenames)

    for filename in filenames:
        print(f'# loading from GSD file "{str(filename)}" ...', end=' ')
        with gsd.hoomd.open(filename, mode='r') as trajectory:

            # trajectory timeslices
            num_frames = (len(trajectory) - 1) // args.frame_step
            time_grid = (np.arange(num_frames) + 1 / 2) / num_frames
            time_step = args.trajectory_duration / num_frames

            for last_frame, this_frame in pairwise(islice(trajectory, 0, None, args.frame_step)):

                # velocity
                vel = periodic(
                    this_frame.particles.position - last_frame.particles.position,
                    last_frame.configuration.box[None, :3]) / time_step
                velocity.append(vel)

                # midpoint position
                pos = periodic(
                    last_frame.particles.position + vel * time_step / 2,
                    last_frame.configuration.box[None, :3])
                position.append(pos)

                # `(i, j)`-type pair data
                pair = pair_data_from_frame(
                    random.choice((last_frame, this_frame)),
                    args.i_type, args.j_type, args.num_neighs)
                pair_vectors.append(pair.vectors)
                pair_indices.append(pair.indices)

            # timesteps
            timestep.append(time_grid[:, None])

        print(f'done!')

    toc = time.monotonic()
    print(f'# time elapsed loading: {datetime.timedelta(seconds=toc-tic)}')

    # --- SAVE DATA TO FILE IN NUMPY FORMAT --- #

    if args.output_basename:
        np.savez(dirname.joinpath(args.output_basename.strip()),
            timestep=np.concatenate(timestep, axis=0).astype(np.float32),
            position=np.stack(position, axis=0).astype(np.float32),
            velocity=np.stack(velocity, axis=0).astype(np.float32),
            pair_vectors=np.stack(pair_vectors, axis=0).astype(np.float32),
            pair_indices=np.stack(pair_indices, axis=0).astype(np.int32))
