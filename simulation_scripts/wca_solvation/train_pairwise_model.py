#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import torch

from pairwise_model import device, PairwisePolicyModel, ModelLoss, ModelDataset

from tqdm import trange
from pathlib import Path
from typing import Tuple, Literal
from argparse import ArgumentParser

parser = ArgumentParser(
    prog=Path(__file__).name, usage='%(prog)s [options]',
    description='''Train a neural network model of the Fokker--Planck
                   velocity field sampled from trajectory data''')

# path search arguments
parser.add_argument('-npz_dirname', type=str, required=True,
                    help='path containing precured training data file', metavar=' ')
parser.add_argument('-npz_basename', type=str, required=True,
                    help='training data file basename (".npz")', metavar=' ')
parser.add_argument('-model_basename', type=str, default=None,
                    help='output model file basename (".pt")', metavar= ' ')

# model hyper-parameter arguments
parser.add_argument('-num_inputs', type=int, default=12,
                    help='number of inputs', metavar=' ')
parser.add_argument('-num_neurons', type=int, default=100,
                    help='number of neurons in hidden layers', metavar=' ')
parser.add_argument('-num_layers', type=int, default=1,
                    help='number of hidden layers', metavar=' ')
parser.add_argument('-num_epochs', type=int, default=10,
                    help='number of SGD sweeps through the dataset', metavar=' ')
parser.add_argument('-batch_size', type=int, default=10,
                    help='SGD minibatch size', metavar=' ')
parser.add_argument('-learning_rate', type=float, default=1e-4,
                    help='SGD learning rate', metavar=' ')

# -------------------------------------------------------
# FIT GRADIENT MODEL TO SAMPLED DIFFUSIVE VELOCITY FIELD
# -------------------------------------------------------

if __name__ == '__main__':

    args = parser.parse_args()

    # --- LOAD TRAJECTORY DATA --- #

    ObsData = np.ndarray[Tuple[int, Literal[1]], np.dtype[np.float64]]
    IdxData = np.ndarray[Tuple[int, int, Literal[2]], np.dtype[np.int32]]
    VecData = np.ndarray[Tuple[int, int, Literal[3]], np.dtype[np.float64]]

    dirname = Path(args.npz_dirname)
    data = np.load(dirname.joinpath(args.npz_basename))

    timestep: ObsData = data['timestep']
    position: VecData = data['position']
    velocity: VecData = data['velocity']

    pair_vectors: VecData = data['pair_vectors']
    pair_indices: IdxData = data['pair_indices']

    # --- SETUP MODEL AND TRAINING / VALIDATION DATASETS --- #

    model = PairwisePolicyModel(args.num_inputs, args.num_neurons, args.num_layers).to(device)
    dataset = ModelDataset(timestep, velocity, pair_vectors, pair_indices)
    criterion = ModelLoss().to(device)

    def init_params(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, a=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    model.apply(init_params)

    def collate_fn(data: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.stack(col, axis=0).to(device) for col in zip(*data))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(100, args.batch_size), eta_min=args.learning_rate / 100)

    # --- TRAIN PAIRWISE POLICY MODEL --- #

    # set up the training progress bar
    tqs = trange(args.num_epochs, desc="training : ", leave=True)
    desc = 'epoch : {:6d} | batch : {:6d} | loss : {:.6e} | rate : {:.6e}'

    for epoch_index in tqs:
        for batch_index, batch in enumerate(dataloader):
            # zero out model gradients
            optimizer.zero_grad()
            # evaluate loss on batched samples
            loss = criterion(model, *batch)
            # evaluate batch gradient wrt model paramaters
            loss.backward()
            # perform SGD step on model parameters
            optimizer.step()
            # update scheduler
            scheduler.step()
            # query current learning rate
            rate, = scheduler.get_last_lr()
            # update the training progress bar
            tqs.set_description(
                desc.format(epoch_index, batch_index, loss.item(), rate), refresh=True)

    # --- SAVE PAIRWISE POLICY MODEL --- #

    if (args.model_basename):
        torch.save(model, dirname.joinpath(args.model_basename.strip()))
