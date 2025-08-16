#!/usr/bin/env python3

import numpy as np
import torch

import abc
from typing import Optional, Literal, Tuple, NamedTuple

device = "cuda" if torch.cuda.is_available() else "cpu"

def _get_model_device(model):
    return next(model.parameters()).device

class PairData(NamedTuple):
    vectors: np.ndarray[Tuple[int, Literal[3]], np.dtype[np.float32]]
    indices: np.ndarray[Tuple[int, Literal[2]], np.dtype[np.int32]]

def periodic(r, L):
    return np.where(r >= L / 2, r - L, np.where(r < -L / 2, r + L, r))

def get_pairs_with_type_mask(
    positions: np.ndarray[Tuple[int, Literal[3]], np.dtype[np.float32]],
    types: np.ndarray[Tuple[int], np.dtype[np.int32]],
    pair_type: Tuple[int, int],
    box_sizes: np.ndarray[Literal[3], np.dtype[np.float32]],
    pair_list: Optional[np.ndarray[Tuple[int, Literal[2]], np.dtype[np.int32]]] = None,
    num_pairs: int = 12) -> PairData:

    # pair indices for all pairs
    if pair_list is None:
        pair_list = np.column_stack(np.triu_indices(len(positions), k=1))

    # sorted types
    type_i, type_j = sorted(pair_type)

    # pair indices labeled by type
    pair_labels = np.sort(types[pair_list], axis=1)

    # pair indices for pairs labeled `(type_i, type_j)`
    indices = pair_list[
        (pair_labels[:, 0] == type_i) & (pair_labels[:, 1] == type_j)]

    # displacement vectors for pairs labeled `(type_i, type_j)`
    vectors = periodic(
        positions[indices[:, 1]] - positions[indices[:, 0]], box_sizes[None])

    # sorting of displacement vectors by their length
    argsort = np.linalg.norm(vectors, axis=-1).argsort()

    return PairData(
        vectors=vectors[argsort][:min(len(argsort), num_pairs)],
        indices=indices[argsort][:min(len(argsort), num_pairs)])


class PairwisePotential(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def stack_inputs(self, time: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
        return torch.hstack([time, vecs.flatten(start_dim=1)])

    def unstack_inputs(self, stack: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        time, vecs = torch.hsplit(stack, [1])
        return time, vecs.reshape(len(time), -1, 3)

    def add_to_frame(self, frame: torch.Tensor, vec: torch.Tensor,
                     idx: torch.Tensor) -> None:
        index = (torch.arange(len(idx), dtype=idx.dtype).reshape(-1, 1, 1),
                 idx.unsqueeze(-1), torch.arange(3, dtype=idx.dtype).reshape(1, 1, -1))
        frame.index_put_(index, vec, accumulate=True)

    def energy(self, time: torch.Tensor, vecs: torch.Tensor,
               idxs: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        u = self.forward(self.stack_inputs(time, vecs))
        x = torch.zeros(shape, dtype=torch.float, device=u.device)
        self.add_to_frame(x, u[..., None].repeat(1, 1, 3) / 2, idxs[..., 0])
        self.add_to_frame(x, u[..., None].repeat(1, 1, 3) / 2, idxs[..., 1])
        return x

    def forces(self, time: torch.Tensor, vecs: torch.Tensor,
               idxs: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        f, = torch.autograd.grad(
            self.forward(self.stack_inputs(time, vecs)).sum(),
            vecs, create_graph=True, allow_unused=False)
        x = torch.zeros(shape, dtype=torch.float, device=f.device)
        self.add_to_frame(x, +f, idxs[..., 0])
        self.add_to_frame(x, -f, idxs[..., 1])
        return x


class PairwisePolicyModel(PairwisePotential):
    """ Pairwise policy model parameterized by a shallow, fully connected MLP. """
    def __init__(self, N_input: int, N_width: int, N_depth: int, cutoff_radius: float = 2):
        super().__init__()

        # fully connected MLP mixes pairwise energies
        self.model = torch.nn.Sequential(
            torch.nn.Linear(N_input, N_width, bias=False), torch.nn.CELU(0.1),
            *(N_depth * [torch.nn.Linear(N_width, N_width, bias=False), torch.nn.CELU(0.1)]),
            torch.nn.Linear(N_width, N_input, bias=False))

        # helpful contants and parameters
        self.π = np.pi
        self.R = cutoff_radius

    def forward(self, samples: torch.Tensor) -> torch.Tensor:

        # `samples` should be a batch of `[t, x]` tuples
        t, x = self.unstack_inputs(samples)

        # clipped and rescaled pair distances
        r = torch.clamp(torch.square(x).sum(dim=-1) / self.R ** 2, min=0, max=1)

        # pairwise interaction energies
        inputs = (1 + torch.cos(self.π * r)) * torch.exp(-r) * t / 2
        return self.model(inputs)


class ModelLoss(torch.nn.Module):
    """ Parametrically differentiable MSE loss between a PairwisePolicyModel ("model")
    and a targeted vector field for its first derivative ("velocity"). """
    def __init__(self):
        super().__init__()

    def forward(self, model: PairwisePolicyModel,
                timestep: torch.Tensor, velocity: torch.Tensor,
                pair_vectors: torch.Tensor, pair_indices: torch.Tensor):

        # mean-squared loss between forces and velocities
        forces = model.forces(timestep, pair_vectors, pair_indices, velocity.shape)
        forces_loss = torch.mean(torch.square(forces) - 2 * forces * velocity)
        return forces_loss


class ModelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 timestep: np.ndarray[Tuple[int, Literal[1]], np.dtype[np.float32]],
                 velocity: np.ndarray[Tuple[int, int, Literal[3]], np.dtype[np.float32]],
                 pair_vectors: np.ndarray[Tuple[int, int, Literal[3]], np.dtype[np.float32]],
                 pair_indices: np.ndarray[Tuple[int, int, Literal[2]], np.dtype[np.int32]]):
        super().__init__()

        self.timestep = torch.as_tensor(timestep).float()
        self.velocity = torch.as_tensor(velocity).float()
        self.pair_vectors = torch.as_tensor(pair_vectors).float().requires_grad_()
        self.pair_indices = torch.as_tensor(pair_indices).int()

    def __len__(self) -> int:
        return len(self.timestep)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return (self.timestep[index], self.velocity[index],
                self.pair_vectors[index], self.pair_indices[index])


def model_parameter_gradient_norm(model):

    device = _get_model_device(model)
    grads = [param.grad.detach().flatten().to(device)
             for param in model.parameters()
             if param.grad is not None]

    if grads:
        return torch.cat(grads).norm().item()
    else:
        return 0.0


if __name__ == '__main__':
    pass
