#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import gsd.hoomd
import hoomd
import torch
import cupy

from pathlib import Path

from pairwise_model import PairwisePolicyModel, get_pairs_with_type_mask

def periodic(r, L):
    return np.where(r >= L / 2, r - L, np.where(r < -L / 2, r + L, r))


def sc_lattice(L, N):
    return float(L) * np.indices(3 * (int(N),)).reshape(3, -1).T


def generate_config(parameters: dict):

    # Set solvent configuration as simple cubic box at target density
    position = sc_lattice(
        parameters["σ"] / np.cbrt(parameters["ρ"]), parameters["N_particles"])

    # Add solute at the origin as the zeroth particle in the array
    position = np.concatenate([
        parameters["σ"] / np.cbrt(parameters["ρ"]) * np.array([[0.5, 0.5, 0.5]]), position])

    # Initialize the hoomd system state container
    frame = gsd.hoomd.Frame()

    # System lives in a cubic box
    L = parameters["N_particles"] / np.cbrt(parameters["ρ"])
    frame.configuration.box = [L, L, L, 0, 0, 0]
    frame.particles.N = len(position)
    frame.particles.position = periodic(position, L)
    frame.particles.orientation = np.asarray([(1, 0, 0, 0)] * frame.particles.N)

    # We have 1 solute particle ('A') and N - 1 solvent particles ('B')
    frame.particles.types = ['A', 'B']
    frame.particles.typeid = np.zeros(frame.particles.N)
    frame.particles.typeid[1:] = 1

    return frame


def sigma_mixing(φ: float, λ: float):
    return np.sqrt((0.1 + 0.9 * λ) * φ)


def sigma_mixing_derivative(φ: float, λ: float):
    # NOTE: Given the definition of `sigma_mixing` right above this function,
    # the factor `0.495` in this function is wrong and should be `0.45`.
    # Too late to change at the moment, but luckily we've been wrong consistently.
    # To correct erroneous outputs for the mixing energy, using that the observable `dUdλ` is linear
    # in the output of this function, simply multiply it by `0.45 / 0.495`.
    return 0.495 * λ / sigma_mixing(φ, λ)


def setup_wca_force(parameters: dict):

    if 0.0 > float(parameters["λ"]) > 1.0:
        raise ValueError(f'Value `λ` ({parameters["λ"]}) must be in [0, 1]')

    sigma_AA = parameters["σ"] * parameters["φ"]
    sigma_AB = parameters["σ"] * sigma_mixing(parameters["φ"], float(parameters["λ"]))
    sigma_BB = parameters["σ"]

    # Instantiate a pair potential for solvent-solvent interactions
    nlist = hoomd.md.nlist.Cell(buffer=0.2, default_r_cut=3.2)
    wca = hoomd.md.pair.LJ(nlist=nlist, mode='shift', tail_correction=False)

    # Solvent particles interact amongst themselves via WCA
    wca.params[('B', 'B')] = dict(epsilon=parameters["ε"], sigma=sigma_BB)
    wca.r_cut[('B', 'B')] = 2 ** (1 / 6) * sigma_BB

    # Solvent particles interact with the solute via WCA (geometric-mean mixing)
    wca.params[('A', 'B')] = dict(epsilon=parameters["ε"], sigma=sigma_AB)
    wca.r_cut[('A', 'B')] = 2 ** (1 / 6) * sigma_AB

    # Solute particles do not interact via WCA
    wca.params[('A', 'A')] = dict(epsilon=parameters["ε"], sigma=sigma_AA)
    wca.r_cut[('A', 'A')] = 0.0

    return wca


def setup_mixing_table(σ: float, ε: float, φ: float, λ: float):

    def dUdλ(r):
        return 4 * ε * (
            12 * (σ / r) ** 12 * sigma_mixing(φ, λ) ** 11 * sigma_mixing_derivative(φ, λ) -
            6 * (σ / r) ** 6 * sigma_mixing(φ, λ) ** 5 * sigma_mixing_derivative(φ, λ))

    def dFdλ(r):
        return 0.0 * r

    r_cut = 2 ** (1 / 6) * σ * sigma_mixing(φ, λ)
    r = np.linspace(0.0, r_cut, num=1001, endpoint=True)[1:]

    return dict(points=r, energy=dUdλ(r), forces=dFdλ(r))


def setup_mixing_force(σ: float, ε: float, φ: float,
                       λ: float, force: hoomd.md.pair.Table):

    # Define the mixing potential between solute (dimer) and solvent particles
    table = setup_mixing_table(σ, ε, φ, λ)

    force.params[('A', 'B')] = dict(
        U=table['energy'], F=table['forces'], r_min=table['points'].min())
    force.r_cut[('A', 'B')] = table['points'].max()

    force.params[('A', 'A')] = force.params[('B', 'B')] = dict(U=[0], F=[0], r_min=0.0)
    force.r_cut[('A', 'A')] = force.r_cut[('B', 'B')] = 0.0

    return force


def setup_simulation(parameters: dict,
                     device: hoomd.device.Device,
                     random: np.random.Generator,
                     frame: gsd.hoomd.Frame):

    # Define the simulation object
    seed = random.integers(65536)
    simulation = hoomd.Simulation(device=device, seed=seed)
    simulation.timestep = 0
    simulation.create_state_from_snapshot(frame)

    # Define force-fields for solute-solvent and solvent-solvent interactions
    U = setup_wca_force(parameters)

    # Define a Brownian integrator for all particles
    method = hoomd.md.methods.Brownian(
        filter=hoomd.filter.All(), kT=1.0 / parameters["β"])
    integrator = hoomd.md.Integrator(
        dt=parameters["dt"], methods=[method], forces=[U])
    simulation.operations.integrator = integrator

    # Add a compute for the mixing energy of solvation
    dUdλ = setup_mixing_force(
        parameters["σ"], parameters["ε"],
        parameters["φ"], float(parameters["λ"]), hoomd.md.pair.Table(U.nlist))
    simulation.operations.computes.append(dUdλ)

    # Create updaters for solute-solvent and solvent-solvent energies
    solvent = hoomd.update.CustomUpdater(
        action=PairTypeEnergy(U, simulation, 'B', 'B'), trigger=1)
    solution = hoomd.update.CustomUpdater(
        action=PairTypeEnergy(U, simulation, 'A', 'B'), trigger=1)
    solvation = hoomd.update.CustomUpdater(
        action=PairTypeEnergy(dUdλ, simulation, 'A', 'B'), trigger=1)
    simulation.operations.updaters.extend([solvation, solution, solvent])

    # Create a logger for observable data (namely, energies and pair indices)
    logger = hoomd.logging.Logger()
    logger.add(solvation, quantities=['energy'], user_name='dUdλ_AB')
    logger.add(solution, quantities=['energy'], user_name='U_AB')
    logger.add(solvent, quantities=['energy'], user_name='U_BB')

    # NOTE: Since `operations.updaters` in a hoomd simulation are evaluated
    # _before_ updating the state of the system in the current timestep, the
    # energy data stored in timestep `i > 0` corresponds to the
    # simulation state at timestep `i - 1`.
    return simulation, logger


def setup_equilibrium_simulation(parameters: dict,
                                 device: hoomd.device.Device,
                                 random: np.random.Generator,
                                 input_filename: str | None = None):

    # Read simulation state from input frame in all ranks
    if input_filename:
        frame = gsd.hoomd.open(input_filename)[-1]
    else:
        frame = generate_config(parameters)

    # Set up core simulation details
    simulation, logger = setup_simulation(parameters, device, random, frame)

    return simulation, logger


def setup_nonequilibrium_simulation(parameters: dict,
                                    device: hoomd.device.Device,
                                    random: np.random.Generator,
                                    input_filename: str,
                                    input_frame_index: int | None = None):

    # Read initial configuration from file in all ranks
    frames = gsd.hoomd.open(input_filename)
    if input_frame_index:
        frame = frames[input_frame_index]
    else:
        frame = frames[random.integers(len(frames), dtype=int)]

    # Set up core simulation details
    simulation, logger = setup_simulation(parameters, device, random, frame)

    # Add linear rampup schedule to solute size and solute-solvent interaction energy
    U = simulation.operations.integrator.forces[0]
    dUdλ = simulation.operations.computes[0]
    wca_parameter_updater = hoomd.update.CustomUpdater(
        action=LinearSoluteSchedule(U, dUdλ, ('A', 'B'), parameters), trigger=1)
    simulation.operations.updaters.append(wca_parameter_updater)

    # Add mixing energy observable to logger
    logger[('wca_monomer.mixing.energy')] = (
        λ: dUdλ.energy * (-1 if parameters["reversed"] else +1), 'scalar')

    return simulation, logger


def setup_nonequilibrium_simulation_with_policy(parameters: dict,
                                                device: hoomd.device.Device,
                                                random: np.random.Generator,
                                                input_filename: str,
                                                model_filename: str):

    # Set up core simulation details
    simulation, logger = setup_nonequilibrium_simulation(
        parameters, device, random, input_filename, input_frame_index=-1)

    # Add PyTorch policy force to the MD integrator
    U = simulation.operations.integrator.forces[0]
    policy_force = PairwisePolicy(
        model_filename, U.nlist, simulation.state.get_snapshot(), parameters)
    simulation.operations.integrator.forces.append(policy_force)

    # Add policy force loggables to the simulation logger
    logger.add(policy_force, user_name='policy', quantities=['energies', 'forces'])
    logger[('wca_monomer.policy.energy')] = (λ: policy_force.energy, 'scalar')
    logger[('wca_monomer.policy.squared_force')] = (λ: np.sum(policy_force.forces ** 2), 'scalar')

    return simulation, logger


def setup_table_writer(simulation, logger, output_filename):

    # Instantiate a table logger for output of scalar and string quantities
    table_logger = hoomd.logging.Logger(categories=['scalar', 'string'])

    # Log the index of the **previous** simulation timestep
    # NOTE: the `hoomd.Simulation.run` logic is such that loggables written
    # during a timestep were computed during the timestep corresponding to
    # `simulation.timestep - 1`.
    table_logger.add(simulation, quantities=['timestep'])

    # Copy loggables of the scalar and string category into the table logger
    for key, val in logger.items():
        if val.category in table_logger.categories:
            table_logger[key] = val

    # Define a table writer that will write the logger with the specified filename
    output_filename_txt = Path(output_filename_gsd).with_suffix('.txt')
    table_writer = hoomd.write.Table(
        trigger=trigger, logger=table_logger, pretty=True, max_precision=10,
        output=open(output_filename_txt, mode='w', encoding='utf-8', newline='\n'))

    # Append the table writer to the simulation object
    simulation.operations.writers.append(table_writer)

    return table_writer


class LinearSoluteSchedule(hoomd.custom.Action):

    def __init__(self, U, dUdλ, pair_key, parameters):
        super().__init__()

        self.U = U
        self.dUdλ = dUdλ
        self.pair_key = pair_key

        self.φ = parameters['φ']
        self.σ = parameters['σ']
        self.ε = parameters['ε']
        self.N_timesteps = parameters['N_timesteps']
        self.reversed = parameters['reversed']
        self.λ = float(parameters['λ'])

        self.act(0)

    def act(self, timestep):

        if self.reversed:
            extent = self.λ * (1.0 - timestep / self.N_timesteps)
        else:
            extent = (1.0 - self.λ) * timestep / self.N_timesteps
        σ = self.σ * sigma_mixing(self.φ, extent)

        self.U.params[self.pair_key]['sigma'] = σ
        self.U.r_cut[self.pair_key] = 2 ** (1 / 6) * σ
        self.dUdλ = setup_mixing_force(self.σ, self.ε, self.φ, extent, self.dUdλ)


class PairTypeEnergy(hoomd.custom.Action):

    def __init__(self, force, simulation, type_i, type_j, scale=1.0):
        super().__init__()

        self._force = force
        self._simulation = simulation
        self._scale = scale

        self.type_i = type_i
        self.type_j = type_j

        self.pair_type_energy = 0.0

    def attach(self, simulation):
        with simulation._state.cpu_local_snapshot as snap:
            self.tag_i = snap.particles.tag[(snap.particles.typeid ==
                 simulation.state.particle_types.index(self.type_i))]
            if self.type_i == self.type_j:
                self.tag_i, self.tag_j = np.split(self.tag_i, [1])
            else:
                self.tag_j = snap.particles.tag[(snap.particles.typeid ==
                     simulation.state.particle_types.index(self.type_j))]

    def act(self, timestep):
        if timestep >= 0:
            self.pair_type_energy = self._force.compute_energy(
                self.tag_i.astype(np.int32), self.tag_j.astype(np.int32))

    @hoomd.logging.log(category='scalar')
    def energy(self):
        return self.pair_type_energy * self._scale


class PairwisePolicy(hoomd.md.force.Custom):
    """Build pair force by combining hoomd.md.pair.Pair with a
    PyTorch-trained model for time-dependent interaction parameters."""
    device = torch.device("cpu")

    def __init__(self, model_path, nlist, state, parameters):
        super().__init__(aniso=False)

        # Setup PyTorch model inputs (timeslice values)
        self.time = torch.arange(parameters["N_timesteps"], -1, -1,
            device=self.device, dtype=torch.float) / parameters["N_timesteps"]

        # Neighbor list instance used to compute forces
        self._nlist = nlist

        # Number of neighboring pairs considered in force evaluation
        self.num_pairs = parameters["N_neighs"]

        # Types comprising the pair type `(I, J)` for which the force is computed
        self.typeid = tuple(sorted(
            [state.particles.types.index(parameters["i_type"]),
             state.particles.types.index(parameters["j_type"])]))

        # load PyTorch model trained on this machine
        self.model = torch.load(model_path, weights_only=False, map_location=self.device)

        # NOTE: Comment out the line above and instead use this code to load a PyTorch model
        # trained on a different machine (by copying its `state_dict` attribute onto a locally instantiated model)
        # model = torch.load(model_path, weights_only=False, map_location=self.device)
        # self.model = PairwisePolicyModel(self.num_pairs, 4 * self.num_pairs, 1).to(self.device)
        # self.model.load_state_dict(model.state_dict())

    def set_forces(self, timestep):
        # PyTorch model inputs are (time, vecs, idxs, shape) where
        # - `time` is the fractional timestep (in the range [0, 1])
        # - `vecs` are the displacement vectors between neighboring pairs
        # - `idxs` are the indices of neighboring pairs
        # - `shape` is the shape of the force array
        with (self._state.cpu_local_snapshot as state,
              self._nlist.cpu_local_nlist_arrays as nlist,
              self.cpu_local_force_arrays as force):

            # Auxiliary variables to extract pair list data
            n_neigh = nlist.n_neigh
            head_list = nlist.head_list
            neighs_iter = zip(head_list, n_neigh)

            # Pair list data
            pair_list = np.column_stack([
                np.repeat(np.arange(len(n_neigh)), n_neigh),
                np.concatenate([nlist.nlist[i:i+n] for (i, n) in neighs_iter]),])

            # Displacement vectors and indices of nearest-neighboring pairs
            vectors, indices = get_pairs_with_type_mask(
                state.particles.position, state.particles.typeid, self.typeid,
                state.global_box.L, pair_list=pair_list, num_pairs=self.num_pairs)

            time = self.time[timestep].reshape(1, -1)
            vecs = torch.as_tensor(vectors,
                dtype=torch.float, device=self.device).unsqueeze(0).requires_grad_()
            idxs = torch.as_tensor(indices,
                dtype=torch.int, device=self.device).unsqueeze(0)
            shape = (1,) + force.force.shape

            # Evaluate model to obtain per-particle energy and force
            force.potential_energy = 2 * self.model.energy(time, vecs, idxs,
                shape).squeeze().detach().cpu().numpy()[:, 0]
            force.force = -2 * self.model.forces(time, vecs.requires_grad_(), idxs,
                shape).squeeze().detach().cpu().numpy()[:, :]


if __name__ == '__main__':
    pass
