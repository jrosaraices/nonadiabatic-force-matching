#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import torch

import freud
import hoomd
import gsd.hoomd

from pathlib import Path

TINY = np.nextafter(np.float32(0), np.float32(1))


def generate_config(parameters: dict) -> gsd.hoomd.Frame:

    # Create the FCC lattice
    fcc = freud.data.UnitCell.fcc()
    box, position = fcc.generate_system(num_replicas=parameters["N_cells_per_side"])
    N_particles = len(position)

    # Define the density to work with and compute the size of the box
    target_density = parameters["ρ"]
    target_L = np.cbrt(N_particles / target_density)

    # Shift and rescale the positions of the particles to the new box
    position += [0.25, 0.25, 0.25]
    position *= [target_L, target_L, target_L] / box.L

    # Initialize the system for the simulation
    frame = gsd.hoomd.Frame()

    # Every lattice site is assigned a pair of particles, one mobile and the other frozen
    # One lattice site comprises solely a frozen particle (henceforth the 'carrier')
    # without a mobile copy; hence there are N * 2 - 1 particles in total
    frame.particles.N = N_particles * 2 - 1

    # We have N - 1 mobile particles and N frozen particles
    frame.particles.position = np.concatenate((position[:-1], position))

    # Mobile particles are of type 'A', and frozen particles are of type 'B' and 'C'
    frame.particles.types = ['A', 'B', 'C']

    # Last N particles are frozen ('B' and 'C')
    frame.particles.typeid = np.zeros(frame.particles.N)
    frame.particles.typeid[frame.particles.N // 2:] = 1

    # Last frozen particle is the carrier ('C')
    frame.particles.typeid[-1] = 2

    # Define a simple cubic box
    frame.configuration.box = [target_L, target_L, target_L, 0, 0, 0]

    # Each mobile ('A') particle is bonded to its frozen ('B') counterpart
    frame.bonds.N = N_particles - 1
    frame.bonds.types = ['A-B']
    frame.bonds.typeid = np.zeros(frame.bonds.N)
    frame.bonds.group = np.column_stack(
        [np.arange(frame.bonds.N), np.arange(frame.bonds.N) + frame.bonds.N])

    return frame


def setup_forces(parameters: dict) -> tuple[hoomd.md.pair.LJ, hoomd.md.bond.Harmonic]:

    σ = float(parameters["σ"])
    ε = float(parameters["ε"])
    κ = float(parameters["κ"])

    # Define a shifted LJ potential
    cell = hoomd.md.nlist.Cell(buffer=0.3 * σ, default_r_cut=1.3 * σ)
    lj = hoomd.md.pair.LJ(nlist=cell, mode='shift')

    # Mobile particles interact via the LJ potential
    lj.params[('A', 'A')] = dict(epsilon=ε, sigma=σ)
    lj.r_cut[('A', 'A')] = 2.7 * σ

    # Carrier interacts with mobile particles via the LJ potential
    lj.params[('A', 'C')] = lj.params[('C', 'C')] = lj.params[('A', 'A')]
    lj.r_cut[('A', 'C')] = lj.r_cut[('C', 'C')] = lj.r_cut[('A', 'A')]

    # Frozen particles do not interact via the LJ potential
    lj.params[('A', 'B')] = lj.params[('B', 'B')] = dict(epsilon=0, sigma=0)
    lj.r_cut[('A', 'B')] = lj.r_cut[('B', 'B')] = 0

    # Carrier does not interact with frozen particles
    lj.params[('B', 'C')] = dict(epsilon=0, sigma=0)
    lj.r_cut[('B', 'C')] = 0.0

    # Define a harmonic potential for bonds between frozen and mobile particles
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params['A-B'] = dict(k=κ, r0=0)

    return lj, harmonic


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
    lj, harmonic = setup_forces(parameters)

    # Define a Brownian integrator for the mobile particles
    method = hoomd.md.methods.Brownian(
        filter=hoomd.filter.Type('A'), kT=1 / float(parameters["β"]))
    integrator = hoomd.md.Integrator(
        dt=float(parameters["dt"]), methods=[method], forces=[lj, harmonic])
    simulation.operations.integrator = integrator

    # Create a switching parameter updater per the Vega--Noya schedule
    λ = hoomd.update.CustomUpdater(
        action=VegaNoyaSchedule(lj, harmonic, parameters),
        trigger=hoomd.trigger.Before(0))
    simulation.operations.updaters.append(λ)

    # Create a logger to output relevant observables
    logger = hoomd.logging.Logger()

    # Append mixing value to the simulation logger
    logger[('mixing.value')] = (lambda: λ.value, 'scalar')

    # Append lj energy to the simulation logger
    logger[('lj.energy')] = (lambda: lj.energy, 'scalar')

    # Append harmonic energy to the simulation logger
    logger[('harmonic.energy')] = (lambda: harmonic.energy, 'scalar')

    # Append mixing energy to the simulation logger
    def mixing_energy():
        harmonic_energy = harmonic.energy / λ.harmonic_scaling
        lj_energy = lj.energy / λ.lj_scaling
        return (harmonic_energy - lj_energy) * (-1 if λ.reversed else +1)
    logger[('mixing.energy')] = (mixing_energy, 'scalar')

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

    # Set the frequency for Vega--Noya switch updates
    λ, = simulation.operations.updaters
    λ.trigger = 1

    return simulation, logger


def setup_table_writer(simulation: hoomd.Simulation,
                       logger: hoomd.logging.Logger,
                       output_filename: str):

    # Instantiate a table logger for output of scalar and string quantities
    table_logger = hoomd.logging.Logger(categories=['scalar', 'string'])

    # Log the index of the **previous** simulation timestep
    # NOTE: the `hoomd.Simulation.run` logic is such that loggables written
    # during a timestep were computed during the timestep corresponding to
    # `simulation.timestep - 1`.
    logger[('simulation.timestep')] = (lambda: simulation.timestep - 1, 'scalar')

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


class VegaNoyaSchedule(hoomd.custom.Action):

    def __init__(self, lj: hoomd.md.pair.LJ,
                 harmonic: hoomd.md.bond.Harmonic,
                 parameters: dict):
        super().__init__()

        # LJ and harmonic force instances
        self.lj = lj
        self.harmonic = harmonic

        # Force amplitude parameters
        self.ε = float(parameters["ε"])
        self.κ = float(parameters["κ"])

        # Schedule parameters
        self.N_timesteps = int(parameters["N_timesteps"])
        self.reversed = bool(parameters["reversed"])
        self._value = float(parameters["λ"])

        # Schedule values
        if self.reversed:
            # Linear switching from from `λ` toward `0`
            self.λ, dλ = np.linspace(self._value, 0, self.N_timesteps+1,
                endpoint=True, retstep=True, dtype=np.float32)
        else:
            # Linear switching from from `λ` toward `1`
            self.λ, dλ = np.linspace(self._value, 1, self.N_timesteps+1,
                endpoint=True, retstep=True, dtype=np.float32)
        self.dλ = np.float32(dλ)

        # Set initial force amplitudes
        self.act(0)

    @property
    def lj_scaling(self):
        return max(self._value, TINY)

    @property
    def harmonic_scaling(self):
        return max(1 - self._value, TINY)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    def act(self, timestep):
        # Update force amplitudes only if `timestep` is within bounds
        if 0 <= timestep <= self.N_timesteps:
            self.value = self.λ[timestep]
            self.lj.params[('A', 'A')]['epsilon'] = self.lj_scaling * self.ε
            self.lj.params[('A', 'C')] = self.lj.params[('C', 'C')] = self.lj.params[('A', 'A')]
            self.harmonic.params['A-B']['k'] = self.harmonic_scaling * self.κ


if __name__ == '__main__':
    pass
