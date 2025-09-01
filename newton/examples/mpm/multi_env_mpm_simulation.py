#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Environment MPM Simulation for Benchmarking

This module provides a multi-environment version of the MPM particle pushing
simulation specifically designed for performance benchmarking. It supports
running multiple independent simulation environments in parallel to test
scaling characteristics.

Features:
- Multiple independent simulation environments
- Environment-specific particle spawning and plate motion
- Optimized for performance benchmarking
- Configurable environment spacing and parameters
- Headless execution support
"""

import numpy as np
import warp as wp
from typing import List, Tuple, Optional
import argparse

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM
from example_mpm_pushing_soil import _make_plate_mesh


class MultiEnvironmentMPMSimulation:
    """Multi-environment MPM simulation for performance benchmarking."""
    
    def __init__(self, num_envs: int, options: argparse.Namespace, viewer=None):
        self.num_envs = num_envs
        self.options = options
        self.viewer = viewer
        
        # Simulation parameters
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        
        # Environment spacing (meters)
        self.env_spacing = 15.0  # Sufficient space to prevent interference

        # Environment-specific data (initialize before building model)
        self.plate_meshes = []
        self.plate_rest_points = []
        self.plate_body_ids = []
        self.env_offsets = []

        # Build multi-environment model
        self.model = self._build_multi_env_model()
        
        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        
        # Setup MPM solver
        self._setup_solver()
        

        
        # Setup viewer if provided
        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True
    
    def _build_multi_env_model(self) -> newton.Model:
        """Build the multi-environment simulation model."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.gravity = wp.vec3(*self.options.gravity)
        
        # Calculate environment positions
        envs_per_row = int(np.ceil(np.sqrt(self.num_envs)))
        
        for env_id in range(self.num_envs):
            # Calculate environment offset
            row = env_id // envs_per_row
            col = env_id % envs_per_row
            
            env_offset = np.array([
                col * self.env_spacing,
                row * self.env_spacing,
                0.0
            ])
            self.env_offsets.append(env_offset)
            
            # Set current environment group
            builder.current_env_group = env_id
            
            # Add particles for this environment
            self._emit_particles_for_env(builder, env_offset)
            
            # Add plate for this environment
            self._add_plate_for_env(builder, env_offset, env_id)
        
        return builder.finalize()
    
    def _emit_particles_for_env(self, builder: newton.ModelBuilder, env_offset: np.ndarray):
        """Emit particles for a specific environment."""
        max_fraction = self.options.max_fraction
        voxel_size = self.options.voxel_size
        particles_per_cell = 3
        
        # Adjust particle bounds for this environment
        particle_lo = np.array(self.options.emit_lo) + env_offset
        particle_hi = np.array(self.options.emit_hi) + env_offset
        
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )
        
        self._spawn_particles_for_env(builder, particle_res, particle_lo, particle_hi, max_fraction)
    
    def _spawn_particles_for_env(self, builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, packing_fraction):
        """Spawn particles for a specific environment."""
        Nx, Ny, Nz = res
        
        px = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
        py = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
        pz = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)
        
        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T
        
        cell_size = (bounds_hi - bounds_lo) / res
        cell_volume = np.prod(cell_size)
        
        radius = np.max(cell_size) * 0.5
        volume = np.prod(cell_volume) * packing_fraction
        
        # Add some randomization
        rng = np.random.default_rng()
        points += 2.0 * radius * (rng.random(points.shape) - 0.5)
        vel = np.zeros_like(points)
        
        # Add particles to builder
        if hasattr(builder, 'particle_q') and builder.particle_q is not None:
            # Append to existing particles
            builder.particle_q = np.vstack([builder.particle_q, points])
            builder.particle_qd = np.vstack([builder.particle_qd, vel])
            builder.particle_mass = np.concatenate([
                builder.particle_mass, 
                np.full(points.shape[0], volume)
            ])
            builder.particle_radius = np.concatenate([
                builder.particle_radius,
                np.full(points.shape[0], radius)
            ])
            builder.particle_flags = np.concatenate([
                builder.particle_flags,
                np.zeros(points.shape[0], dtype=int)
            ])
        else:
            # First environment
            builder.particle_q = points
            builder.particle_qd = vel
            builder.particle_mass = np.full(points.shape[0], volume)
            builder.particle_radius = np.full(points.shape[0], radius)
            builder.particle_flags = np.zeros(points.shape[0], dtype=int)
    
    def _add_plate_for_env(self, builder: newton.ModelBuilder, env_offset: np.ndarray, env_id: int):
        """Add a plate for a specific environment."""
        # Plate parameters (same as original simulation)
        plate_width = 1.2    # Width along X-axis
        plate_length = 0.2   # Length along Y-axis  
        plate_height = 0.6   # Height along Z-axis
        
        # Position plate for this environment
        ground_level = 0.0
        plate_bottom = ground_level + 0.05
        plate_center_z = plate_bottom + plate_height / 2
        plate_center = np.array([0.0, -1.5, plate_center_z]) + env_offset
        
        # Create plate mesh
        plate_mesh = _make_plate_mesh(plate_width, plate_length, plate_height, plate_center)
        plate_rest_points = wp.array(
            plate_mesh.points.numpy() - plate_center, dtype=wp.vec3
        )
        
        # Add plate as kinematic body
        plate_body_id = builder.add_body(xform=wp.transform(plate_center, wp.quat_identity()))
        builder.add_shape_box(
            plate_body_id,
            hx=plate_width * 0.5,
            hy=plate_length * 0.5,
            hz=plate_height * 0.5,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),  # kinematic
        )
        
        # Store environment-specific data
        self.plate_meshes.append(plate_mesh)
        self.plate_rest_points.append(plate_rest_points)
        self.plate_body_ids.append(plate_body_id)
    
    def _setup_solver(self):
        """Setup the MPM solver with all colliders."""
        # Setup MPM solver options
        self.options.grid_padding = 0 if self.options.dynamic_grid else 5
        self.options.yield_stresses = wp.vec3(
            self.options.yield_stress,
            -self.options.stretching_yield_stress,
            self.options.compression_yield_stress,
        )
        
        self.solver = SolverImplicitMPM(self.model, self.options)
        
        # Setup colliders for all environments
        self.solver.setup_collider(self.model, colliders=self.plate_meshes)
        
        # Enrich states with MPM-specific fields
        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)
        
        # Set particle friction
        self.model.particle_mu = self.options.friction_coeff
    
    def simulate(self):
        """Run one simulation step for all environments."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._update_all_plates(self.sim_time, self.sim_dt)
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def step(self):
        """Execute one complete time step."""
        self.simulate()
        self.sim_time += self.frame_dt
    
    def render(self):
        """Render the simulation if viewer is available."""
        if self.viewer:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()
    
    def _update_all_plates(self, t: float, dt: float):
        """Update all plate positions with synchronized motion."""
        # Motion parameters
        amplitude = 2.0
        period = 8.0
        
        # Calculate motion offset
        s = np.sin(2.0 * np.pi * t / period)
        current_offset = amplitude * s
        
        # Update each environment's plate
        for env_id in range(self.num_envs):
            env_offset = self.env_offsets[env_id]
            plate_center = np.array([0.0, -1.5, 0.35]) + env_offset  # Base position
            
            # Calculate new position
            new_pos = np.array([
                plate_center[0],
                plate_center[1] + current_offset,
                plate_center[2]
            ])
            
            # Update collision mesh
            center0 = wp.vec3(plate_center[0], plate_center[1], plate_center[2])
            dir_axis = wp.vec3(0.0, 1.0, 0.0)
            
            wp.launch(
                self._move_plate_mesh_kernel,
                dim=self.plate_rest_points[env_id].shape[0],
                inputs=[
                    self.plate_rest_points[env_id],
                    self.plate_meshes[env_id].id,
                    center0,
                    dir_axis,
                    float(amplitude),
                    float(period),
                    float(t),
                    float(dt),
                ],
            )
            self.plate_meshes[env_id].refit()
            
            # Update visual body
            new_transform = wp.transform(new_pos, wp.quat_identity())
            
            if self.plate_body_ids[env_id] < self.model.body_count:
                wp.launch(
                    self._update_body_transform_kernel,
                    dim=self.model.body_count,
                    inputs=[self.state_0.body_q, self.plate_body_ids[env_id], new_transform],
                )
    
    @staticmethod
    @wp.kernel
    def _move_plate_mesh_kernel(
        rest_points: wp.array(dtype=wp.vec3),
        mesh_id: wp.uint64,
        center0: wp.vec3,
        dir_axis: wp.vec3,
        amplitude: float,
        period: float,
        t: float,
        dt: float,
    ):
        """Kernel to move plate mesh with smooth motion."""
        v = wp.tid()
        mesh = wp.mesh_get(mesh_id)
        
        s = wp.sin(2.0 * 3.14159 * t / period)
        
        cur_p = mesh.points[v] + dt * mesh.velocities[v]
        tgt_p = center0 + rest_points[v] + dir_axis * (amplitude * s)
        vel = (tgt_p - cur_p) / dt
        
        mesh.velocities[v] = vel
        mesh.points[v] = cur_p
    
    @staticmethod
    @wp.kernel
    def _update_body_transform_kernel(
        body_q: wp.array(dtype=wp.transform), 
        body_id: int, 
        new_transform: wp.transform
    ):
        """Kernel to update body transform."""
        if wp.tid() == body_id:
            body_q[body_id] = new_transform


def create_multi_env_simulation(num_envs: int, options: argparse.Namespace, viewer=None) -> MultiEnvironmentMPMSimulation:
    """Factory function to create multi-environment simulation."""
    return MultiEnvironmentMPMSimulation(num_envs, options, viewer)
