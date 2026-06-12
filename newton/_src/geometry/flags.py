# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum


# Particle flags
class ParticleFlags(IntEnum):
    """
    Flags for particle properties.
    """

    ACTIVE = 1 << 0
    """Indicates that the particle is active."""

    FLUID = 1 << 1
    """Indicates that the particle is part of a fluid.

    Fluid particles generate position-based fluid density constraints against
    other fluid particles instead of pairwise contact constraints in solvers
    that support fluids (see :class:`newton.solvers.SolverXPBD`). Interactions
    with non-fluid particles and shapes are still handled as regular contacts.
    """


# Shape flags
class ShapeFlags(IntEnum):
    """
    Flags for shape properties.
    """

    VISIBLE = 1 << 0
    """Indicates that the shape is visible."""

    COLLIDE_SHAPES = 1 << 1
    """Indicates that the shape collides with other shapes."""

    COLLIDE_PARTICLES = 1 << 2
    """Indicates that the shape collides with particles."""

    SITE = 1 << 3
    """Indicates that the shape is a site (non-colliding reference point)."""

    HYDROELASTIC = 1 << 4
    """Indicates that the shape uses hydroelastic collision."""


__all__ = [
    "ParticleFlags",
    "ShapeFlags",
]
