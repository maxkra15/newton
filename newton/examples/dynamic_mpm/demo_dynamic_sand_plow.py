#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Demo Script for Dynamic Sand Plow System
"""
Simple demonstration of the dynamic mesh-to-particle conversion system.

This script shows the core functionality without the full Newton simulation,
making it easier to understand and test the conversion algorithms.
"""

import numpy as np
import time
from typing import List, Tuple

# Import our modules
try:
    from dynamic_sand_plow_design import DynamicSandManager, SandState
    from mesh_sphere_intersection_simple import MeshSphereIntersector
    from mesh_particle_conversion import MeshParticleConverter, ParticleProperties
    from particle_mesh_reconstruction import ParticleMeshReconstructor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all the dynamic sand plow files are in the same directory.")
    exit(1)

def create_test_terrain() -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple test terrain mesh."""
    # Create a simple heightfield terrain
    resolution = 20
    size = 2.0
    
    x = np.linspace(-size/2, size/2, resolution)
    z = np.linspace(-size/2, size/2, resolution)
    xx, zz = np.meshgrid(x, z, indexing='ij')
    
    # Simple sine wave terrain
    heights = 0.1 * np.sin(xx * 2) * np.cos(zz * 2)
    
    # Generate vertices
    vertices = []
    for i in range(resolution):
        for j in range(resolution):
            vertices.append([xx[i, j], heights[i, j], zz[i, j]])
    
    vertices = np.array(vertices, dtype=np.float32)
    
    # Generate triangle indices
    indices = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Two triangles per quad
            v0 = i * resolution + j
            v1 = v0 + 1
            v2 = v0 + resolution
            v3 = v2 + 1
            
            indices.extend([v0, v1, v2, v1, v3, v2])
    
    indices = np.array(indices, dtype=np.int32)
    
    return vertices, indices

def simulate_plow_movement(
    sand_manager: DynamicSandManager,
    intersector: MeshSphereIntersector,
    converter: MeshParticleConverter,
    reconstructor: ParticleMeshReconstructor,
    terrain_vertices: np.ndarray,
    terrain_indices: np.ndarray
) -> None:
    """Simulate plow movement and demonstrate conversions."""
    
    print("🚜 Starting plow simulation...")
    print("=" * 50)
    
    # Plow path
    start_x, end_x = -0.8, 0.8
    plow_y = 0.2
    plow_z = 0.0
    num_steps = 10
    conversion_radius = 0.3
    
    # Track statistics
    total_conversions = 0
    total_particles_created = 0
    total_regions_affected = 0
    
    for step in range(num_steps):
        # Calculate plow position
        progress = step / (num_steps - 1)
        plow_x = start_x + progress * (end_x - start_x)
        plow_pos = np.array([plow_x, plow_y, plow_z])
        
        print(f"\nStep {step + 1}/{num_steps}: Plow at ({plow_x:.2f}, {plow_y:.2f}, {plow_z:.2f})")
        
        # Update conversion zone
        sand_manager.update_conversion_zone(plow_pos)
        
        # Get affected regions
        affected_regions = sand_manager.get_affected_regions(plow_pos, conversion_radius)
        total_regions_affected += len(affected_regions)
        
        print(f"  📍 {len(affected_regions)} regions affected by conversion zone")
        
        # Process each affected region
        step_conversions = 0
        step_particles = 0
        
        for region in affected_regions:
            if region.state == SandState.MESH and region.mesh_vertices is not None:
                # Check intersection
                intersects = intersector.check_intersection_exact(
                    plow_pos, conversion_radius,
                    region.mesh_vertices, region.mesh_indices
                )
                
                if intersects:
                    print(f"    🔄 Converting region {region.id} from MESH to PARTICLES")
                    
                    # Convert to particles
                    used_indices, positions = converter.convert_mesh_to_particles(
                        plow_pos, conversion_radius,
                        region.mesh_vertices, region.mesh_indices,
                        particle_spacing=0.1,
                        available_particle_indices=list(range(100))
                    )
                    
                    if len(used_indices) > 0:
                        region.state = SandState.PARTICLES
                        region.particle_indices = used_indices
                        step_conversions += 1
                        step_particles += len(used_indices)
                        
                        print(f"      ✓ Created {len(used_indices)} particles")
            
            elif region.state == SandState.PARTICLES and region.particle_indices is not None:
                # Check if plow has moved away
                region_center = (region.bounds_min + region.bounds_max) * 0.5
                distance = np.linalg.norm(plow_pos - region_center)
                
                if distance > conversion_radius * 1.5:  # Hysteresis
                    print(f"    🔄 Converting region {region.id} from PARTICLES to MESH")
                    
                    # Generate fake particle positions for demo
                    num_particles = len(region.particle_indices)
                    particle_positions = np.random.rand(num_particles, 3).astype(np.float32)
                    particle_positions = (particle_positions - 0.5) * 0.2 + region_center
                    
                    # Convert back to mesh
                    vertices, indices = reconstructor.reconstruct_mesh_from_particles(
                        particle_positions, (region.bounds_min, region.bounds_max),
                        method="heightfield", resolution=8
                    )
                    
                    # Smooth the mesh
                    vertices = reconstructor.smooth_mesh(vertices, indices, iterations=2)
                    
                    region.mesh_vertices = vertices
                    region.mesh_indices = indices
                    region.state = SandState.MESH
                    region.particle_indices = None
                    
                    print(f"      ✓ Reconstructed mesh with {len(vertices)} vertices")
        
        total_conversions += step_conversions
        total_particles_created += step_particles
        
        if step_conversions > 0:
            print(f"  ✨ Step summary: {step_conversions} conversions, {step_particles} particles created")
        else:
            print(f"  💤 No conversions needed this step")
        
        # Small delay for visualization
        time.sleep(0.1)
    
    print("\n" + "=" * 50)
    print("🏁 Simulation completed!")
    print(f"📊 Final statistics:")
    print(f"   • Total conversions: {total_conversions}")
    print(f"   • Total particles created: {total_particles_created}")
    print(f"   • Total regions affected: {total_regions_affected}")
    print(f"   • Average particles per conversion: {total_particles_created / max(1, total_conversions):.1f}")

def main():
    """Main demo function."""
    print("🌟 Dynamic Sand Plow System Demo")
    print("=" * 50)
    
    # Create test terrain
    print("🏔️  Creating test terrain...")
    terrain_vertices, terrain_indices = create_test_terrain()
    print(f"   Created terrain with {len(terrain_vertices)} vertices, {len(terrain_indices)//3} triangles")
    
    # Initialize sand management system
    print("\n🏗️  Initializing sand management system...")
    sand_bounds = (
        np.array([-1.0, 0.0, -1.0], dtype=np.float32),
        np.array([1.0, 0.2, 1.0], dtype=np.float32)
    )
    
    sand_manager = DynamicSandManager(
        sand_bounds=sand_bounds,
        region_size=0.4,
        max_particles=1000,
        conversion_radius=0.3
    )
    
    # Initialize terrain in regions
    for x_regions in sand_manager.regions:
        for y_regions in x_regions:
            for region in y_regions:
                # Use the test terrain for each region (simplified)
                region.mesh_vertices = terrain_vertices
                region.mesh_indices = terrain_indices
    
    print(f"   Created {len(sand_manager.regions)} x {len(sand_manager.regions[0])} x {len(sand_manager.regions[0][0])} region grid")
    
    # Initialize conversion components
    print("\n🔧 Initializing conversion components...")
    intersector = MeshSphereIntersector()
    converter = MeshParticleConverter(
        particle_properties=ParticleProperties(density=2000.0, friction=0.6)
    )
    reconstructor = ParticleMeshReconstructor()
    print("   All components ready!")
    
    # Run simulation
    print("\n🚀 Starting simulation...")
    simulate_plow_movement(
        sand_manager, intersector, converter, reconstructor,
        terrain_vertices, terrain_indices
    )
    
    print("\n✅ Demo completed successfully!")
    print("\nThis demo showed the core functionality of the dynamic sand plow system:")
    print("• Spatial partitioning of sand regions")
    print("• Mesh-sphere intersection detection")
    print("• Dynamic mesh-to-particle conversion")
    print("• Particle-to-mesh reconstruction")
    print("• State management with hysteresis")

if __name__ == "__main__":
    main()
