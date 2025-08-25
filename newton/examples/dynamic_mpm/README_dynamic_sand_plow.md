# Dynamic Sand Plow with Mesh-to-Particle Conversion

This implementation enhances the Newton sand plow example with dynamic mesh-to-particle conversion, creating a more realistic excavation simulation where sand starts as static mesh geometry and converts to MPM particles when disturbed by the plow.

## Overview

The dynamic sand plow system implements the following key features:

1. **Initial Setup**: Sand represented as static mesh geometry for performance
2. **Dynamic Conversion Zone**: 1-meter diameter spherical zone that follows the plow
3. **Mesh-to-Particle Conversion**: Automatic conversion of mesh regions to MPM particles when the plow approaches
4. **Realistic Sand Simulation**: Full MPM physics simulation for disturbed sand
5. **Particle-to-Mesh Conversion**: Conversion back to mesh when the plow moves away
6. **Mesh Reconstruction**: Updated mesh geometry reflects the sand's deformed state

## Architecture

### Core Components

#### 1. `dynamic_sand_plow_design.py`
- **DynamicSandManager**: Main coordinator for mesh-particle conversions
- **SandRegion**: Represents spatial regions that can be in mesh or particle state
- **ConversionZone**: Tracks the spherical zone that triggers conversions
- **SandState**: Enum for tracking region states (MESH, PARTICLES, TRANSITIONING)

#### 2. `mesh_sphere_intersection.py`
- **MeshSphereIntersector**: Efficient sphere-mesh intersection detection
- Implements both exact triangle-based and approximate raycast-based methods
- Uses Warp kernels for GPU acceleration

#### 3. `mesh_particle_conversion.py`
- **MeshParticleConverter**: Converts mesh regions to MPM particles
- Handles particle property calculation and activation
- Manages particle pool allocation and deactivation

#### 4. `particle_mesh_reconstruction.py`
- **ParticleMeshReconstructor**: Reconstructs mesh from particle positions
- Supports multiple reconstruction methods (heightfield, convex hull, alpha shapes)
- Includes mesh smoothing for natural appearance

#### 5. `example_dynamic_sand_plow.py`
- **DynamicSandPlowExample**: Main simulation class integrating all components
- Based on the original Newton sand plow example
- Adds dynamic conversion logic and state management

## Key Design Decisions

### Pre-allocated Particle Pool
Since Newton doesn't support runtime particle creation, the system pre-allocates a large pool of "dormant" particles that can be activated/deactivated as needed. Dormant particles are moved far away and flagged as inactive.

### Spatial Partitioning
Sand regions are organized in a 3D spatial grid for efficient intersection queries. Each region can independently transition between mesh and particle states.

### Hysteresis for Stability
The system uses hysteresis in conversion decisions - particles convert back to mesh only when the plow is significantly farther away than the initial conversion distance. This prevents oscillation between states.

### Performance Optimization
- GPU-accelerated intersection detection using Warp kernels
- Spatial grid reduces intersection queries to relevant regions only
- Configurable conversion check intervals balance accuracy and performance
- Multiple reconstruction methods allow quality/performance tradeoffs

## Usage

### Basic Usage

```bash
# Run with default settings
python newton/examples/example_dynamic_sand_plow.py

# Run with custom parameters
python newton/examples/example_dynamic_sand_plow.py \
    --conversion-radius 0.3 \
    --max-particles 25000 \
    --plow-speed 0.3 \
    --voxel-size 0.04
```

### Command Line Options

- `--conversion-radius`: Radius of conversion sphere (default: 0.5m)
- `--max-particles`: Maximum number of particles to pre-allocate (default: 50000)
- `--plow-speed`: Speed of plow movement in m/s (default: 0.5)
- `--voxel-size`: MPM voxel size (default: 0.05)
- `--headless`: Run without visualization
- `--dynamic-grid`: Enable dynamic MPM grid (GPU only)

### Testing

```bash
# Run comprehensive test suite
python newton/examples/test_dynamic_sand_plow.py
```

The test suite validates:
- Component initialization
- Mesh-sphere intersection detection
- Mesh-to-particle conversion accuracy
- Particle-to-mesh reconstruction quality
- Performance benchmarks
- Memory usage
- Full system integration

## Performance Considerations

### Recommended Settings

For real-time performance (60 FPS):
- `conversion_radius`: 0.3-0.5m
- `max_particles`: 25000-50000
- `voxel_size`: 0.04-0.06
- `conversion_check_interval`: 0.1s

For high quality (slower):
- `conversion_radius`: 0.5-0.8m
- `max_particles`: 75000-100000
- `voxel_size`: 0.03-0.04
- `conversion_check_interval`: 0.05s

### Performance Bottlenecks

1. **Particle Count**: More particles = better quality but slower simulation
2. **Conversion Frequency**: More frequent checks = smoother transitions but higher overhead
3. **Mesh Resolution**: Higher resolution meshes = better reconstruction but slower conversion
4. **MPM Voxel Size**: Smaller voxels = better accuracy but exponentially slower

### Optimization Tips

1. **Tune Particle Spacing**: Larger spacing reduces particle count while maintaining visual quality
2. **Use Dynamic Grid**: Enable for better MPM performance on GPU
3. **Adjust Check Interval**: Reduce conversion check frequency for better performance
4. **Limit Conversion Radius**: Smaller radius reduces particles but may miss interactions

## Limitations and Future Work

### Current Limitations

1. **No Runtime Particle Creation**: Must pre-allocate maximum particle count
2. **Simplified Mesh Reconstruction**: Uses basic algorithms (heightfield, convex hull)
3. **No Volume Conservation**: Particle-mesh conversion doesn't strictly preserve volume
4. **Limited Material Properties**: Single material type for all sand

### Future Enhancements

1. **Advanced Reconstruction**: Implement marching cubes or Poisson reconstruction
2. **Volume Conservation**: Add volume tracking and correction
3. **Multi-Material Support**: Different sand types with varying properties
4. **Adaptive Conversion**: Dynamic particle spacing based on deformation
5. **Temporal Coherence**: Smooth transitions between conversion states
6. **Memory Optimization**: Dynamic particle allocation if Newton supports it

## Technical Details

### Mesh-Sphere Intersection

The system uses two intersection methods:

1. **Exact Method**: Tests each triangle against the sphere using closest point calculation
2. **Raycast Method**: Casts rays from sphere surface to detect mesh intersection

### Particle Sampling

Particles are sampled from mesh regions using:
- Regular 3D grid within the conversion sphere
- Filtering based on proximity to mesh surface
- Jittering for natural particle distribution

### Mesh Reconstruction

Three reconstruction methods are available:

1. **Heightfield**: Creates terrain-like surface by interpolating particle heights
2. **Convex Hull**: Generates convex mesh enclosing all particles
3. **Alpha Shapes**: Creates detailed surface capturing concave features (simplified)

### State Management

Each sand region tracks:
- Current state (MESH, PARTICLES, TRANSITIONING)
- Mesh geometry (vertices, indices)
- Associated particle indices
- Bounding box and spatial location
- Last update timestamp

## Dependencies

- Newton Physics Framework
- Warp (for GPU acceleration)
- NumPy (for numerical operations)
- SciPy (for mesh reconstruction algorithms)
- Optional: psutil (for memory testing)

## File Structure

```
newton/examples/
├── example_dynamic_sand_plow.py          # Main simulation
├── dynamic_sand_plow_design.py           # Core architecture
├── mesh_sphere_intersection.py           # Intersection detection
├── mesh_particle_conversion.py           # Mesh-to-particle conversion
├── particle_mesh_reconstruction.py       # Particle-to-mesh reconstruction
├── test_dynamic_sand_plow.py            # Test suite
└── README_dynamic_sand_plow.md          # This documentation
```

## Contributing

When modifying the system:

1. Run the test suite to ensure functionality
2. Benchmark performance for any changes to core algorithms
3. Update documentation for new features or parameters
4. Consider backward compatibility with the original sand plow example

## License

This implementation follows the same Apache 2.0 license as the Newton framework.
