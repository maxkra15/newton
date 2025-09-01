# Newton MPM Performance Benchmarking System

This comprehensive benchmarking system evaluates the performance characteristics of Newton's Material Point Method (MPM) particle simulation across multiple parameter configurations and environment counts.

## Features

- **Configurable Parameters**: Test different particle sizes, grid resolutions, solver settings, and material properties
- **Multi-Environment Support**: Evaluate performance scaling with parallel simulation environments
- **GPU Performance Monitoring**: Track memory usage, utilization, and kernel execution times
- **Physics Accuracy Metrics**: Monitor particle conservation and energy stability
- **Automated Report Generation**: Generate comprehensive PDF reports with visualizations
- **Headless Execution**: Run benchmarks without GUI for automated testing

## Quick Start

### Basic Usage

```bash
# Quick benchmark (fast, basic parameter sweep)
python benchmark_mpm_performance.py --config quick

# Comprehensive benchmark (thorough analysis, takes longer)
python benchmark_mpm_performance.py --config comprehensive --generate-report

# Multi-environment scaling analysis only
python benchmark_mpm_performance.py --scaling-only --env-counts 1 2 4 8 16
```

### Custom Configuration

```bash
# Use custom parameter configuration
python benchmark_mpm_performance.py --custom-params benchmark_configs/scaling_analysis.json

# Specify output directory
python benchmark_mpm_performance.py --config quick --output results/quick_test/
```

## Configuration Files

The system includes predefined configurations in `benchmark_configs/`:

- **`quick_test.json`**: Fast validation with basic parameter sweep
- **`comprehensive.json`**: Thorough analysis with extensive parameter ranges
- **`scaling_analysis.json`**: Focus on multi-environment scaling performance

### Custom Configuration Format

```json
{
  "description": "Custom benchmark configuration",
  "base_config": {
    "voxel_size": 0.05,
    "particles_per_cell": 3,
    "num_frames": 100,
    "tolerance": 1e-5,
    "max_iterations": 250,
    "friction_coeff": 0.6,
    "dynamic_grid": true
  },
  "param_ranges": {
    "voxel_size": [0.02, 0.05, 0.1],
    "particles_per_cell": [2, 3, 4],
    "tolerance": [1e-6, 1e-5, 1e-4]
  },
  "env_counts": [1, 2, 4, 8, 16]
}
```

## Key Parameters

### Core MPM Parameters
- **`voxel_size`**: Grid resolution (smaller = higher accuracy, more computation)
- **`particles_per_cell`**: Particle density (higher = more particles, better accuracy)
- **`max_fraction`**: Particle packing fraction (affects particle count)

### Solver Parameters
- **`tolerance`**: Solver convergence tolerance (smaller = more accurate, slower)
- **`max_iterations`**: Maximum solver iterations per step
- **`substeps`**: Number of physics substeps per frame

### Environment Parameters
- **`num_envs`**: Number of parallel simulation environments
- **`emit_lo/emit_hi`**: Particle spawn area bounds

### Material Properties
- **`friction_coeff`**: Particle-surface friction coefficient
- **`yield_stress`**: Material yield stress for plasticity
- **`poisson_ratio`**: Material Poisson's ratio

## Performance Metrics

The benchmarking system tracks:

### Timing Metrics
- Average step time and solver time
- Steps per second and particles per second
- Multi-environment throughput

### GPU Metrics
- GPU utilization percentage
- GPU memory usage (current and peak)
- Kernel execution timing

### Physics Metrics
- Particle conservation error
- Energy drift over time
- Solver convergence rates

### System Metrics
- CPU usage and RAM consumption
- Scaling efficiency across environments

## Report Generation

Generate comprehensive PDF reports with:

```bash
python benchmark_mpm_performance.py --config comprehensive --generate-report
```

Reports include:
- Performance vs accuracy trade-off analysis
- Multi-environment scaling efficiency
- Parameter sensitivity heatmaps
- Optimization recommendations
- Detailed configuration tables

### Report Dependencies

Install additional dependencies for PDF report generation:

```bash
pip install matplotlib reportlab pandas seaborn
```

## Multi-Environment Simulation

The system automatically uses multi-environment simulation for `num_envs > 1`:

- **Environment Isolation**: Each environment has independent particles and plates
- **Spatial Separation**: Environments are spaced to prevent interference
- **Synchronized Motion**: All plates move in sync for consistent benchmarking
- **Optimized Collision**: Single solver handles all environments efficiently

## Optimization Guidelines

### For Maximum Throughput
- Use larger voxel sizes (0.1-0.2m)
- Moderate particle density (2-3 particles per cell)
- Multiple environments to saturate GPU
- Relaxed solver tolerance (1e-4 to 1e-5)

### For Maximum Accuracy
- Smaller voxel sizes (0.02-0.05m)
- Higher particle density (4-5 particles per cell)
- Tight solver tolerance (1e-6)
- More solver iterations (300+)

### For Balanced Performance
- Medium voxel size (0.05-0.08m)
- Standard particle density (3 particles per cell)
- Moderate tolerance (1e-5)
- Environment count matching GPU capacity

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce particle count, increase voxel size, or reduce environments
2. **Poor GPU Utilization**: Increase particle count or add more environments
3. **Slow Performance**: Increase voxel size, reduce solver iterations, or use fewer particles
4. **Convergence Issues**: Reduce time step, increase solver iterations, or adjust material properties

### Performance Tips

- Run warm-up iterations before benchmarking
- Use headless mode for automated testing
- Monitor GPU memory usage to avoid swapping
- Consider CUDA graph optimization for repeated runs

## Example Workflows

### Development Testing
```bash
# Quick validation during development
python benchmark_mpm_performance.py --config quick --verbose
```

### Production Benchmarking
```bash
# Comprehensive analysis for optimization
python benchmark_mpm_performance.py --config comprehensive --generate-report --output production_results/
```

### Scaling Analysis
```bash
# Test multi-environment scaling
python benchmark_mpm_performance.py --scaling-only --env-counts 1 2 4 8 16 32 64
```

### Custom Parameter Study
```bash
# Study specific parameter ranges
python benchmark_mpm_performance.py --custom-params my_config.json --generate-report
```

## Integration with Original Simulation

The benchmarking system maintains full compatibility with the original MPM pushing simulation:

```bash
# Original simulation still works independently
python example_mpm_pushing_soil.py --viewer gl

# Benchmarking runs headless by default
python benchmark_mpm_performance.py --config quick
```

This ensures that development and testing workflows remain unaffected while providing comprehensive performance analysis capabilities.
