# Newton MPM Benchmarking System - Complete Guide

## Quick Start

### 1. Generate PDF Report from Existing Results

If you already have benchmark results, generate a PDF report:

```bash
# Generate PDF from existing results
uv run benchmark_report_generator.py benchmark_results/benchmark_results.json --output benchmark_results/

# The PDF will be created as: benchmark_results/mmp_benchmark_report_YYYYMMDD_HHMMSS.pdf
```

### 2. Run Realistic Comprehensive Benchmarks

For production-ready benchmarking with realistic parameter values:

```bash
# Quick test (2-4 scenarios, ~5 minutes)
uv run run_realistic_benchmark.py --quick-test --verbose

# Test specific scenario types
uv run run_realistic_benchmark.py --scenario accuracy_levels    # Different accuracy levels
uv run run_realistic_benchmark.py --scenario materials         # Different material types  
uv run run_realistic_benchmark.py --scenario sensitivity       # Parameter sensitivity
uv run run_realistic_benchmark.py --scenario scaling          # Multi-environment scaling

# Full comprehensive test (all scenarios, ~30-60 minutes)
uv run run_realistic_benchmark.py --scenario all --verbose
```

## Understanding the Results

### Performance Metrics Explained

**Throughput Metrics:**
- **Particles/second**: Total particle simulation throughput (higher = better)
- **Steps/second**: Simulation timesteps per second (higher = better)
- **Environments/second**: Multi-environment throughput

**GPU Metrics:**
- **GPU Utilization**: Percentage of GPU compute used (70-90% is optimal)
- **GPU Memory**: VRAM usage in GB (monitor to avoid out-of-memory)

**Physics Metrics:**
- **Particle Count**: Total particles in simulation
- **Solver Convergence**: How well the physics solver converges

### Realistic Scenarios Tested

**1. Accuracy Levels:**
- **High Accuracy Research**: 0.01m voxels, 6 particles/cell, tight tolerance
- **Production Simulation**: 0.025m voxels, 4 particles/cell, balanced settings
- **Real-time Preview**: 0.05m voxels, 3 particles/cell, relaxed tolerance
- **Interactive Demo**: 0.1m voxels, 2 particles/cell, very fast

**2. Material Types:**
- **Wet Clay**: High friction (0.9), high yield stress (8000 Pa)
- **Dry Sand**: Medium friction (0.7), medium yield stress (2000 Pa)  
- **Wet Soil**: High friction (0.8), medium yield stress (5000 Pa)
- **Loose Gravel**: Low friction (0.6), low yield stress (1000 Pa)

**3. Parameter Sensitivity:**
- Voxel size: 0.01m to 0.1m (accuracy vs performance)
- Particles per cell: 2-6 (quality vs computation)
- Solver tolerance: 1e-6 to 1e-4 (precision vs speed)
- Substeps: 1-4 (stability vs performance)

## Performance Optimization Guidelines

### For Maximum Throughput
```json
{
  "voxel_size": 0.05,           // Larger voxels = fewer grid points
  "particles_per_cell": 3,      // Moderate particle density
  "tolerance": 1e-4,            // Relaxed solver tolerance
  "max_iterations": 100,        // Fewer solver iterations
  "substeps": 1,                // Single substep per frame
  "num_envs": 4                 // Multiple environments for GPU saturation
}
```

### For Maximum Accuracy
```json
{
  "voxel_size": 0.01,           // Fine grid resolution
  "particles_per_cell": 6,      // High particle density
  "tolerance": 1e-6,            // Tight solver tolerance
  "max_iterations": 300,        // More solver iterations
  "substeps": 4,                // Multiple substeps for stability
  "num_envs": 1                 // Single environment for precision
}
```

### For Balanced Production Use
```json
{
  "voxel_size": 0.025,          // Good balance of accuracy/speed
  "particles_per_cell": 4,      // Standard particle density
  "tolerance": 5e-6,            // Moderate solver tolerance
  "max_iterations": 200,        // Reasonable solver iterations
  "substeps": 2,                // Some stability improvement
  "num_envs": 2                 // Light multi-environment use
}
```

## Performance Targets

### Minimum Acceptable Performance
- **Particles/second**: 100,000+
- **Steps/second**: 5.0+
- **GPU Memory**: < 8 GB

### Production Target Performance  
- **Particles/second**: 500,000+
- **Steps/second**: 15.0+
- **GPU Memory**: < 16 GB

### High Performance Target
- **Particles/second**: 2,000,000+
- **Steps/second**: 30.0+
- **GPU Memory**: < 32 GB

## Interpreting PDF Reports

The generated PDF reports include:

**1. Executive Summary**
- Key performance findings
- System configuration details
- Overall statistics

**2. Performance Visualizations**
- Throughput scaling with environment count
- GPU utilization vs memory usage scatter plots
- Parameter sensitivity heatmaps
- Performance vs accuracy trade-off curves

**3. Optimization Recommendations**
- Best configurations for different use cases
- Performance bottleneck identification
- Memory usage optimization suggestions

**4. Detailed Results Tables**
- Complete configuration and metrics for all tests
- Easy comparison between different scenarios

## Common Issues and Solutions

### Out of Memory Errors
**Problem**: `CUDA error 2: out of memory`
**Solutions**:
- Increase voxel size (0.05 → 0.1)
- Reduce particles per cell (4 → 3 → 2)
- Reduce particle spawn area
- Use fewer environments

### Poor GPU Utilization
**Problem**: GPU utilization < 50%
**Solutions**:
- Decrease voxel size (more computation)
- Increase particles per cell
- Add more environments
- Reduce solver tolerance (more iterations)

### Slow Performance
**Problem**: Steps/second < 5
**Solutions**:
- Increase voxel size
- Reduce particles per cell
- Relax solver tolerance
- Reduce max iterations
- Use fewer substeps

### Solver Convergence Issues
**Problem**: Physics instability or unrealistic behavior
**Solutions**:
- Decrease time step (increase substeps)
- Tighten solver tolerance
- Increase max iterations
- Check material parameters

## Advanced Usage

### Custom Parameter Studies

Create custom configuration files:

```json
{
  "description": "Custom parameter study",
  "base_config": {
    "voxel_size": 0.03,
    "particles_per_cell": 4,
    "num_frames": 100
  },
  "param_ranges": {
    "voxel_size": [0.02, 0.03, 0.04, 0.05],
    "friction_coeff": [0.5, 0.7, 0.9]
  },
  "env_counts": [1, 2, 4]
}
```

Run with custom config:
```bash
uv run benchmark_mpm_performance.py --custom-params my_config.json --generate-report
```

### Automated Testing

For CI/CD integration:
```bash
# Quick validation test
uv run run_realistic_benchmark.py --quick-test > benchmark_results.log

# Check if performance targets are met
python -c "
import json
with open('realistic_benchmark_results/realistic_benchmark_results.json') as f:
    data = json.load(f)
best_throughput = max(r['metrics']['particles_per_second'] for r in data['results'])
print(f'Best throughput: {best_throughput:,.0f} particles/s')
assert best_throughput > 100000, 'Performance below minimum threshold'
print('Performance test PASSED')
"
```

## File Structure

```
newton/newton/examples/mpm/
├── benchmark_mpm_performance.py          # Main benchmarking system
├── run_realistic_benchmark.py            # Realistic scenario runner
├── benchmark_report_generator.py         # PDF report generation
├── multi_env_mpm_simulation.py          # Multi-environment support
├── benchmark_configs/                    # Configuration files
│   ├── realistic_comprehensive.json     # Realistic parameters
│   ├── quick_test.json                  # Fast validation
│   └── comprehensive.json               # Thorough analysis
├── benchmark_results/                    # Basic benchmark outputs
└── realistic_benchmark_results/          # Realistic benchmark outputs
    ├── realistic_benchmark_results.json # Raw data
    ├── mpm_benchmark_report_*.pdf       # Generated reports
    └── figures/                         # Plot images
```

This comprehensive benchmarking system provides everything needed to systematically optimize Newton MPM simulations for any performance/accuracy requirements.
