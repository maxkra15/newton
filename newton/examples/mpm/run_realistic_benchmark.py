#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Realistic Comprehensive MPM Benchmark Runner

This script runs a comprehensive benchmark using realistic parameter values
for production MPM simulations. It tests multiple scenarios including:
- Different accuracy levels (research, production, real-time, demo)
- Various material types (clay, sand, soil, gravel)
- Performance scaling analysis
- Parameter sensitivity for production use cases

Usage:
    python run_realistic_benchmark.py --scenario all
    python run_realistic_benchmark.py --scenario accuracy_levels
    python run_realistic_benchmark.py --scenario materials
    python run_realistic_benchmark.py --scenario scaling
    python run_realistic_benchmark.py --quick-test
"""

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

from benchmark_mpm_performance import MPMBenchmarkRunner, SimulationConfig
from benchmark_report_generator import generate_pdf_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealisticBenchmarkRunner:
    """Runner for realistic comprehensive MPM benchmarks."""
    
    def __init__(self, output_dir: str = "realistic_benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.runner = MPMBenchmarkRunner(str(self.output_dir))
        
        # Load realistic configuration
        config_file = Path(__file__).parent / "benchmark_configs" / "realistic_comprehensive.json"
        with open(config_file, 'r') as f:
            self.config_data = json.load(f)
        
        self.base_config = SimulationConfig(**self.config_data['base_config'])
        self.scenarios = self.config_data.get('realistic_scenarios', {})
        self.materials = self.config_data.get('material_scenarios', {})
        self.performance_targets = self.config_data.get('performance_targets', {})
    
    def run_accuracy_levels_benchmark(self) -> List[Dict[str, Any]]:
        """Test different accuracy levels for various use cases."""
        logger.info("Running accuracy levels benchmark")
        
        results = []
        
        for scenario_name, scenario_config in self.scenarios.items():
            logger.info(f"Testing {scenario_name}: {scenario_config['description']}")
            
            # Create configuration for this scenario
            config = SimulationConfig(**self.config_data['base_config'])
            
            # Override with scenario-specific parameters
            for key, value in scenario_config.items():
                if key != 'description' and hasattr(config, key):
                    setattr(config, key, value)
            
            # Run benchmark
            try:
                metrics = self.runner.run_single_benchmark(config)
                
                result = {
                    'scenario': scenario_name,
                    'description': scenario_config['description'],
                    'config': config.__dict__,
                    'metrics': metrics.__dict__,
                    'meets_targets': self._evaluate_performance_targets(metrics)
                }
                
                results.append(result)
                self.runner.results.append({
                    'config': config.__dict__,
                    'metrics': metrics.__dict__,
                    'scenario_type': 'accuracy_level',
                    'scenario_name': scenario_name
                })
                
                logger.info(f"  {scenario_name}: {metrics.particles_per_second:,.0f} particles/s, "
                          f"{metrics.steps_per_second:.1f} steps/s")
                
            except Exception as e:
                logger.error(f"Failed benchmark for {scenario_name}: {e}")
                continue
        
        return results
    
    def run_materials_benchmark(self) -> List[Dict[str, Any]]:
        """Test different material properties."""
        logger.info("Running materials benchmark")
        
        results = []
        
        for material_name, material_config in self.materials.items():
            logger.info(f"Testing {material_name}: {material_config['description']}")
            
            # Create configuration for this material
            config = SimulationConfig(**self.config_data['base_config'])
            
            # Override with material-specific parameters
            for key, value in material_config.items():
                if key != 'description' and hasattr(config, key):
                    setattr(config, key, value)
            
            # Run benchmark
            try:
                metrics = self.runner.run_single_benchmark(config)
                
                result = {
                    'material': material_name,
                    'description': material_config['description'],
                    'config': config.__dict__,
                    'metrics': metrics.__dict__,
                    'meets_targets': self._evaluate_performance_targets(metrics)
                }
                
                results.append(result)
                self.runner.results.append({
                    'config': config.__dict__,
                    'metrics': metrics.__dict__,
                    'scenario_type': 'material',
                    'scenario_name': material_name
                })
                
                logger.info(f"  {material_name}: {metrics.particles_per_second:,.0f} particles/s, "
                          f"{metrics.steps_per_second:.1f} steps/s")
                
            except Exception as e:
                logger.error(f"Failed benchmark for {material_name}: {e}")
                continue
        
        return results
    
    def run_parameter_sensitivity_benchmark(self) -> List[Dict[str, Any]]:
        """Run parameter sensitivity analysis with realistic ranges."""
        logger.info("Running parameter sensitivity benchmark")
        
        # Focus on most important parameters for production use
        key_params = {
            'voxel_size': [0.02, 0.025, 0.03, 0.04, 0.05],
            'particles_per_cell': [3, 4, 5],
            'tolerance': [1e-5, 5e-5, 1e-4],
            'max_iterations': [100, 150, 200],
            'substeps': [1, 2, 3]
        }
        
        return self.runner.run_parameter_sweep(key_params, self.base_config)
    
    def run_scaling_benchmark(self) -> List[Dict[str, Any]]:
        """Run multi-environment scaling analysis."""
        logger.info("Running scaling benchmark")
        
        env_counts = self.config_data.get('env_counts', [1, 2, 4, 8])
        return self.runner.run_multi_environment_scaling(self.base_config, env_counts)
    
    def _evaluate_performance_targets(self, metrics) -> Dict[str, bool]:
        """Evaluate if metrics meet performance targets."""
        evaluation = {}
        
        for target_name, targets in self.performance_targets.items():
            meets_target = True
            
            if 'particles_per_second' in targets:
                meets_target &= metrics.particles_per_second >= targets['particles_per_second']
            
            if 'steps_per_second' in targets:
                meets_target &= metrics.steps_per_second >= targets['steps_per_second']
            
            if 'max_memory_gb' in targets:
                meets_target &= metrics.max_gpu_memory_gb <= targets['max_memory_gb']
            
            evaluation[target_name] = meets_target
        
        return evaluation
    
    def generate_comprehensive_report(self) -> Path:
        """Generate a comprehensive PDF report with all results."""
        logger.info("Generating comprehensive PDF report")
        
        # Save results
        results_file = self.runner.save_results("realistic_benchmark_results.json")
        
        # Generate PDF report
        try:
            report_file = generate_pdf_report(results_file, self.output_dir)
            logger.info(f"Comprehensive report generated: {report_file}")
            return report_file
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise
    
    def print_summary(self):
        """Print a comprehensive summary of all results."""
        if not self.runner.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*100)
        print("REALISTIC MPM BENCHMARK COMPREHENSIVE SUMMARY")
        print("="*100)
        
        # Group results by scenario type
        scenario_groups = {}
        for result in self.runner.results:
            scenario_type = result.get('scenario_type', 'parameter_sweep')
            if scenario_type not in scenario_groups:
                scenario_groups[scenario_type] = []
            scenario_groups[scenario_type].append(result)
        
        # Print results by category
        for scenario_type, results in scenario_groups.items():
            print(f"\n{scenario_type.upper().replace('_', ' ')} RESULTS:")
            print("-" * 60)
            
            for result in results:
                scenario_name = result.get('scenario_name', 'Unknown')
                metrics = result['metrics']
                
                print(f"  {scenario_name}:")
                print(f"    Particles/second: {metrics['particles_per_second']:,.0f}")
                print(f"    Steps/second: {metrics['steps_per_second']:.1f}")
                print(f"    GPU memory: {metrics['avg_gpu_memory_gb']:.2f} GB")
                print(f"    Particle count: {metrics['particle_count']:,}")
                
                # Show performance target evaluation if available
                if 'meets_targets' in result:
                    targets_met = [name for name, met in result['meets_targets'].items() if met]
                    if targets_met:
                        print(f"    Meets targets: {', '.join(targets_met)}")
                print()
        
        # Overall statistics
        all_throughputs = [r['metrics']['particles_per_second'] for r in self.runner.results]
        all_step_rates = [r['metrics']['steps_per_second'] for r in self.runner.results]
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Configurations tested: {len(self.runner.results)}")
        print(f"  Throughput range: {min(all_throughputs):,.0f} - {max(all_throughputs):,.0f} particles/s")
        print(f"  Step rate range: {min(all_step_rates):.1f} - {max(all_step_rates):.1f} steps/s")
        print(f"  Best throughput: {max(all_throughputs):,.0f} particles/s")
        print(f"  Best step rate: {max(all_step_rates):.1f} steps/s")
        
        print("="*100 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Realistic Comprehensive MPM Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--scenario',
        choices=['all', 'accuracy_levels', 'materials', 'sensitivity', 'scaling'],
        default='all',
        help='Which benchmark scenario to run'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run a quick test with reduced parameters'
    )
    
    parser.add_argument(
        '--output',
        default='realistic_benchmark_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        default=True,
        help='Generate PDF report (default: True)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Realistic Comprehensive MPM Benchmark")
    
    # Initialize benchmark runner
    benchmark = RealisticBenchmarkRunner(args.output)
    
    try:
        # Run selected scenarios
        if args.quick_test:
            logger.info("Running quick test")
            # Just test one scenario from each category
            benchmark.run_accuracy_levels_benchmark()
        elif args.scenario == 'all':
            logger.info("Running all benchmark scenarios")
            benchmark.run_accuracy_levels_benchmark()
            benchmark.run_materials_benchmark()
            benchmark.run_parameter_sensitivity_benchmark()
            benchmark.run_scaling_benchmark()
        elif args.scenario == 'accuracy_levels':
            benchmark.run_accuracy_levels_benchmark()
        elif args.scenario == 'materials':
            benchmark.run_materials_benchmark()
        elif args.scenario == 'sensitivity':
            benchmark.run_parameter_sensitivity_benchmark()
        elif args.scenario == 'scaling':
            benchmark.run_scaling_benchmark()
        
        # Generate report
        if args.generate_report:
            benchmark.generate_comprehensive_report()
        
        # Print summary
        benchmark.print_summary()
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
    
    logger.info("Realistic comprehensive benchmark completed")


if __name__ == "__main__":
    main()
