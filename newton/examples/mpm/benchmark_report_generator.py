#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Newton MPM Benchmark Report Generator

Generates comprehensive PDF reports from benchmark results with performance
visualizations, parameter sensitivity analysis, and optimization recommendations.

Dependencies:
    pip install matplotlib reportlab pandas seaborn
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Report generation dependencies not available: {e}")
    print("Install with: pip install matplotlib reportlab pandas seaborn")
    DEPENDENCIES_AVAILABLE = False


class BenchmarkReportGenerator:
    """Generate comprehensive PDF reports from benchmark results."""
    
    def __init__(self, results_file: Path, output_dir: Path):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required dependencies not available for report generation")
        
        self.results_file = results_file
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load results
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.system_info = self.data['system_info']
        self.results = self.data['results']
        self.timestamp = datetime.fromtimestamp(self.data['timestamp'])
        
        # Convert to DataFrame for easier analysis
        self.df = self._create_dataframe()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        rows = []
        for result in self.results:
            row = {}
            # Flatten config and metrics
            for key, value in result['config'].items():
                row[f'config_{key}'] = value
            for key, value in result['metrics'].items():
                row[f'metrics_{key}'] = value
            
            # Add derived metrics
            row['efficiency_score'] = (
                result['metrics']['particles_per_second'] * 
                result['metrics']['avg_gpu_utilization'] / 100
            )
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_performance_plots(self) -> List[str]:
        """Generate performance visualization plots."""
        plot_files = []
        
        # 1. Throughput vs Environment Count
        if 'config_num_envs' in self.df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            env_groups = self.df.groupby('config_num_envs')
            env_counts = sorted(env_groups.groups.keys())
            throughputs = [env_groups.get_group(env)['metrics_particles_per_second'].mean() 
                          for env in env_counts]
            
            ax.plot(env_counts, throughputs, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Environments')
            ax.set_ylabel('Particles per Second')
            ax.set_title('Throughput Scaling with Environment Count')
            ax.grid(True, alpha=0.3)
            
            # Add ideal scaling line
            if len(env_counts) > 1:
                ideal_scaling = [throughputs[0] * env / env_counts[0] for env in env_counts]
                ax.plot(env_counts, ideal_scaling, '--', alpha=0.7, label='Ideal Linear Scaling')
                ax.legend()
            
            plot_file = self.figures_dir / "throughput_scaling.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
        
        # 2. GPU Utilization vs Memory Usage
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            self.df['metrics_avg_gpu_memory_gb'],
            self.df['metrics_avg_gpu_utilization'],
            c=self.df['metrics_particles_per_second'],
            s=60,
            alpha=0.7,
            cmap='viridis'
        )
        
        ax.set_xlabel('GPU Memory Usage (GB)')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Utilization vs Memory Usage')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Particles per Second')
        
        plot_file = self.figures_dir / "gpu_utilization_memory.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        # 3. Parameter Sensitivity Heatmap
        if len(self.df) > 5:  # Only if we have enough data points
            # Select numeric config parameters
            config_cols = [col for col in self.df.columns if col.startswith('config_') and 
                          self.df[col].dtype in ['int64', 'float64']]
            
            if len(config_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Calculate correlation with performance metrics
                perf_cols = ['metrics_particles_per_second', 'metrics_avg_gpu_utilization']
                corr_data = []
                
                for config_col in config_cols:
                    row = []
                    for perf_col in perf_cols:
                        corr = self.df[config_col].corr(self.df[perf_col])
                        row.append(corr if not np.isnan(corr) else 0)
                    corr_data.append(row)
                
                corr_matrix = np.array(corr_data)
                
                im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                
                # Set ticks and labels
                ax.set_xticks(range(len(perf_cols)))
                ax.set_xticklabels([col.replace('metrics_', '') for col in perf_cols])
                ax.set_yticks(range(len(config_cols)))
                ax.set_yticklabels([col.replace('config_', '') for col in config_cols])
                
                # Add correlation values
                for i in range(len(config_cols)):
                    for j in range(len(perf_cols)):
                        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black")
                
                ax.set_title('Parameter Sensitivity Analysis')
                plt.colorbar(im, label='Correlation Coefficient')
                
                plot_file = self.figures_dir / "parameter_sensitivity.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(str(plot_file))
        
        # 4. Performance vs Accuracy Trade-off
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use voxel size as proxy for accuracy (smaller = more accurate)
        if 'config_voxel_size' in self.df.columns:
            scatter = ax.scatter(
                1 / self.df['config_voxel_size'],  # Inverse for "accuracy"
                self.df['metrics_particles_per_second'],
                c=self.df['metrics_avg_gpu_utilization'],
                s=60,
                alpha=0.7,
                cmap='plasma'
            )
            
            ax.set_xlabel('Simulation Accuracy (1/voxel_size)')
            ax.set_ylabel('Particles per Second')
            ax.set_title('Performance vs Accuracy Trade-off')
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('GPU Utilization (%)')
            
            plot_file = self.figures_dir / "performance_accuracy_tradeoff.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
        
        return plot_files
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on results."""
        recommendations = []
        
        # Find best configurations
        best_throughput = self.df.loc[self.df['metrics_particles_per_second'].idxmax()]
        best_efficiency = self.df.loc[self.df['efficiency_score'].idxmax()]
        
        recommendations.append(
            f"Best Throughput Configuration: "
            f"voxel_size={best_throughput.get('config_voxel_size', 'N/A')}, "
            f"particles_per_cell={best_throughput.get('config_particles_per_cell', 'N/A')}, "
            f"achieving {best_throughput['metrics_particles_per_second']:,.0f} particles/second"
        )
        
        recommendations.append(
            f"Best Efficiency Configuration: "
            f"voxel_size={best_efficiency.get('config_voxel_size', 'N/A')}, "
            f"particles_per_cell={best_efficiency.get('config_particles_per_cell', 'N/A')}, "
            f"efficiency score: {best_efficiency['efficiency_score']:,.0f}"
        )
        
        # GPU utilization analysis
        avg_gpu_util = self.df['metrics_avg_gpu_utilization'].mean()
        if avg_gpu_util < 50:
            recommendations.append(
                "Low GPU utilization detected. Consider increasing particle count, "
                "reducing voxel size, or adding more environments to better utilize GPU resources."
            )
        elif avg_gpu_util > 95:
            recommendations.append(
                "Very high GPU utilization detected. Consider reducing particle count "
                "or increasing voxel size to prevent GPU saturation."
            )
        
        # Memory usage analysis
        max_memory = self.df['metrics_max_gpu_memory_gb'].max()
        if max_memory > 8:  # Assuming typical GPU memory limits
            recommendations.append(
                f"High GPU memory usage detected ({max_memory:.1f} GB). "
                "Consider reducing particle count or using larger voxel sizes for memory-constrained systems."
            )
        
        # Scaling efficiency analysis
        if 'config_num_envs' in self.df.columns and len(self.df['config_num_envs'].unique()) > 1:
            env_groups = self.df.groupby('config_num_envs')
            if len(env_groups) >= 2:
                single_env_perf = env_groups.get_group(1)['metrics_particles_per_second'].mean()
                multi_env_groups = [group for name, group in env_groups if name > 1]
                
                if multi_env_groups:
                    best_multi_env = max(multi_env_groups, 
                                       key=lambda g: g['metrics_particles_per_second'].mean())
                    best_multi_perf = best_multi_env['metrics_particles_per_second'].mean()
                    scaling_factor = best_multi_perf / single_env_perf
                    
                    if scaling_factor < 0.8:
                        recommendations.append(
                            "Poor multi-environment scaling detected. "
                            "Single environment may be more efficient for this configuration."
                        )
                    elif scaling_factor > 1.5:
                        recommendations.append(
                            "Excellent multi-environment scaling detected. "
                            "Consider using more environments for better throughput."
                        )
        
        return recommendations

    def generate_pdf_report(self) -> Path:
        """Generate comprehensive PDF report."""
        report_file = self.output_dir / f"mpm_benchmark_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"

        # Generate plots
        plot_files = self.generate_performance_plots()
        recommendations = self.generate_optimization_recommendations()

        # Create PDF document
        doc = SimpleDocTemplate(str(report_file), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Newton MPM Performance Benchmark Report", title_style))
        story.append(Spacer(1, 20))

        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))

        summary_text = f"""
        This report presents the performance analysis of Newton's Material Point Method (MPM)
        particle simulation system. The benchmark was conducted on {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        testing {len(self.results)} different configurations.

        <b>Key Findings:</b><br/>
        • Best throughput: {self.df['metrics_particles_per_second'].max():,.0f} particles/second<br/>
        • Average GPU utilization: {self.df['metrics_avg_gpu_utilization'].mean():.1f}%<br/>
        • Memory usage range: {self.df['metrics_avg_gpu_memory_gb'].min():.1f} - {self.df['metrics_max_gpu_memory_gb'].max():.1f} GB<br/>
        • Configurations tested: {len(self.results)}
        """

        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # System Information
        story.append(Paragraph("System Configuration", styles['Heading2']))

        system_text = f"""
        <b>Hardware:</b><br/>
        • CPU: {self.system_info['cpu']['model']} ({self.system_info['cpu']['cores']} cores)<br/>
        • Memory: {self.system_info['memory']['total_gb']:.1f} GB<br/>
        • GPU: {self.system_info['gpu']['device_name']}<br/>

        <b>Software:</b><br/>
        • Warp Version: {self.system_info.get('warp_version', 'Unknown')}<br/>
        • Newton Version: {self.system_info.get('newton_version', 'Unknown')}<br/>
        • GPU Monitoring: {'Available' if self.system_info['gpu']['available'] else 'Unavailable'}
        """

        story.append(Paragraph(system_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # Performance Analysis
        story.append(Paragraph("Performance Analysis", styles['Heading2']))

        # Add plots
        for plot_file in plot_files:
            if Path(plot_file).exists():
                story.append(Image(plot_file, width=6*inch, height=3.6*inch))
                story.append(Spacer(1, 10))

        # Optimization Recommendations
        story.append(Paragraph("Optimization Recommendations", styles['Heading2']))

        for i, recommendation in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {recommendation}", styles['Normal']))
            story.append(Spacer(1, 10))

        # Detailed Results Table
        story.append(Paragraph("Detailed Results", styles['Heading2']))

        # Create summary table
        table_data = [['Configuration', 'Particles/sec', 'GPU Util %', 'Memory GB']]

        for _, row in self.df.iterrows():
            config_str = f"voxel={row.get('config_voxel_size', 'N/A')}, envs={row.get('config_num_envs', 'N/A')}"
            table_data.append([
                config_str,
                f"{row['metrics_particles_per_second']:,.0f}",
                f"{row['metrics_avg_gpu_utilization']:.1f}",
                f"{row['metrics_avg_gpu_memory_gb']:.2f}"
            ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)

        # Build PDF
        doc.build(story)

        return report_file


def generate_pdf_report(results_file: Path, output_dir: Path) -> Path:
    """Main function to generate PDF report from benchmark results."""
    if not DEPENDENCIES_AVAILABLE:
        raise ImportError("Required dependencies not available for report generation")

    generator = BenchmarkReportGenerator(results_file, output_dir)
    return generator.generate_pdf_report()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate PDF report from benchmark results")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--output", default=".", help="Output directory for report")

    args = parser.parse_args()

    results_file = Path(args.results_file)
    output_dir = Path(args.output)

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        exit(1)

    try:
        report_file = generate_pdf_report(results_file, output_dir)
        print(f"Report generated: {report_file}")
    except Exception as e:
        print(f"Failed to generate report: {e}")
        exit(1)
