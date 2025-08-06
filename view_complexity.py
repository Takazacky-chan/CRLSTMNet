# view_complexity.py - Complexity analysis JSON file viewer
"""
Tool for viewing and analyzing complexity analysis JSON files
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import sys

# Set matplotlib to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_complexity_json(file_path: str) -> Dict[str, Any]:
    """Load complexity analysis JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None


def print_complexity_summary(data: Dict[str, Any]):
    """Print complexity analysis summary"""
    if not data:
        return

    print("=" * 60)
    print("🔍 COMPLEXITY ANALYSIS SUMMARY")
    print("=" * 60)

    # Basic information
    if 'model_info' in data:
        model_info = data['model_info']
        print(f"Model: {model_info.get('model_name', 'Unknown')}")
        print(f"Config: {Path(model_info.get('config_path', '')).name}")
        print(f"Device: {model_info.get('device', 'Unknown')}")
        print(f"Time: {model_info.get('evaluation_time', 'Unknown')}")

    # Compression information
    if 'compression_info' in data:
        comp_info = data['compression_info']
        print(f"Compression Ratio: {comp_info.get('cr', 'Unknown')}")
        print(f"Measurements: {comp_info.get('cr_num', 'Unknown')}")

    print()

    # Complexity analysis results
    if 'complexity_analysis' in data:
        complexity = data['complexity_analysis']

        # Parameter statistics
        if 'parameters' in complexity:
            params = complexity['parameters']
            print("📊 PARAMETERS:")
            print(f"  Total: {params.get('total_params', 0):,} ({params.get('total_params_M', 0):.2f}M)")
            print(f"  Trainable: {params.get('trainable_params', 0):,} ({params.get('trainable_params_M', 0):.2f}M)")

            # Display by module
            if 'param_details' in params:
                print("  By Module:")
                for module, details in params['param_details'].items():
                    print(f"    {module}: {details['total']:,} ({details['total'] / 1e6:.2f}M)")
            print()

        # FLOPs statistics
        if 'flops_main' in complexity:
            flops = complexity['flops_main']
            print("⚡ FLOPs:")
            if 'error' not in flops:
                print(f"  Total: {flops.get('total_flops', 0):,} ({flops.get('total_flops_G', 0):.2f}G)")
                print(f"  Method: {flops.get('method', 'Unknown')}")
            else:
                print(f"  Error: {flops['error']}")
            print()

        # Memory usage
        if 'memory' in complexity:
            memory = complexity['memory']
            print("💾 MEMORY:")
            if 'error' not in memory:
                print(f"  Peak Allocated: {memory.get('peak_memory_allocated_MB', 0):.1f} MB")
                print(f"  Peak Reserved: {memory.get('peak_memory_reserved_MB', 0):.1f} MB")
            else:
                print(f"  Error: {memory['error']}")
            print()

        # Efficiency metrics
        if 'efficiency' in complexity:
            eff = complexity['efficiency']
            print("📈 EFFICIENCY:")
            print(f"  FLOPs/Parameter: {eff.get('flops_per_parameter', 0):.1f}")
            print(f"  Parameters/GFLOPs: {eff.get('parameters_per_gflop', 0):.1f}")
            print()

    # Performance evaluation results
    if 'performance_evaluation' in data:
        perf = data['performance_evaluation']
        if 'overall' in perf:
            overall = perf['overall']
            print("📈 PERFORMANCE:")
            print(f"  NMSE(dB): {overall.get('nmse_db', 'N/A'):.3f}")
            print(f"  Correlation (ρ): {overall.get('rho', 'N/A'):.4f}")
            print(f"  Total Samples: {overall.get('total_samples', 'N/A')}")
            print()

    print("=" * 60)


def extract_complexity_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key complexity information from JSON data"""
    result = {
        'File': 'Unknown',
        'Model': 'Unknown',
        'CR': 'Unknown',
        'Params_M': 0,
        'FLOPs_G': 'N/A',
        'Memory_MB': 'N/A',
        'NMSE_dB': 'N/A',
        'Correlation': 'N/A'
    }

    # Model information
    if 'model_info' in data:
        result['Model'] = data['model_info'].get('model_name', 'Unknown')
    elif 'model_name' in data:
        result['Model'] = data['model_name']

    # Compression ratio
    if 'compression_info' in data:
        result['CR'] = data['compression_info'].get('cr', 'Unknown')
    elif 'input_shape' in data:
        result['CR'] = 'Unknown'

    # Complexity information
    complexity = data.get('complexity_analysis', data)  # Compatible with different formats

    if 'parameters' in complexity:
        result['Params_M'] = complexity['parameters'].get('total_params_M', 0)

    if 'flops_main' in complexity and 'error' not in complexity['flops_main']:
        result['FLOPs_G'] = complexity['flops_main'].get('total_flops_G', 0)

    if 'memory' in complexity and 'error' not in complexity['memory']:
        result['Memory_MB'] = complexity['memory'].get('peak_memory_allocated_MB', 0)

    # Performance information
    if 'performance_evaluation' in data and 'overall' in data['performance_evaluation']:
        overall = data['performance_evaluation']['overall']
        result['NMSE_dB'] = overall.get('nmse_db', 'N/A')
        result['Correlation'] = overall.get('rho', 'N/A')

    return result


def create_comparison_table(json_files: List[str]) -> pd.DataFrame:
    """Create comparison table for multiple models"""
    comparison_data = []

    for file_path in json_files:
        data = load_complexity_json(file_path)
        if not data:
            continue

        row = extract_complexity_data(data)
        row['File'] = Path(file_path).name
        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def create_visualization(data: Dict[str, Any], save_path: str = None):
    """Create complexity visualization charts"""
    try:
        # Compatible with different JSON formats
        complexity = data.get('complexity_analysis', data)

        # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Get model name and compression ratio
        model_name = data.get('model_info', {}).get('model_name') or data.get('model_name', 'Unknown')
        cr = data.get('compression_info', {}).get('cr') or 'Unknown'

        fig.suptitle(f'{model_name} Complexity Analysis - CR: {cr}', fontsize=16, fontweight='bold')

        # 1. Parameter distribution pie chart
        if 'parameters' in complexity and 'param_details' in complexity['parameters']:
            param_details = complexity['parameters']['param_details']
            modules = list(param_details.keys())
            sizes = [details['total'] / 1e6 for details in param_details.values()]

            if modules and sizes:
                colors = plt.cm.Set3(np.linspace(0, 1, len(modules)))
                wedges, texts, autotexts = axes[0, 0].pie(sizes, labels=modules, autopct='%1.1f%%',
                                                          startangle=90, colors=colors)

                # Beautify text
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

                total_params = complexity['parameters'].get('total_params_M', 0)
                axes[0, 0].set_title(f'Parameter Distribution\nTotal: {total_params:.2f}M')
            else:
                axes[0, 0].text(0.5, 0.5, 'No parameter\ndetails available',
                                ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Parameter Distribution')
        else:
            total_params = complexity.get('parameters', {}).get('total_params_M', 0)
            axes[0, 0].text(0.5, 0.5, f'Total Parameters\n{total_params:.2f}M',
                            ha='center', va='center', fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[0, 0].set_title('Parameters')

        # 2. Key metrics bar chart
        metrics = []
        values = []
        colors = []

        if 'parameters' in complexity:
            metrics.append('Params(M)')
            values.append(complexity['parameters'].get('total_params_M', 0))
            colors.append('skyblue')

        if 'flops_main' in complexity and 'error' not in complexity['flops_main']:
            metrics.append('FLOPs(G)')
            values.append(complexity['flops_main'].get('total_flops_G', 0))
            colors.append('lightcoral')

        if 'memory' in complexity and 'error' not in complexity['memory']:
            metrics.append('Memory(MB)')
            values.append(complexity['memory'].get('peak_memory_allocated_MB', 0))
            colors.append('lightgreen')

        if metrics and values:
            bars = axes[0, 1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Key Metrics')
            axes[0, 1].set_ylabel('Value')

            # Display values on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.01,
                                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'No metrics\navailable',
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Key Metrics')

        # 3. Performance vs complexity (if performance data available)
        if 'performance_evaluation' in data and 'overall' in data['performance_evaluation']:
            perf = data['performance_evaluation']['overall']
            complexity_score = complexity.get('parameters', {}).get('total_params_M', 0)
            performance_score = perf.get('rho', 0)

            if complexity_score > 0 and performance_score > 0:
                axes[1, 0].scatter(complexity_score, performance_score, s=200, c='red', alpha=0.7,
                                   edgecolors='black', linewidth=2)
                axes[1, 0].set_xlabel('Model Complexity (Params M)')
                axes[1, 0].set_ylabel('Performance (Correlation ρ)')
                axes[1, 0].set_title('Performance vs Complexity')
                axes[1, 0].grid(True, alpha=0.3)

                # Add annotation
                axes[1, 0].annotate(f'({complexity_score:.1f}M, {performance_score:.3f})',
                                    (complexity_score, performance_score),
                                    xytext=(10, 10), textcoords='offset points',
                                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

                # Set reasonable coordinate range
                axes[1, 0].set_xlim(0, complexity_score * 1.2)
                axes[1, 0].set_ylim(0, min(1.0, performance_score * 1.1))
            else:
                axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor analysis',
                                ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Performance vs Complexity')
        else:
            axes[1, 0].text(0.5, 0.5, 'No performance\ndata available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Performance vs Complexity')

        # 4. Summary information
        summary_text = f"Model: {model_name}\n"
        summary_text += f"Compression Ratio: {cr}\n\n"

        if 'parameters' in complexity:
            summary_text += f"Parameters: {complexity['parameters'].get('total_params_M', 0):.2f}M\n"

        if 'flops_main' in complexity and 'error' not in complexity['flops_main']:
            summary_text += f"FLOPs: {complexity['flops_main'].get('total_flops_G', 0):.2f}G\n"
            summary_text += f"Method: {complexity['flops_main'].get('method', 'N/A')}\n"

        if 'memory' in complexity and 'error' not in complexity['memory']:
            summary_text += f"Peak Memory: {complexity['memory'].get('peak_memory_allocated_MB', 0):.1f}MB\n"

        if 'performance_evaluation' in data and 'overall' in data['performance_evaluation']:
            perf = data['performance_evaluation']['overall']
            summary_text += f"\nNMSE: {perf.get('nmse_db', 'N/A'):.3f}dB\n"
            summary_text += f"Correlation: {perf.get('rho', 'N/A'):.4f}\n"

        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, pad=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Information')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='View Complexity Analysis JSON Files')
    parser.add_argument('files', nargs='+', help='JSON file(s) to analyze')
    parser.add_argument('--compare', action='store_true',
                        help='Create comparison table for multiple files')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--save-csv', type=str,
                        help='Save comparison table to CSV file')
    parser.add_argument('--save-plot', type=str,
                        help='Save visualization plot to file')

    args = parser.parse_args()

    # Check if files exist
    valid_files = []
    for file_path in args.files:
        if Path(file_path).exists():
            valid_files.append(file_path)
        else:
            print(f"❌ File not found: {file_path}")

    if not valid_files:
        print("❌ No valid files found")
        return

    print(f"📁 Found {len(valid_files)} valid file(s)")

    # Single file detailed analysis
    if len(valid_files) == 1 and not args.compare:
        data = load_complexity_json(valid_files[0])
        if data:
            print_complexity_summary(data)

            if args.visualize:
                create_visualization(data, args.save_plot)

    # Multiple file comparison
    elif args.compare or len(valid_files) > 1:
        print("\n📊 Creating comparison table...")
        df = create_comparison_table(valid_files)

        if not df.empty:
            print("\n" + "=" * 80)
            print("📋 COMPARISON TABLE")
            print("=" * 80)
            print(df.to_string(index=False))
            print("=" * 80)

            if args.save_csv:
                df.to_csv(args.save_csv, index=False)
                print(f"💾 Comparison table saved to {args.save_csv}")
        else:
            print("❌ No valid data found for comparison")

        # Create visualization for the first file
        if args.visualize and valid_files:
            data = load_complexity_json(valid_files[0])
            if data:
                create_visualization(data, args.save_plot)


if __name__ == "__main__":
    # If no command line arguments, show usage instructions
    if len(sys.argv) == 1:
        print("""
🔍 Complexity Analysis JSON Viewer

Usage Examples:
  # View single file
  python view_complexity.py complexity_analysis.json

  # Compare multiple files
  python view_complexity.py file1.json file2.json --compare

  # Create visualization charts
  python view_complexity.py complexity_analysis.json --visualize

  # Save comparison table and charts
  python view_complexity.py *.json --compare --save-csv comparison.csv --visualize --save-plot plot.png

  # View your file (use quotes for Windows paths)
  python view_complexity.py "F:/CRLSTMNet/checkpoints/indoor_cr32_enhanced/evaluation/complexity_analysis_20250801_170053.json"

  # View your file with visualization
  python view_complexity.py "F:/CRLSTMNet/checkpoints/indoor_cr32_enhanced/evaluation/complexity_analysis_20250801_170053.json" --visualize
        """)
        sys.exit(0)

    main()