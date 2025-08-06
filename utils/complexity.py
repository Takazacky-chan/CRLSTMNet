# utils/complexity.py - Model complexity analysis tool (Complete version)
"""
Model complexity statistics module
Supports multi-dimensional analysis of FLOPs, parameters, memory usage, etc.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table

    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    warnings.warn("fvcore not available, using fallback parameter counting")

try:
    from thop import profile, clever_format

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

try:
    from ptflops import get_model_complexity_info

    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False

try:
    from torchinfo import summary

    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False


class ModelComplexityAnalyzer:
    """Model complexity analyzer"""

    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...], device: str = 'cuda'):
        """
        Args:
            model: Model to analyze
            input_shape: Input shape (excluding batch dimension), e.g. (T, 2, 32, 32)
            device: Computing device
        """
        self.model = model.to(device)
        self.input_shape = input_shape
        self.device = device
        self.model.eval()

        # Create dummy input
        self.dummy_input = torch.randn(1, *input_shape).to(device)

    def get_parameter_stats(self) -> Dict[str, Any]:
        """Get parameter statistics"""
        total_params = 0
        trainable_params = 0
        param_details = {}

        for name, param in self.model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

            # Group statistics by module
            module_name = name.split('.')[0]
            if module_name not in param_details:
                param_details[module_name] = {'total': 0, 'trainable': 0}

            param_details[module_name]['total'] += num_params
            if param.requires_grad:
                param_details[module_name]['trainable'] += num_params

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params,
            'total_params_M': total_params / 1e6,
            'trainable_params_M': trainable_params / 1e6,
            'param_details': param_details
        }

    def get_flops_fvcore(self) -> Dict[str, Any]:
        """Calculate FLOPs using fvcore"""
        if not FVCORE_AVAILABLE:
            return {'error': 'fvcore not available'}

        try:
            flops_analysis = FlopCountAnalysis(self.model, self.dummy_input)
            total_flops = flops_analysis.total()

            # Get detailed FLOPs table
            flop_table = flop_count_table(flops_analysis)

            return {
                'total_flops': total_flops,
                'total_flops_G': total_flops / 1e9,
                'total_flops_M': total_flops / 1e6,
                'flop_table': flop_table,
                'method': 'fvcore'
            }
        except Exception as e:
            return {'error': f'fvcore analysis failed: {str(e)}'}

    def get_flops_thop(self) -> Dict[str, Any]:
        """Calculate FLOPs using thop"""
        if not THOP_AVAILABLE:
            return {'error': 'thop not available'}

        try:
            # Create model copy to avoid modifying original model
            model_copy = type(self.model)(self.model.cfg if hasattr(self.model, 'cfg') else None)
            model_copy.load_state_dict(self.model.state_dict())
            model_copy = model_copy.to(self.device)

            flops, params = profile(model_copy, inputs=(self.dummy_input,), verbose=False)
            flops_str, params_str = clever_format([flops, params], "%.3f")

            return {
                'total_flops': flops,
                'total_flops_G': flops / 1e9,
                'total_flops_M': flops / 1e6,
                'total_params': params,
                'flops_formatted': flops_str,
                'params_formatted': params_str,
                'method': 'thop'
            }
        except Exception as e:
            return {'error': f'thop analysis failed: {str(e)}'}

    def get_flops_ptflops(self) -> Dict[str, Any]:
        """Calculate FLOPs using ptflops"""
        if not PTFLOPS_AVAILABLE:
            return {'error': 'ptflops not available'}

        try:
            # ptflops needs input shape (excluding batch)
            input_shape_no_batch = self.input_shape

            macs, params = get_model_complexity_info(
                self.model,
                input_shape_no_batch,
                print_per_layer_stat=False,
                as_strings=False
            )

            # MACs to FLOPs (usually MACs * 2 = FLOPs)
            flops = macs * 2

            return {
                'total_flops': flops,
                'total_flops_G': flops / 1e9,
                'total_flops_M': flops / 1e6,
                'total_macs': macs,
                'total_macs_G': macs / 1e9,
                'total_params': params,
                'method': 'ptflops'
            }
        except Exception as e:
            return {'error': f'ptflops analysis failed: {str(e)}'}

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available for memory analysis'}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Forward pass
        with torch.no_grad():
            _ = self.model(self.dummy_input)

        # Get memory statistics
        memory_stats = {
            'peak_memory_allocated_MB': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'peak_memory_reserved_MB': torch.cuda.max_memory_reserved() / 1024 ** 2,
            'current_memory_allocated_MB': torch.cuda.memory_allocated() / 1024 ** 2,
            'current_memory_reserved_MB': torch.cuda.memory_reserved() / 1024 ** 2,
        }

        return memory_stats

    def analyze_all(self, print_table: bool = False) -> Dict[str, Any]:
        """Complete complexity analysis"""
        results = {
            'input_shape': self.input_shape,
            'device': self.device,
            'model_name': self.model.__class__.__name__
        }

        # Parameter statistics
        results['parameters'] = self.get_parameter_stats()

        # FLOPs statistics - try multiple methods
        flops_methods = []

        fvcore_result = self.get_flops_fvcore()
        if 'error' not in fvcore_result:
            flops_methods.append('fvcore')
            results['flops_fvcore'] = fvcore_result
            if print_table and 'flop_table' in fvcore_result:
                print("=== FLOPs Breakdown (fvcore) ===")
                print(fvcore_result['flop_table'])

        thop_result = self.get_flops_thop()
        if 'error' not in thop_result:
            flops_methods.append('thop')
            results['flops_thop'] = thop_result

        ptflops_result = self.get_flops_ptflops()
        if 'error' not in ptflops_result:
            flops_methods.append('ptflops')
            results['flops_ptflops'] = ptflops_result

        # Select main FLOPs result
        if 'flops_fvcore' in results:
            results['flops_main'] = results['flops_fvcore']
        elif 'flops_thop' in results:
            results['flops_main'] = results['flops_thop']
        elif 'flops_ptflops' in results:
            results['flops_main'] = results['flops_ptflops']
        else:
            results['flops_main'] = {'error': 'No FLOPs calculation method available'}

        # Memory usage
        results['memory'] = self.get_memory_usage()

        # Efficiency metrics
        if 'error' not in results['flops_main']:
            flops_per_param = results['flops_main']['total_flops'] / results['parameters']['total_params']
            results['efficiency'] = {
                'flops_per_parameter': flops_per_param,
                'parameters_per_gflop': results['parameters']['total_params'] / results['flops_main']['total_flops_G']
            }

        return results

    def print_summary(self, results: Optional[Dict] = None):
        """Print complexity analysis summary"""
        if results is None:
            results = self.analyze_all()

        print(f"\n{'=' * 60}")
        print(f"🔍 Model Complexity Analysis: {results['model_name']}")
        print(f"{'=' * 60}")
        print(f"📋 Input Shape: {results['input_shape']}")
        print(f"🖥️  Device: {results['device']}")

        # Parameter statistics
        params = results['parameters']
        print(f"\n📊 Parameters:")
        print(f"  Total: {params['total_params']:,} ({params['total_params_M']:.2f}M)")
        print(f"  Trainable: {params['trainable_params']:,} ({params['trainable_params_M']:.2f}M)")
        print(f"  Non-trainable: {params['non_trainable_params']:,}")

        # Show parameter distribution by module
        if params['param_details']:
            print(f"\n📦 Parameters by Module:")
            for module, details in params['param_details'].items():
                print(f"  {module}: {details['total']:,} ({details['total'] / 1e6:.2f}M)")

        # FLOPs statistics
        if 'error' not in results['flops_main']:
            flops = results['flops_main']
            print(f"\n⚡ FLOPs ({flops['method']}):")
            print(f"  Total: {flops['total_flops']:,} ({flops['total_flops_G']:.2f}G)")

            if 'total_macs' in flops:
                print(f"  MACs: {flops['total_macs']:,} ({flops['total_macs_G']:.2f}G)")
        else:
            print(f"\n⚡ FLOPs: {results['flops_main']['error']}")

        # Memory usage
        if 'error' not in results['memory']:
            memory = results['memory']
            print(f"\n💾 Memory Usage (Peak):")
            print(f"  Allocated: {memory['peak_memory_allocated_MB']:.1f} MB")
            print(f"  Reserved: {memory['peak_memory_reserved_MB']:.1f} MB")

        # Efficiency metrics
        if 'efficiency' in results:
            eff = results['efficiency']
            print(f"\n📈 Efficiency:")
            print(f"  FLOPs/Parameter: {eff['flops_per_parameter']:.1f}")
            print(f"  Parameters/GFLOPs: {eff['parameters_per_gflop']:.1f}")

        print(f"{'=' * 60}\n")


def analyze_model_complexity(model: nn.Module,
                             input_shape: Tuple[int, ...],
                             device: str = 'cuda',
                             print_summary: bool = True,
                             print_table: bool = False) -> Dict[str, Any]:
    """
    Convenience function: analyze model complexity

    Args:
        model: Model to analyze
        input_shape: Input shape (excluding batch dimension)
        device: Computing device
        print_summary: Whether to print summary
        print_table: Whether to print detailed FLOPs table

    Returns:
        dict: Complete analysis results
    """
    analyzer = ModelComplexityAnalyzer(model, input_shape, device)
    results = analyzer.analyze_all(print_table=print_table)

    if print_summary:
        analyzer.print_summary(results)

    return results


def create_complexity_png(complexity_results, save_path, model_name="CRLSTMNet", compression_ratio="1/32"):
    """
    Create simple and practical complexity analysis PNG chart

    Args:
        complexity_results: Complexity analysis result dictionary
        save_path: Save path
        model_name: Model name
        compression_ratio: Compression ratio
    """
    try:
        # Extract key data
        params = complexity_results.get('parameters', {})
        flops = complexity_results.get('flops_main', {})
        memory = complexity_results.get('memory', {})

        # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{model_name} Complexity Analysis - CR: {compression_ratio}',
                     fontsize=16, fontweight='bold')

        # 1. Parameter pie chart (top left)
        if 'param_details' in params and params['param_details']:
            module_names = []
            param_counts = []

            for module, details in params['param_details'].items():
                module_names.append(module.replace('_', '\n'))  # Line break for display
                param_counts.append(details['total'] / 1e6)  # Convert to M

            colors = plt.cm.Set3(np.linspace(0, 1, len(module_names)))
            wedges, texts, autotexts = axes[0, 0].pie(param_counts, labels=module_names,
                                                      autopct='%1.1f%%', startangle=90,
                                                      colors=colors)
            axes[0, 0].set_title(f'Parameters Distribution\nTotal: {params.get("total_params_M", 0):.2f}M')

            # Beautify text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            axes[0, 0].text(0.5, 0.5, f'Total Parameters\n{params.get("total_params_M", 0):.2f}M',
                            ha='center', va='center', fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[0, 0].set_title('Parameters')

        # 2. Key metrics bar chart (top right)
        metrics = []
        values = []
        colors = []

        if params.get('total_params_M'):
            metrics.append('Params\n(M)')
            values.append(params['total_params_M'])
            colors.append('skyblue')

        if 'error' not in flops and flops.get('total_flops_G'):
            metrics.append('FLOPs\n(G)')
            values.append(flops['total_flops_G'])
            colors.append('lightcoral')

        if 'error' not in memory and memory.get('peak_memory_allocated_MB'):
            metrics.append('Memory\n(MB)')
            values.append(memory['peak_memory_allocated_MB'])
            colors.append('lightgreen')

        if metrics:
            bars = axes[0, 1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Key Metrics')
            axes[0, 1].set_ylabel('Value')

            # Show values on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.01,
                                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'Metrics\nNot Available',
                            ha='center', va='center', fontsize=12)
            axes[0, 1].set_title('Key Metrics')

        # 3. Model efficiency scatter plot (bottom left)
        if 'error' not in flops and params.get('total_params_M') and flops.get('total_flops_G'):
            x = params['total_params_M']
            y = flops['total_flops_G']

            axes[1, 0].scatter(x, y, s=200, c='red', alpha=0.7, edgecolors='black', linewidth=2)
            axes[1, 0].set_xlabel('Parameters (M)')
            axes[1, 0].set_ylabel('FLOPs (G)')
            axes[1, 0].set_title('Model Efficiency')
            axes[1, 0].grid(True, alpha=0.3)

            # Add value annotation
            axes[1, 0].annotate(f'({x:.1f}M, {y:.1f}G)',
                                (x, y), xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

            # Set reasonable coordinate range
            axes[1, 0].set_xlim(0, x * 1.2)
            axes[1, 0].set_ylim(0, y * 1.2)
        else:
            axes[1, 0].text(0.5, 0.5, 'Efficiency Analysis\nInsufficient Data',
                            ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Model Efficiency')

        # 4. Detailed information text (bottom right)
        info_text = f"🏷️ Model: {model_name}\n"
        info_text += f"📊 Compression Ratio: {compression_ratio}\n"
        info_text += f"⚙️ Device: {complexity_results.get('device', 'N/A')}\n\n"

        # Parameter information
        info_text += f"📈 Parameters:\n"
        info_text += f"  Total: {params.get('total_params_M', 0):.2f}M\n"
        info_text += f"  Trainable: {params.get('trainable_params_M', 0):.2f}M\n\n"

        # FLOPs information
        if 'error' not in flops:
            info_text += f"⚡ FLOPs:\n"
            info_text += f"  Total: {flops.get('total_flops_G', 0):.2f}G\n"
            info_text += f"  Method: {flops.get('method', 'N/A')}\n\n"
        else:
            info_text += f"⚡ FLOPs: {flops.get('error', 'N/A')}\n\n"

        # Memory information
        if 'error' not in memory:
            info_text += f"💾 Memory:\n"
            info_text += f"  Peak: {memory.get('peak_memory_allocated_MB', 0):.1f}MB\n"

        # Efficiency metrics
        if 'efficiency' in complexity_results:
            eff = complexity_results['efficiency']
            info_text += f"\n📊 Efficiency:\n"
            info_text += f"  FLOPs/Param: {eff.get('flops_per_parameter', 0):.1f}\n"

        axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, pad=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Information')

        # Adjust layout and save
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Complexity visualization saved to {save_path}")
        plt.close()
        return str(save_path)

    except Exception as e:
        print(f"❌ Failed to save complexity visualization: {e}")
        return None


def analyze_model_complexity_safe(model, cfg, logger, save_dir):
    """Safe model complexity analysis function - Enhanced version"""
    try:
        # Get input shape
        T = cfg.model.sequence.get('T', 32)
        input_shape = (T, 2, 32, 32)

        logger.info("🔍 Analyzing model complexity...")
        complexity_results = analyze_model_complexity(
            model,
            input_shape,
            device=cfg.get('device', 'cuda'),
            print_summary=True,
            print_table=False
        )

        # Save JSON results
        complexity_file = Path(
            save_dir) / f"complexity_analysis_{cfg.data.get('split', 'unknown')}_cr{cfg.data.get('cr', '').replace('/', '_')}.json"

        # Ensure results are JSON serializable
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (torch.Tensor, np.ndarray)):
                if hasattr(obj, 'numel') and obj.numel() < 100:
                    return obj.tolist()
                else:
                    return f"Tensor/Array of shape {obj.shape}"
            elif isinstance(obj, torch.device):
                return str(obj)
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return str(obj)

        serializable_results = make_json_serializable(complexity_results)

        with open(complexity_file, 'w') as f:
            import json
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"💾 Complexity analysis (JSON) saved to {complexity_file}")

        # 🎨 Create PNG chart
        png_file = Path(
            save_dir) / f"complexity_visualization_{cfg.data.get('split', 'unknown')}_cr{cfg.data.get('cr', '').replace('/', '_')}.png"

        create_complexity_png(
            complexity_results,
            png_file,
            model_name=model.__class__.__name__,
            compression_ratio=cfg.data.get('cr', '1/32')
        )

        # Extract key information for summary
        complexity_summary = {}
        if 'parameters' in complexity_results:
            complexity_summary['total_params_M'] = complexity_results['parameters']['total_params_M']
            complexity_summary['trainable_params_M'] = complexity_results['parameters']['trainable_params_M']

        if 'flops_main' in complexity_results and 'error' not in complexity_results['flops_main']:
            complexity_summary['total_flops_G'] = complexity_results['flops_main']['total_flops_G']
            complexity_summary['flops_method'] = complexity_results['flops_main']['method']

        if 'efficiency' in complexity_results:
            complexity_summary['flops_per_param'] = complexity_results['efficiency']['flops_per_parameter']

        if 'memory' in complexity_results and 'error' not in complexity_results['memory']:
            complexity_summary['peak_memory_MB'] = complexity_results['memory']['peak_memory_allocated_MB']

        return complexity_summary

    except ImportError as e:
        logger.warning(f"Cannot perform complexity analysis: {e}")
        logger.info("Install required packages: pip install fvcore thop ptflops")
        return None
    except Exception as e:
        logger.error(f"Complexity analysis failed: {e}")
        return None


def compare_model_complexity(models: Dict[str, nn.Module],
                             input_shape: Tuple[int, ...],
                             device: str = 'cuda') -> Dict[str, Dict]:
    """
    Compare complexity of multiple models

    Args:
        models: {'model_name': model} dictionary
        input_shape: Input shape
        device: Computing device

    Returns:
        dict: Analysis results for each model
    """
    results = {}

    print(f"\n🔄 Comparing {len(models)} models...")

    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        analyzer = ModelComplexityAnalyzer(model, input_shape, device)
        results[name] = analyzer.analyze_all()

    # Print comparison table
    print(f"\n{'=' * 80}")
    print(f"📊 Model Complexity Comparison")
    print(f"{'=' * 80}")
    print(f"{'Model':<20} {'Params(M)':<12} {'FLOPs(G)':<12} {'Memory(MB)':<12}")
    print(f"{'-' * 80}")

    for name, result in results.items():
        params_m = result['parameters']['total_params_M']

        if 'error' not in result['flops_main']:
            flops_g = result['flops_main']['total_flops_G']
        else:
            flops_g = "N/A"

        if 'error' not in result['memory']:
            memory_mb = result['memory']['peak_memory_allocated_MB']
        else:
            memory_mb = "N/A"

        print(f"{name:<20} {params_m:<12.2f} {flops_g:<12} {memory_mb:<12}")

    print(f"{'=' * 80}\n")

    return results