# evaluate_model_enhanced.py - Enhanced evaluation script with complexity analysis and visualization
"""
CR-LSTM-Net Model Evaluation Script (Enhanced Version)
Integrated with complexity analysis, visualization comparison and other features

Usage:
- Comprehensive evaluation and save: python evaluate_model_enhanced.py --model "model path/to/model.pth" --config configs/model.yaml --create_visualizations --comprehensive
- Comprehensive evaluation: python evaluate_model_enhanced.py
- Specify model: python evaluate_model_enhanced.py --model path/to/model.pth
- Analyze complexity: python evaluate_model_enhanced.py --analyze_complexity
- Generate visualizations: python evaluate_model_enhanced.py --create_visualizations
- Save prediction results: python evaluate_model_enhanced.py --save_predictions
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
import json
from datetime import datetime
from tqdm.auto import tqdm
import warnings
import sys
import os

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

# Set matplotlib to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EnhancedModelEvaluator:
    """Enhanced Model Evaluator - Integrated with complexity analysis and visualization"""

    def __init__(self, model_path, config_path, device='cuda', split='val'):
        self.model_path = Path(model_path)
        self.config_path = config_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.split = split

        # Load configuration
        self.cfg = OmegaConf.load(config_path)

        # Load model
        self.model, self.checkpoint_info = self._load_model()

        # Verify compression rate consistency
        self._verify_compression_settings()

        # Create evaluation results directory
        self.eval_dir = self.model_path.parent / "evaluation"
        self.eval_dir.mkdir(exist_ok=True)

        # Create visualization directory
        self.viz_dir = self.eval_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)

        # Print key experimental configuration
        self._print_experiment_info()

    def _load_model(self):
        """Load model (Enhanced version)"""
        print(f"Loading model from {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Check checkpoint format
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            checkpoint_info = {
                'compression_info': checkpoint.get('compression_info', {}),
                'config': checkpoint.get('config', {}),
                'timestamp': checkpoint.get('timestamp', 'Unknown')
            }
            print("✅ Loaded full checkpoint with metadata")
        else:
            state_dict = checkpoint
            checkpoint_info = {}
            print("✅ Loaded state dict only")

        # Create model
        from models.crlstmnet import CRLSTMNet
        model = CRLSTMNet(self.cfg)

        # Load weights and check mismatched keys
        load_result = model.load_state_dict(state_dict, strict=False)

        if hasattr(load_result, 'missing_keys') and load_result.missing_keys:
            print(
                f"⚠️ Missing keys: {load_result.missing_keys[:5]} {'...' if len(load_result.missing_keys) > 5 else ''}")
        if hasattr(load_result, 'unexpected_keys') and load_result.unexpected_keys:
            print(
                f"⚠️ Unexpected keys: {load_result.unexpected_keys[:5]} {'...' if len(load_result.unexpected_keys) > 5 else ''}")

        model.to(self.device)
        model.eval()

        # Print model information
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        return model, checkpoint_info

    def _verify_compression_settings(self):
        """Verify compression rate setting consistency"""
        cfg_cr = self.cfg.data.get('cr', 'Unknown')
        cfg_M = self.cfg.data.get('cr_num', None)
        cfg_N = 32 * 32  # Fixed to 1024

        # Get compression information from checkpoint
        ckpt_info = self.checkpoint_info.get('compression_info', {})
        ckpt_cr = ckpt_info.get('cr', None)
        ckpt_M = ckpt_info.get('cr_num', None)

        print(f"📊 Compression Settings Verification:")
        print(f"  Config CR: {cfg_cr} (M={cfg_M}, N={cfg_N})")

        if ckpt_cr is not None:
            print(f"  Checkpoint CR: {ckpt_cr} (M={ckpt_M})")
            if str(cfg_cr) != str(ckpt_cr):
                print(f"⚠️ CR mismatch between config and checkpoint!")

        # Calculate actual compression rate
        actual_cr_rate = cfg_M / cfg_N if cfg_M is not None else None
        if actual_cr_rate:
            print(f"  Actual CR rate: {actual_cr_rate:.6f} ({cfg_M}/{cfg_N})")

        self.compression_info = {
            'cr': cfg_cr,
            'cr_num': cfg_M,
            'N': cfg_N,
            'actual_rate': actual_cr_rate
        }

    def _print_experiment_info(self):
        """Print key experimental factors"""
        print(f"\n🔬 Experiment Configuration:")
        print(f"  Model: {self.model_path.name}")
        print(f"  Device: {self.device}")
        print(f"  Split: {self.split}")
        print(f"  Sequence Length (T): {self.cfg.model.sequence.get('T', 'Unknown')}")
        print(f"  Normalization: {self.cfg.data.get('normalize', 'Unknown')}")
        print(f"  Dataset Type: {self.cfg.data.get('split', 'Unknown')}")
        print(f"  Batch Size: {self.cfg.training.get('batch_size', 'Unknown')}")
        print(f"  Evaluation Directory: {self.eval_dir}")

    def analyze_model_complexity(self):
        """Analyze model complexity"""
        print(f"\n🔍 Analyzing Model Complexity...")

        try:
            from utils.complexity import analyze_model_complexity

            # Get input shape
            T = self.cfg.model.sequence.get('T', 32)
            input_shape = (T, 2, 32, 32)

            # Analyze complexity
            complexity_results = analyze_model_complexity(
                self.model,
                input_shape,
                device=str(self.device),
                print_summary=True,
                print_table=True
            )

            # Save complexity analysis results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            complexity_file = self.eval_dir / f"complexity_analysis_{timestamp}.json"

            # Ensure results are JSON serializable
            serializable_results = self._make_json_serializable(complexity_results)

            with open(complexity_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            print(f"💾 Complexity analysis saved to {complexity_file}")

            return complexity_results

        except ImportError as e:
            print(f"⚠️ Cannot import complexity analyzer: {e}")
            print("Please install required packages: pip install fvcore thop ptflops")
            return None
        except Exception as e:
            print(f"❌ Complexity analysis failed: {e}")
            return None

    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if obj.numel() < 100 else f"Tensor/Array of shape {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    def evaluate_on_dataset(self, save_predictions=False, create_visualizations=False):
        """Evaluate model on dataset (Enhanced version)"""
        print(f"\n🔍 Evaluating on {self.split} dataset...")

        # Create data loader
        try:
            from datasets.cost2100 import create_dataloaders
            train_loader, val_loader = create_dataloaders(self.cfg)

            # Select corresponding loader based on split
            if self.split == 'train':
                eval_loader = train_loader
            else:  # val or other
                eval_loader = val_loader

            print(f"✅ Created data loaders: {len(eval_loader)} {self.split} batches")
        except Exception as e:
            print(f"❌ Failed to create real data loaders: {e}")
            # Check if dummy data is allowed
            if not self.cfg.get('allow_dummy_data', False):
                print("⚠️ Falling back to dummy data")
                from losses.metrics import create_dummy_data
                _, eval_loader = create_dummy_data(self.cfg)
            else:
                raise RuntimeError("Data loading failed and dummy data not allowed")

        # Evaluation metrics collection
        all_metrics = []
        prediction_samples = []
        target_samples = []

        # For more accurate NMSE calculation
        total_mse = 0.0
        total_power = 0.0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(eval_loader, desc=f"Evaluating on {self.split}")
            for batch_idx, H in enumerate(pbar):
                H = H.to(self.device, non_blocking=True)
                batch_size = H.size(0)

                # Forward propagation
                H_hat = self.model(H)

                # Calculate batch-level metrics
                from losses.metrics import compute_metrics
                metrics = compute_metrics(H_hat, H)
                metrics['batch_size'] = batch_size  # Record batch size for weighting
                all_metrics.append(metrics)

                # Accumulate linear domain MSE and power (for more accurate overall NMSE)
                H_flat = H.flatten()
                H_hat_flat = H_hat.flatten()
                batch_mse = torch.mean((H_hat_flat - H_flat) ** 2).item()
                batch_power = torch.mean(H_flat ** 2).item()

                total_mse += batch_mse * batch_size
                total_power += batch_power * batch_size
                total_samples += batch_size

                # Save samples for visualization (first few batches)
                if (save_predictions or create_visualizations) and batch_idx < 5:
                    prediction_samples.append(H_hat.cpu())
                    target_samples.append(H.cpu())

                # Update progress bar
                pbar.set_postfix({
                    'NMSE(dB)': f"{metrics['nmse_db']:.2f}",
                    'ρ': f"{metrics['rho']:.4f}"
                })

        # Calculate overall statistics
        overall_stats = self._compute_overall_stats(all_metrics, total_mse, total_power, total_samples)

        # Save evaluation results
        self._save_evaluation_results(overall_stats, all_metrics)

        # Plot evaluation results
        self._plot_evaluation_results(all_metrics, overall_stats)

        # Create CSI signal comparison visualizations
        if create_visualizations and prediction_samples:
            self._create_csi_visualizations(prediction_samples, target_samples)

        # Save prediction samples
        if save_predictions and prediction_samples:
            self._save_prediction_samples(prediction_samples, target_samples)

        return overall_stats

    def _compute_overall_stats(self, all_metrics, total_mse, total_power, total_samples):
        """Calculate overall statistics (Enhanced version)"""
        # Extract various metrics
        nmse_db_values = [m['nmse_db'] for m in all_metrics]
        rho_values = [m['rho'] for m in all_metrics]
        mse_values = [m['mse'] for m in all_metrics]
        batch_sizes = [m.get('batch_size', 1) for m in all_metrics]

        # More accurate overall NMSE calculation (linear domain weighted average)
        overall_nmse_linear = total_mse / (total_power + 1e-12)
        overall_nmse_db = 10 * np.log10(overall_nmse_linear + 1e-12)

        # Weighted average ρ (by batch size)
        weighted_rho = np.average(rho_values, weights=batch_sizes)

        stats = {
            # More accurate overall metrics
            'overall': {
                'nmse_db': overall_nmse_db,
                'rho': weighted_rho,
                'total_samples': total_samples
            },
            # Distribution statistics
            'nmse_db_dist': {
                'mean': np.mean(nmse_db_values),
                'std': np.std(nmse_db_values),
                'median': np.median(nmse_db_values),
                'min': np.min(nmse_db_values),
                'max': np.max(nmse_db_values),
                'percentile_25': np.percentile(nmse_db_values, 25),
                'percentile_75': np.percentile(nmse_db_values, 75)
            },
            'rho_dist': {
                'mean': np.mean(rho_values),
                'std': np.std(rho_values),
                'median': np.median(rho_values),
                'min': np.min(rho_values),
                'max': np.max(rho_values),
                'percentile_25': np.percentile(rho_values, 25),
                'percentile_75': np.percentile(rho_values, 75)
            },
            'mse_dist': {
                'mean': np.mean(mse_values),
                'std': np.std(mse_values),
                'median': np.median(mse_values),
                'min': np.min(mse_values),
                'max': np.max(mse_values)
            },
            'experiment_info': {
                'compression_ratio': self.compression_info['cr'],
                'cr_num': self.compression_info['cr_num'],
                'dataset_type': self.cfg.data.get('split', 'Unknown'),
                'sequence_length': self.cfg.model.sequence.get('T', 'Unknown'),
                'normalization': self.cfg.data.get('normalize', 'Unknown'),
                'split': self.split
            }
        }

        # Print results
        print(f"\n📈 Evaluation Results:")
        print(f"  📊 Overall NMSE(dB): {overall_nmse_db:.3f}")
        print(f"  🎯 Overall ρ (correlation): {weighted_rho:.4f}")
        print(f"  📦 Total samples: {total_samples}")
        print(f"  📉 NMSE(dB) distribution: {stats['nmse_db_dist']['mean']:.3f} ± {stats['nmse_db_dist']['std']:.3f}")
        print(f"  🎭 ρ distribution: {stats['rho_dist']['mean']:.4f} ± {stats['rho_dist']['std']:.4f}")

        return stats

    def _save_evaluation_results(self, stats, all_metrics):
        """Save evaluation results"""
        # Save statistical results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.eval_dir / f"evaluation_results_{self.split}_{timestamp}.json"

        # Prepare data for saving
        save_data = {
            'model_path': str(self.model_path),
            'config_path': str(self.config_path),
            'evaluation_time': datetime.now().isoformat(),
            'split': self.split,
            'checkpoint_info': self.checkpoint_info,
            'compression_info': self.compression_info,
            'overall_stats': stats,
            'detailed_metrics': all_metrics[:50]  # Only save first 50 detailed metrics
        }

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"💾 Results saved to {results_file}")

    def _plot_evaluation_results(self, all_metrics, stats):
        """Plot evaluation results (Fixed title issue)"""
        # Extract data
        nmse_db_values = [m['nmse_db'] for m in all_metrics]
        rho_values = [m['rho'] for m in all_metrics]

        # Get correct compression ratio title
        cr_title = self.compression_info['cr']
        dataset_type = stats['experiment_info']['dataset_type']

        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Model Evaluation Results - {dataset_type.title()} CR: {cr_title} ({self.split})',
                     fontsize=14, fontweight='bold')

        # 1. NMSE(dB) histogram
        axes[0, 0].hist(nmse_db_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(stats['overall']['nmse_db'], color='red', linestyle='--',
                           label=f'Overall: {stats["overall"]["nmse_db"]:.2f}')
        axes[0, 0].axvline(stats['nmse_db_dist']['mean'], color='orange', linestyle=':',
                           label=f'Batch Mean: {stats["nmse_db_dist"]["mean"]:.2f}')
        axes[0, 0].set_title('NMSE(dB) Distribution')
        axes[0, 0].set_xlabel('NMSE(dB)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. ρ histogram
        axes[0, 1].hist(rho_values, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(stats['overall']['rho'], color='blue', linestyle='--',
                           label=f'Overall: {stats["overall"]["rho"]:.4f}')
        axes[0, 1].axvline(stats['rho_dist']['mean'], color='orange', linestyle=':',
                           label=f'Batch Mean: {stats["rho_dist"]["mean"]:.4f}')
        axes[0, 1].set_title('Correlation Coefficient (ρ) Distribution')
        axes[0, 1].set_xlabel('ρ')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. NMSE vs ρ scatter plot
        axes[1, 0].scatter(nmse_db_values, rho_values, alpha=0.6, s=10)
        axes[1, 0].set_title('NMSE(dB) vs Correlation')
        axes[1, 0].set_xlabel('NMSE(dB)')
        axes[1, 0].set_ylabel('ρ')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Metrics over time (batch index)
        batch_indices = range(len(nmse_db_values))
        axes[1, 1].plot(batch_indices, nmse_db_values, alpha=0.7, label='NMSE(dB)', color='blue')
        ax_twin = axes[1, 1].twinx()
        ax_twin.plot(batch_indices, rho_values, alpha=0.7, label='ρ', color='red')

        axes[1, 1].set_title('Metrics vs Batch Index')
        axes[1, 1].set_xlabel('Batch Index')
        axes[1, 1].set_ylabel('NMSE(dB)', color='blue')
        ax_twin.set_ylabel('ρ', color='red')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = self.eval_dir / f"evaluation_plots_{self.split}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')

        print(f"📊 Evaluation plots saved to {plot_file}")
        plt.close()

    def _create_csi_visualizations(self, prediction_samples, target_samples):
        """Create CSI signal comparison visualizations - Completely safe version"""
        print(f"\n🎨 Creating CSI visualizations...")

        try:
            # Use fixed version of visualization module
            from utils.visualization import create_csi_visualizations

            # Merge samples
            all_predictions = torch.cat(prediction_samples[:2], dim=0)  # Only take first 2 batches
            all_targets = torch.cat(target_samples[:2], dim=0)

            # Get model information
            model_name = self.model.__class__.__name__
            cr_str = str(self.compression_info['cr'])
            dataset_type = self.cfg.data.get('split', 'Unknown')

            # Use safe version to create visualizations
            saved_files = create_csi_visualizations(
                all_targets,
                all_predictions,
                save_dir=str(self.viz_dir),
                model_name=model_name,
                compression_ratio=cr_str,
                dataset_type=dataset_type
            )

            if saved_files:
                print(f"✅ Created {len(saved_files)} CSI visualization files")
                for path in saved_files:
                    print(f"  📊 {Path(path).name}")
            else:
                print("⚠️ No visualizations were created, trying fallback...")
                # Fallback: built-in simple visualization
                fallback_path = self._create_fallback_visualization(
                    all_targets, all_predictions, model_name, cr_str, dataset_type
                )
                if fallback_path:
                    print(f"✅ Fallback visualization saved to {Path(fallback_path).name}")

        except ImportError as e:
            print(f"⚠️ Cannot import visualization module: {e}")
            # Use built-in simple visualization
            fallback_path = self._create_fallback_visualization(
                torch.cat(prediction_samples[:1], dim=0),
                torch.cat(target_samples[:1], dim=0),
                self.model.__class__.__name__,
                str(self.compression_info['cr']),
                self.cfg.data.get('split', 'Unknown')
            )
            if fallback_path:
                print(f"✅ Fallback visualization saved to {Path(fallback_path).name}")

        except Exception as e:
            print(f"❌ CSI visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def _create_fallback_visualization(self, targets, predictions, model_name, cr_str, dataset_type):
        """Create built-in fallback visualization - Safest solution"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Ensure correct data format
            if targets.ndim == 5:  # [B, T, 2, H, W]
                target_frame = targets[0, 0]  # Take first frame of first sample
                pred_frame = predictions[0, 0]
            elif targets.ndim == 4:  # [B, 2, H, W]
                target_frame = targets[0]
                pred_frame = predictions[0]
            else:
                return None

            # Convert to numpy
            target_frame = target_frame.detach().cpu().numpy()
            pred_frame = pred_frame.detach().cpu().numpy()

            # Calculate magnitude
            target_complex = target_frame[0] + 1j * target_frame[1]
            pred_complex = pred_frame[0] + 1j * pred_frame[1]

            target_mag = np.abs(target_complex)
            pred_mag = np.abs(pred_complex)
            error_mag = np.abs(pred_mag - target_mag)

            # Create simple 1x3 layout - completely without colorbar
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{model_name} {dataset_type} CR: {cr_str} - Fallback Comparison',
                         fontsize=14, fontweight='bold')

            # Original signal
            axes[0].imshow(target_mag, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            # Reconstructed signal
            axes[1].imshow(pred_mag, cmap='gray', vmin=target_mag.min(), vmax=target_mag.max())
            axes[1].set_title('Reconstructed')
            axes[1].axis('off')

            # Error
            axes[2].imshow(error_mag, cmap='Reds')
            axes[2].set_title('Error')
            axes[2].axis('off')

            # Add numeric range description (completely replace colorbar)
            range_text = f"Original: {target_mag.min():.3f} → {target_mag.max():.3f}\n"
            range_text += f"Reconstructed: {pred_mag.min():.3f} → {pred_mag.max():.3f}\n"
            range_text += f"Max Error: {error_mag.max():.3f}"

            fig.text(0.02, 0.02, range_text, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            # Calculate simple metrics
            try:
                orig_flat = target_mag.flatten()
                pred_flat = pred_mag.flatten()
                mse = np.mean((pred_flat - orig_flat) ** 2)
                power = np.mean(orig_flat ** 2)
                nmse_db = 10 * np.log10(mse / (power + 1e-12))
                rho = np.corrcoef(orig_flat, pred_flat)[0, 1]

                metrics_text = f"NMSE: {nmse_db:.2f} dB\nCorrelation: {rho:.4f}"
                fig.text(0.98, 0.02, metrics_text, fontsize=10, ha='right',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            except:
                pass

            plt.tight_layout()

            # Save
            save_name = f"fallback_comparison_{model_name}_{dataset_type}_cr{cr_str.replace('/', '_')}.png"
            save_path = self.viz_dir / save_name

            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            return str(save_path)

        except Exception as e:
            print(f"❌ Fallback visualization failed: {e}")
            return None

    def _save_prediction_samples(self, predictions, targets):
        """Save prediction samples (Simplified version)"""
        print(f"\n💾 Saving prediction samples...")

        # Merge all predictions and targets
        all_preds = torch.cat(predictions, dim=0)
        all_targets = torch.cat(targets, dim=0)

        # Save as .pt files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pred_file = self.eval_dir / f"predictions_{timestamp}.pt"
        target_file = self.eval_dir / f"targets_{timestamp}.pt"

        torch.save(all_preds, pred_file)
        torch.save(all_targets, target_file)

        print(f"✅ Saved predictions to {pred_file}")
        print(f"✅ Saved targets to {target_file}")

    def create_comprehensive_report(self, analyze_complexity=True, create_visualizations=True):
        """Create comprehensive evaluation report"""
        print(f"\n📋 Creating Comprehensive Evaluation Report...")

        results = {
            'model_info': {
                'model_path': str(self.model_path),
                'model_name': self.model.__class__.__name__,
                'config_path': str(self.config_path),
                'device': str(self.device),
                'split': self.split,
                'evaluation_time': datetime.now().isoformat()
            },
            'compression_info': self.compression_info,
            'checkpoint_info': self.checkpoint_info
        }

        # 1. Model complexity analysis
        if analyze_complexity:
            complexity_results = self.analyze_model_complexity()
            if complexity_results:
                results['complexity_analysis'] = complexity_results

        # 2. Performance evaluation
        performance_results = self.evaluate_on_dataset(
            save_predictions=True,
            create_visualizations=create_visualizations
        )
        results['performance_evaluation'] = performance_results

        # 3. Save comprehensive report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.eval_dir / f"comprehensive_report_{timestamp}.json"

        # Ensure results are JSON serializable
        serializable_results = self._make_json_serializable(results)

        with open(report_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"\n✅ Comprehensive report saved to {report_file}")

        # Print summary
        self._print_report_summary(results)

        return results

    def _print_report_summary(self, results):
        """Print report summary"""
        print(f"\n{'=' * 80}")
        print(f"📊 COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'=' * 80}")

        # Model information
        model_info = results['model_info']
        print(f"🏷️  Model: {model_info['model_name']}")
        print(f"📁 Path: {Path(model_info['model_path']).name}")
        print(f"🗂️  Split: {model_info['split']}")

        # Compression information
        comp_info = results['compression_info']
        print(f"📊 Compression Ratio: {comp_info['cr']} (M={comp_info['cr_num']})")

        # Complexity information
        if 'complexity_analysis' in results:
            complexity = results['complexity_analysis']
            params = complexity.get('parameters', {})
            flops = complexity.get('flops_main', {})

            print(f"🔢 Parameters: {params.get('total_params_M', 'N/A'):.2f}M")
            if 'error' not in flops:
                print(f"⚡ FLOPs: {flops.get('total_flops_G', 'N/A'):.2f}G")

        # Performance information
        if 'performance_evaluation' in results:
            perf = results['performance_evaluation']['overall']
            print(f"📈 NMSE: {perf['nmse_db']:.3f} dB")
            print(f"🎯 Correlation: {perf['rho']:.4f}")
            print(f"📦 Samples: {perf['total_samples']}")

        print(f"📁 Results saved in: {self.eval_dir}")
        print(f"{'=' * 80}\n")


def find_latest_model(checkpoint_dir="./checkpoints"):
    """Find latest model file (Improved version)"""
    checkpoint_path = Path(checkpoint_dir)

    # Find all .pth files, including final_model.pth
    model_files = []
    for pattern in ["**/final_model.pth", "**/final_*.pth", "**/stage2_*.pth", "**/*_best.pth"]:
        model_files.extend(checkpoint_path.glob(pattern))

    if not model_files:
        raise FileNotFoundError(f"No model files found in {checkpoint_dir}")

    # Sort by modification time, return the latest
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    return latest_model


def main():
    parser = argparse.ArgumentParser(description='Enhanced CR-LSTM-Net Model Evaluator')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (.pth)')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to config file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    # New feature options
    parser.add_argument('--analyze_complexity', action='store_true',
                        help='Analyze model complexity (FLOPs, parameters)')
    parser.add_argument('--create_visualizations', action='store_true',
                        help='Create CSI comparison visualizations')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction samples')
    parser.add_argument('--comprehensive', action='store_true',
                        help='Run comprehensive evaluation (all features)')

    args = parser.parse_args()

    # Determine model path
    if args.model is None:
        try:
            model_path = find_latest_model()
            print(f"🔍 Auto-detected latest model: {model_path}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return
    else:
        model_path = args.model

    # Ensure model file exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return

    # Create enhanced evaluator
    try:
        evaluator = EnhancedModelEvaluator(model_path, args.config, args.device, args.split)
    except Exception as e:
        print(f"❌ Failed to create evaluator: {e}")
        import traceback
        traceback.print_exc()
        return

    # Execute evaluation
    try:
        if args.comprehensive:
            # Comprehensive evaluation (includes all features)
            results = evaluator.create_comprehensive_report(
                analyze_complexity=True,
                create_visualizations=True
            )
        else:
            # Execute specified features separately
            if args.analyze_complexity:
                evaluator.analyze_model_complexity()

            # Basic performance evaluation
            stats = evaluator.evaluate_on_dataset(
                save_predictions=args.save_predictions,
                create_visualizations=args.create_visualizations
            )

        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 All results saved in: {evaluator.eval_dir}")

    except KeyboardInterrupt:
        print(f"\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()