# utils/visualization.py - CSI signal visualization comparison tool (Completely fixed version)
"""
CSI signal visualization module
Supports various comparison methods for original vs reconstructed signals - completely avoids colorbar issues
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Set matplotlib to support Chinese characters and better display effects
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class CSIVisualizer:
    """CSI signal visualizer"""

    def __init__(self, save_dir: Union[str, Path] = "./visualizations"):
        """
        Args:
            save_dir: Save directory
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def complex_to_magnitude_phase(self, csi_data: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert CSI complex data to magnitude and phase

        Args:
            csi_data: [2, H, W] or [B, 2, H, W] or [B, T, 2, H, W]

        Returns:
            magnitude, phase: Magnitude and phase arrays
        """
        # Ensure it's numpy array
        if isinstance(csi_data, torch.Tensor):
            csi_data = csi_data.detach().cpu().numpy()

        # Handle different dimensions
        if csi_data.ndim == 3:  # [2, H, W]
            real, imag = csi_data[0], csi_data[1]
        elif csi_data.ndim == 4:  # [B, 2, H, W] - take first sample
            real, imag = csi_data[0, 0], csi_data[0, 1]
        elif csi_data.ndim == 5:  # [B, T, 2, H, W] - take first sample's first frame
            real, imag = csi_data[0, 0, 0], csi_data[0, 0, 1]
        else:
            raise ValueError(f"Unsupported CSI data shape: {csi_data.shape}")

        # Convert to complex
        complex_csi = real + 1j * imag

        # Calculate magnitude and phase
        magnitude = np.abs(complex_csi)
        phase = np.angle(complex_csi)

        return magnitude, phase

    def plot_single_comparison(self,
                               original: torch.Tensor,
                               reconstructed: torch.Tensor,
                               title: str = "CSI Comparison",
                               sample_idx: int = 0,
                               time_idx: int = 0,
                               metrics: Optional[Dict] = None,
                               save_name: Optional[str] = None) -> str:
        """
        Plot single sample comparison (original vs reconstructed vs error) - colorbar-free safe version

        Args:
            original: Original CSI [B, T, 2, H, W] or [B, 2, H, W]
            reconstructed: Reconstructed CSI, same shape as above
            title: Chart title
            sample_idx: Sample index
            time_idx: Time index (for sequential data)
            metrics: Evaluation metrics dictionary
            save_name: Save filename

        Returns:
            str: Path of saved file
        """
        try:
            # Ensure input is 5D [B, T, 2, H, W]
            if original.ndim == 4:  # [B, 2, H, W]
                original = original.unsqueeze(1)
            if reconstructed.ndim == 4:
                reconstructed = reconstructed.unsqueeze(1)

            # Extract specified sample and time
            orig_frame = original[sample_idx, time_idx]  # [2, H, W]
            recon_frame = reconstructed[sample_idx, time_idx]  # [2, H, W]

            # Convert to magnitude and phase
            orig_mag, orig_phase = self.complex_to_magnitude_phase(orig_frame)
            recon_mag, recon_phase = self.complex_to_magnitude_phase(recon_frame)

            # Calculate error
            mag_error = np.abs(recon_mag - orig_mag)
            phase_error = np.abs(np.angle(np.exp(1j * (recon_phase - orig_phase))))

            # Create figure - 2 rows 3 columns, remove third row to avoid colorbar complexity
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            fig.suptitle(f'{title} - Sample {sample_idx}, Time {time_idx}', fontsize=16, fontweight='bold')

            # Set unified color scale range
            mag_vmax = max(orig_mag.max(), recon_mag.max())
            mag_vmin = 0

            # First row: Magnitude comparison
            axes[0, 0].imshow(orig_mag, cmap='viridis', vmin=mag_vmin, vmax=mag_vmax)
            axes[0, 0].set_title('Original Magnitude', fontsize=12)
            axes[0, 0].axis('off')

            axes[0, 1].imshow(recon_mag, cmap='viridis', vmin=mag_vmin, vmax=mag_vmax)
            axes[0, 1].set_title('Reconstructed Magnitude', fontsize=12)
            axes[0, 1].axis('off')

            axes[0, 2].imshow(mag_error, cmap='Reds', vmin=0)
            axes[0, 2].set_title('Magnitude Error', fontsize=12)
            axes[0, 2].axis('off')

            # Second row: Phase comparison
            axes[1, 0].imshow(orig_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
            axes[1, 0].set_title('Original Phase', fontsize=12)
            axes[1, 0].axis('off')

            axes[1, 1].imshow(recon_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
            axes[1, 1].set_title('Reconstructed Phase', fontsize=12)
            axes[1, 1].axis('off')

            axes[1, 2].imshow(phase_error, cmap='Reds', vmin=0)
            axes[1, 2].set_title('Phase Error', fontsize=12)
            axes[1, 2].axis('off')

            # Add numerical range description (replace colorbar)
            range_text = f"Magnitude: {mag_vmin:.2f} → {mag_vmax:.2f}\n"
            range_text += f"Phase: -π → π\n"
            range_text += f"Max Mag Error: {mag_error.max():.3f}\n"
            range_text += f"Max Phase Error: {phase_error.max():.3f}"

            fig.text(0.02, 0.02, range_text, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            # Add statistical information
            if metrics:
                stats_text = f"NMSE: {metrics.get('nmse_db', 'N/A'):.2f} dB\n"
                stats_text += f"ρ: {metrics.get('rho', 'N/A'):.4f}\n"
                stats_text += f"MSE: {metrics.get('mse', 'N/A'):.6f}"

                fig.text(0.98, 0.02, stats_text, fontsize=12, ha='right',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            plt.tight_layout()

            # Save image
            if save_name is None:
                save_name = f"csi_comparison_s{sample_idx}_t{time_idx}.png"

            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            return str(save_path)

        except Exception as e:
            print(f"❌ Single comparison visualization failed: {e}")
            return None

    def plot_multi_sample_grid_safe(self,
                                    original: torch.Tensor,
                                    reconstructed: torch.Tensor,
                                    n_samples: int = 6,
                                    title: str = "CSI Multi-Sample Comparison (Safe)",
                                    time_idx: int = 0,
                                    save_name: Optional[str] = None) -> str:
        """
        Plot multi-sample grid comparison - completely safe version (no colorbar)

        Args:
            original: Original CSI [B, T, 2, H, W]
            reconstructed: Reconstructed CSI
            n_samples: Number of samples to display
            title: Chart title
            time_idx: Time index
            save_name: Save filename

        Returns:
            str: Path of saved file
        """
        try:
            # Ensure input is 5D
            if original.ndim == 4:
                original = original.unsqueeze(1)
            if reconstructed.ndim == 4:
                reconstructed = reconstructed.unsqueeze(1)

            batch_size = original.size(0)
            n_samples = min(n_samples, batch_size)

            if n_samples < 1:
                print("No samples to visualize.")
                return None

            # Select samples (simple evenly-spaced selection)
            sample_indices = np.linspace(0, batch_size - 1, n_samples, dtype=int)

            # Create figure - 3 row layout
            fig, axes = plt.subplots(3, n_samples, figsize=(2 * n_samples, 6))

            if n_samples == 1:
                axes = axes.reshape(3, 1)

            fig.suptitle(title, fontsize=14, fontweight='bold')

            # Pre-calculate all data and ranges
            all_orig_mags = []
            all_recon_mags = []
            all_errors = []

            for idx in sample_indices:
                orig_frame = original[idx, time_idx].detach().cpu().numpy()
                recon_frame = reconstructed[idx, time_idx].detach().cpu().numpy()

                # Calculate complex magnitude
                orig_complex = orig_frame[0] + 1j * orig_frame[1]
                recon_complex = recon_frame[0] + 1j * recon_frame[1]

                orig_mag = np.abs(orig_complex)
                recon_mag = np.abs(recon_complex)
                error_mag = np.abs(recon_mag - orig_mag)

                all_orig_mags.append(orig_mag)
                all_recon_mags.append(recon_mag)
                all_errors.append(error_mag)

            # Calculate global range
            global_vmax = max([mag.max() for mag in all_orig_mags + all_recon_mags])
            global_vmin = 0
            error_vmax = max([err.max() for err in all_errors])

            # Plot each sample
            for col, sample_idx in enumerate(sample_indices):
                orig_mag = all_orig_mags[col]
                recon_mag = all_recon_mags[col]
                error_mag = all_errors[col]

                # First row: Original signal
                axes[0, col].imshow(orig_mag, cmap='gray', vmin=global_vmin, vmax=global_vmax)
                axes[0, col].axis('off')
                axes[0, col].set_title(f'Sample {sample_idx}', fontsize=10)
                if col == 0:
                    axes[0, col].text(-0.1, 0.5, 'Original', transform=axes[0, col].transAxes,
                                      fontsize=12, rotation=90, va='center', ha='right')

                # Second row: Reconstructed signal
                axes[1, col].imshow(recon_mag, cmap='gray', vmin=global_vmin, vmax=global_vmax)
                axes[1, col].axis('off')
                if col == 0:
                    axes[1, col].text(-0.1, 0.5, 'Reconstructed', transform=axes[1, col].transAxes,
                                      fontsize=12, rotation=90, va='center', ha='right')

                # Third row: Error
                axes[2, col].imshow(error_mag, cmap='Reds', vmin=0, vmax=error_vmax)
                axes[2, col].axis('off')
                if col == 0:
                    axes[2, col].text(-0.1, 0.5, '|Error|', transform=axes[2, col].transAxes,
                                      fontsize=12, rotation=90, va='center', ha='right')

                # Add metrics text
                try:
                    from losses.metrics import compute_metrics
                    orig_tensor = original[sample_idx:sample_idx + 1, time_idx:time_idx + 1]
                    recon_tensor = reconstructed[sample_idx:sample_idx + 1, time_idx:time_idx + 1]
                    metrics = compute_metrics(recon_tensor, orig_tensor)

                    metrics_text = f"{metrics['nmse_db']:.1f}dB\nρ={metrics['rho']:.3f}"
                    axes[1, col].text(0.05, 0.95, metrics_text, transform=axes[1, col].transAxes,
                                      color='lime', fontsize=8, va='top',
                                      bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                except Exception:
                    pass

            # Add range description (replace colorbar)
            range_text = f"Grayscale: {global_vmin:.2f} → {global_vmax:.2f}\n"
            range_text += f"Error (Red): 0 → {error_vmax:.3f}"

            fig.text(0.02, 0.02, range_text, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            plt.tight_layout()

            # Save image
            if save_name is None:
                save_name = f"csi_safe_comparison_{n_samples}samples_t{time_idx}.png"

            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✅ Safe multi-sample comparison saved to {save_path}")
            return str(save_path)

        except Exception as e:
            print(f"❌ Safe multi-sample comparison failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_ultra_simple_comparison(self,
                                       original: torch.Tensor,
                                       reconstructed: torch.Tensor,
                                       title: str = "CSI Ultra Simple Comparison",
                                       save_name: Optional[str] = None) -> str:
        """
        Create ultra-simple 1x3 comparison plot - last fallback option
        """
        try:
            # Take first sample
            if original.ndim == 5:
                orig_frame = original[0, 0]
                recon_frame = reconstructed[0, 0]
            elif original.ndim == 4:
                orig_frame = original[0]
                recon_frame = reconstructed[0]
            else:
                orig_frame = original
                recon_frame = reconstructed

            # Convert to numpy and calculate magnitude
            orig_frame = orig_frame.detach().cpu().numpy()
            recon_frame = recon_frame.detach().cpu().numpy()

            orig_complex = orig_frame[0] + 1j * orig_frame[1]
            recon_complex = recon_frame[0] + 1j * recon_frame[1]

            orig_mag = np.abs(orig_complex)
            recon_mag = np.abs(recon_complex)
            error_mag = np.abs(recon_mag - orig_mag)

            # Create simple 1x3 layout
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(title, fontsize=14)

            # Three subplots
            axes[0].imshow(orig_mag, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(recon_mag, cmap='gray')
            axes[1].set_title('Reconstructed')
            axes[1].axis('off')

            axes[2].imshow(error_mag, cmap='Reds')
            axes[2].set_title('Error')
            axes[2].axis('off')

            # Add range description
            range_text = f"Original: {orig_mag.min():.3f} → {orig_mag.max():.3f}\n"
            range_text += f"Reconstructed: {recon_mag.min():.3f} → {recon_mag.max():.3f}\n"
            range_text += f"Error: 0 → {error_mag.max():.3f}"

            fig.text(0.02, 0.02, range_text, fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            plt.tight_layout()

            if save_name is None:
                save_name = "ultra_simple_csi_comparison.png"

            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✅ Ultra simple comparison saved to {save_path}")
            return str(save_path)

        except Exception as e:
            print(f"❌ Ultra simple comparison failed: {e}")
            return None

    def create_comparison_report_safe(self,
                                      original: torch.Tensor,
                                      reconstructed: torch.Tensor,
                                      model_name: str = "CRLSTMNet",
                                      compression_ratio: str = "1/32",
                                      dataset_type: str = "Indoor") -> List[str]:
        """
        Create safe version of comparison report (no colorbar)

        Args:
            original: Original CSI
            reconstructed: Reconstructed CSI
            model_name: Model name
            compression_ratio: Compression ratio
            dataset_type: Dataset type

        Returns:
            List[str]: All saved file paths
        """
        saved_files = []
        base_title = f"{model_name} {dataset_type} CR: {compression_ratio}"

        # 1. Safe version multi-sample grid comparison
        try:
            grid_path = self.plot_multi_sample_grid_safe(
                original, reconstructed,
                n_samples=6,
                title=f"{base_title} - Multi-Sample Comparison",
                save_name=f"safe_comparison_{model_name}_{dataset_type}_cr{compression_ratio.replace('/', '_')}.png"
            )
            if grid_path:
                saved_files.append(grid_path)
        except Exception as e:
            print(f"Warning: Safe grid plot failed: {e}")

        # 2. Single sample detailed comparison
        try:
            single_path = self.plot_single_comparison(
                original, reconstructed,
                title=f"{base_title} - Detailed View",
                sample_idx=0,
                save_name=f"detailed_comparison_{model_name}_{dataset_type}_cr{compression_ratio.replace('/', '_')}.png"
            )
            if single_path:
                saved_files.append(single_path)
        except Exception as e:
            print(f"Warning: Single comparison failed: {e}")

        # 3. Ultra-simple fallback option
        if not saved_files:
            try:
                simple_path = self.create_ultra_simple_comparison(
                    original, reconstructed,
                    title=f"{base_title} - Simple View",
                    save_name=f"simple_comparison_{model_name}_{dataset_type}_cr{compression_ratio.replace('/', '_')}.png"
                )
                if simple_path:
                    saved_files.append(simple_path)
            except Exception as e:
                print(f"Warning: Simple comparison failed: {e}")

        print(f"✅ Safe comparison report created with {len(saved_files)} visualizations")
        for path in saved_files:
            print(f"  📊 {Path(path).name}")

        return saved_files


def create_csi_visualizations(original: torch.Tensor,
                              reconstructed: torch.Tensor,
                              save_dir: str = "./visualizations",
                              model_name: str = "CRLSTMNet",
                              compression_ratio: str = "1/32",
                              dataset_type: str = "Indoor") -> List[str]:
    """
    Convenience function: Create safe CSI visualization comparison plots

    Args:
        original: Original CSI [B, T, 2, H, W] or [B, 2, H, W]
        reconstructed: Reconstructed CSI, same shape as above
        save_dir: Save directory
        model_name: Model name
        compression_ratio: Compression ratio (e.g., "1/32")
        dataset_type: Dataset type

    Returns:
        List[str]: All saved file paths
    """
    visualizer = CSIVisualizer(save_dir)
    return visualizer.create_comparison_report_safe(
        original, reconstructed, model_name, compression_ratio, dataset_type
    )