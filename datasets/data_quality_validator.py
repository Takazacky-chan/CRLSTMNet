# data_quality_validator.py - Validate 32×64→32×32 downsampling quality
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path


class DataQualityValidator:
    """Tool class for validating data downsampling quality"""

    def __init__(self, data_path):
        self.data_path = Path(data_path)

    def load_sample_data(self, max_files=3, max_frames_per_file=100):
        """Load sample data for analysis"""
        mat_files = list(self.data_path.glob("*.mat"))[:max_files]

        all_64_data = []
        all_32_data = []

        for file_path in mat_files:
            print(f"Loading {file_path.name}...")
            mat_data = sio.loadmat(file_path)

            # Find data variable
            h_matrix = None
            possible_keys = ['Hur_down_t1', 'H', 'channel', 'CSI']
            for key in possible_keys:
                if key in mat_data:
                    h_matrix = mat_data[key]
                    break

            if h_matrix is None:
                continue

            # Check if it's 32×64 complex data
            if h_matrix.ndim == 3 and h_matrix.shape[-2:] == (32, 64) and \
                    h_matrix.dtype in (np.complex64, np.complex128):
                # Take first max_frames_per_file frames
                frames_to_use = min(max_frames_per_file, h_matrix.shape[0])
                sample_data = h_matrix[:frames_to_use]

                all_64_data.append(sample_data)

                # Apply different downsampling methods
                downsampled = self._downsample_methods_comparison(sample_data)
                all_32_data.append(downsampled)

        if not all_64_data:
            raise ValueError("No suitable 32×64 complex data found!")

        return np.concatenate(all_64_data, axis=0), all_32_data

    def _downsample_methods_comparison(self, complex_data_64):
        """Compare different downsampling methods"""
        methods = {}

        # Method 1: Simple averaging (not recommended)
        reshaped = complex_data_64.reshape(complex_data_64.shape[0], 32, 32, 2)
        methods['simple_avg'] = np.mean(reshaped, axis=-1)

        # Method 2: Smart averaging + energy normalization (recommended)
        methods['smart_avg'] = np.mean(reshaped, axis=-1) * np.sqrt(2.0)

        # Method 3: Center cropping
        start_idx = (64 - 32) // 2
        methods['center_crop'] = complex_data_64[:, :, start_idx:start_idx + 32]

        # Method 4: Low-pass filtering + decimation (if scipy available)
        try:
            from scipy import signal
            b = np.array([0.25, 0.5, 0.25])
            filtered_data = np.zeros((complex_data_64.shape[0], 32, 32), dtype=complex_data_64.dtype)

            for t in range(complex_data_64.shape[0]):
                for ant in range(32):
                    filtered = signal.filtfilt(b, [1.0], complex_data_64[t, ant, :])
                    filtered_data[t, ant, :] = filtered[::2]

            methods['lowpass_decimate'] = filtered_data
        except ImportError:
            print("Scipy not available, skipping lowpass_decimate method")

        return methods

    def analyze_energy_preservation(self, data_64, data_32_methods):
        """Analyze energy preservation"""
        print("\n" + "=" * 60)
        print("ENERGY PRESERVATION ANALYSIS")
        print("=" * 60)

        orig_power = np.mean(np.abs(data_64) ** 2)
        print(f"Original 64-point power: {orig_power:.6f}")

        results = {}

        for file_idx, methods_dict in enumerate(data_32_methods):
            print(f"\nFile {file_idx + 1}:")

            for method_name, downsampled_data in methods_dict.items():
                down_power = np.mean(np.abs(downsampled_data) ** 2)
                power_ratio = down_power / orig_power if orig_power > 0 else 0
                power_ratio_db = 10 * np.log10(power_ratio) if power_ratio > 0 else -np.inf

                if method_name not in results:
                    results[method_name] = []
                results[method_name].append(power_ratio_db)

                status = "✅" if abs(power_ratio_db) < 1.0 else "⚠️" if abs(power_ratio_db) < 3.0 else "❌"
                print(f"  {method_name:15s}: {power_ratio_db:+6.2f} dB  {status}")

        # Statistical results
        print(f"\n{'Method':<15s} {'Mean ΔP(dB)':<12s} {'Std(dB)':<10s} {'Status'}")
        print("-" * 50)

        for method_name, power_changes in results.items():
            mean_change = np.mean(power_changes)
            std_change = np.std(power_changes)
            status = "✅ Good" if abs(mean_change) < 1.0 else "⚠️ Fair" if abs(mean_change) < 3.0 else "❌ Poor"
            print(f"{method_name:<15s} {mean_change:+6.2f}      {std_change:6.2f}    {status}")

        return results

    def analyze_frequency_response(self, data_64, data_32_methods, num_samples=5):
        """Analyze frequency response preservation"""
        print("\n" + "=" * 60)
        print("FREQUENCY RESPONSE ANALYSIS")
        print("=" * 60)

        # Randomly select several frames for analysis
        sample_indices = np.random.choice(data_64.shape[0], min(num_samples, data_64.shape[0]), replace=False)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for method_idx, (method_name, downsampled_data) in enumerate(data_32_methods[0].items()):
            if method_idx >= 4:  # Show at most 4 methods
                break

            ax = axes[method_idx]

            for sample_idx in sample_indices:
                # Select frequency response of first antenna
                original_freq = data_64[sample_idx, 0, :]  # 64 points
                downsampled_freq = downsampled_data[sample_idx, 0, :]  # 32 points

                # Interpolate 32 points back to 64 points for comparison
                from scipy.interpolate import interp1d
                f_32 = np.linspace(0, 1, 32)
                f_64 = np.linspace(0, 1, 64)

                real_interp = interp1d(f_32, np.real(downsampled_freq), kind='linear')
                imag_interp = interp1d(f_32, np.imag(downsampled_freq), kind='linear')
                reconstructed = real_interp(f_64) + 1j * imag_interp(f_64)

                # Plot magnitude spectrum
                ax.plot(f_64, np.abs(original_freq), 'b-', alpha=0.3, linewidth=0.8,
                        label='Original' if sample_idx == sample_indices[0] else "")
                ax.plot(f_64, np.abs(reconstructed), 'r--', alpha=0.7, linewidth=0.8,
                        label='Reconstructed' if sample_idx == sample_indices[0] else "")

            ax.set_title(f'Method: {method_name}')
            ax.set_xlabel('Normalized Frequency')
            ax.set_ylabel('Magnitude')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('frequency_response_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def compute_reconstruction_metrics(self, data_64, data_32_methods):
        """Compute reconstruction quality metrics"""
        print("\n" + "=" * 60)
        print("RECONSTRUCTION QUALITY METRICS")
        print("=" * 60)

        results = {}

        for method_name, downsampled_data in data_32_methods[0].items():
            # Upsample 32 points back to 64 points
            upsampled = self._upsample_32_to_64(downsampled_data)

            # Calculate NMSE
            mse = np.mean(np.abs(data_64 - upsampled) ** 2)
            signal_power = np.mean(np.abs(data_64) ** 2)
            nmse_linear = mse / signal_power if signal_power > 0 else np.inf
            nmse_db = 10 * np.log10(nmse_linear) if nmse_linear > 0 else -np.inf

            # Calculate correlation coefficient
            data_64_flat = data_64.flatten()
            upsampled_flat = upsampled.flatten()
            correlation = np.abs(np.corrcoef(data_64_flat.real, upsampled_flat.real)[0, 1])

            results[method_name] = {
                'nmse_db': nmse_db,
                'correlation': correlation,
                'mse': mse
            }

            print(f"{method_name:15s}: NMSE = {nmse_db:6.2f} dB, ρ = {correlation:.4f}")

        return results

    def _upsample_32_to_64(self, data_32):
        """Upsample 32-point data to 64 points (simple repetition method)"""
        # Repeat each 32-point twice to become 64 points
        return np.repeat(data_32, 2, axis=-1)

    def generate_comprehensive_report(self, data_path, output_file='data_quality_report.txt'):
        """Generate comprehensive quality report"""
        print("Generating comprehensive data quality report...")

        try:
            # Load data
            data_64, data_32_methods = self.load_sample_data()

            # Analyze energy preservation
            energy_results = self.analyze_energy_preservation(data_64, data_32_methods)

            # Analyze frequency response
            freq_fig = self.analyze_frequency_response(data_64, data_32_methods)

            # Calculate reconstruction metrics
            recon_results = self.compute_reconstruction_metrics(data_64, data_32_methods)

            # Generate report
            with open(output_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("DATA QUALITY VALIDATION REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Data path: {self.data_path}\n")
                f.write(f"Analysis date: {np.datetime64('now')}\n\n")

                f.write("SUMMARY RECOMMENDATIONS:\n")
                f.write("-" * 40 + "\n")

                # Find best method
                best_method = min(recon_results.keys(),
                                  key=lambda k: recon_results[k]['nmse_db'])

                f.write(f"🏆 Best overall method: {best_method}\n")
                f.write(f"   NMSE: {recon_results[best_method]['nmse_db']:.2f} dB\n")
                f.write(f"   Correlation: {recon_results[best_method]['correlation']:.4f}\n")

                # Best energy preservation method
                best_energy_method = min(energy_results.keys(),
                                         key=lambda k: abs(np.mean(energy_results[k])))
                f.write(f"⚡ Best energy preservation: {best_energy_method}\n")
                f.write(f"   Average power change: {np.mean(energy_results[best_energy_method]):+.2f} dB\n\n")

                f.write("DETAILED RESULTS:\n")
                f.write("-" * 40 + "\n")
                for method_name in recon_results.keys():
                    f.write(f"\n{method_name.upper()}:\n")
                    f.write(f"  NMSE: {recon_results[method_name]['nmse_db']:6.2f} dB\n")
                    f.write(f"  Correlation: {recon_results[method_name]['correlation']:6.4f}\n")
                    f.write(
                        f"  Power change: {np.mean(energy_results[method_name]):+6.2f} ± {np.std(energy_results[method_name]):5.2f} dB\n")

            print(f"\n📄 Comprehensive report saved to: {output_file}")
            print(f"📊 Frequency response plot saved to: frequency_response_comparison.png")

            return {
                'energy_results': energy_results,
                'recon_results': recon_results,
                'best_method': best_method,
                'best_energy_method': best_energy_method
            }

        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return None


def main():
    """Main function - validate data quality"""
    import argparse

    parser = argparse.ArgumentParser(description='Validate 32×64 to 32×32 downsampling quality')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to directory containing .mat files')
    parser.add_argument('--output_report', type=str, default='data_quality_report.txt',
                        help='Output report filename')

    args = parser.parse_args()

    # Create validator
    validator = DataQualityValidator(args.data_path)

    # Generate report
    results = validator.generate_comprehensive_report(args.data_path, args.output_report)

    if results:
        print("\n" + "=" * 60)
        print("QUICK RECOMMENDATIONS")
        print("=" * 60)
        print(f"🎯 For training: Use '{results['best_method']}' method")
        print(f"⚡ For energy preservation: Use '{results['best_energy_method']}' method")
        print("\nAdd to your config file:")
        print("data:")
        print(f"  downsample_method: '{results['best_method']}'")
        print("  stride: 5  # or 10 for non-overlapping")


if __name__ == '__main__':
    main()