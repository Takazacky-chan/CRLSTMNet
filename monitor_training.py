# monitor_training.py - Training monitoring and visualization
"""
Training monitoring script providing real-time monitoring and result analysis functionality

Usage:
- Real-time monitoring: python monitor_training.py --watch
- Analyze results: python monitor_training.py --analyze
- Plot curves: python monitor_training.py --plot
"""

import argparse
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrainingMonitor:
    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_file = self.checkpoint_dir / "training.log"
        self.summary_file = self.checkpoint_dir / "training_summary.json"
        self.metrics_file = self.checkpoint_dir / "metrics.json"

    def watch_training(self):
        """Real-time monitoring of training process"""
        print("=== Real-time Training Monitor ===")
        print("Press Ctrl+C to stop monitoring\n")

        last_size = 0
        try:
            while True:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size > last_size:
                        # Read new content
                        with open(self.log_file, 'r', encoding='utf-8') as f:
                            f.seek(last_size)
                            new_content = f.read()
                            if new_content.strip():
                                print(new_content, end='')
                        last_size = current_size

                time.sleep(2)  # Check every 2 seconds

        except KeyboardInterrupt:
            print("\n=== Monitoring stopped ===")

    def analyze_results(self):
        """Analyze training results"""
        print("=== Training Results Analysis ===\n")

        # Analyze summary file
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                summary = json.load(f)

            print("📊 Training Summary:")
            print(f"  Total Parameters: {summary.get('total_params', 'N/A'):,}")
            print(f"  Trainable Parameters: {summary.get('trainable_params', 'N/A'):,}")
            print(f"  GPU: {summary.get('gpu_info', {}).get('name', 'N/A')}")
            print(f"  GPU Memory: {summary.get('gpu_info', {}).get('memory_gb', 'N/A'):.1f}GB")
            print()

            # Results for each stage
            for stage in ['stage0', 'stage1', 'stage2']:
                if stage in summary:
                    stage_info = summary[stage]
                    if stage_info.get('completed', False):
                        print(f"✅ {stage.upper()} completed")
                        if 'best_loss' in stage_info:
                            print(f"    Best Loss: {stage_info['best_loss']:.6f}")
                        if stage == 'stage0':
                            print(f"    Low-CR Loss: {stage_info.get('low_loss', 'N/A'):.6f}")
                            print(f"    High-CR Loss: {stage_info.get('high_loss', 'N/A'):.6f}")
                    else:
                        print(f"❌ {stage.upper()} not completed")
            print()

        # Check checkpoint files
        checkpoints = list(self.checkpoint_dir.glob("*.pth"))
        if checkpoints:
            print("💾 Available Checkpoints:")
            for ckpt in sorted(checkpoints):
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(ckpt.stat().st_mtime)
                print(f"  {ckpt.name} ({size_mb:.1f}MB, {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")

        # Parse training logs to get detailed metrics
        self._parse_training_logs()

    def _parse_training_logs(self):
        """Parse training logs to extract metrics"""
        if not self.log_file.exists():
            return

        print("\n📈 Training Progress:")

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Extract key metrics
            stage0_metrics = []
            stage1_metrics = []
            stage2_metrics = []

            current_stage = None

            for line in lines:
                line = line.strip()

                # Identify training stage
                if "Stage 0:" in line:
                    current_stage = 0
                elif "Stage 1:" in line:
                    current_stage = 1
                elif "Stage 2:" in line:
                    current_stage = 2

                # Extract validation metrics
                if "NMSE(dB):" in line and "ρ:" in line:
                    try:
                        # Parse "Val Loss: 0.123456 | NMSE(dB): -5.67 | ρ: 0.8901"
                        parts = line.split("|")

                        val_loss = None
                        nmse_db = None
                        rho = None

                        for part in parts:
                            part = part.strip()
                            if part.startswith("Val Loss:"):
                                val_loss = float(part.split(":")[1].strip())
                            elif "NMSE(dB):" in part:
                                nmse_db = float(part.split(":")[1].strip())
                            elif "ρ:" in part:
                                rho = float(part.split(":")[1].strip())

                        if val_loss and nmse_db and rho:
                            metric = {
                                'val_loss': val_loss,
                                'nmse_db': nmse_db,
                                'rho': rho
                            }

                            if current_stage == 0:
                                stage0_metrics.append(metric)
                            elif current_stage == 1:
                                stage1_metrics.append(metric)
                            elif current_stage == 2:
                                stage2_metrics.append(metric)

                    except (ValueError, IndexError):
                        continue

            # Display best results for each stage
            stages_data = [
                ("Stage 0", stage0_metrics),
                ("Stage 1", stage1_metrics),
                ("Stage 2", stage2_metrics)
            ]

            for stage_name, metrics in stages_data:
                if metrics:
                    best_idx = np.argmin([m['val_loss'] for m in metrics])
                    best = metrics[best_idx]
                    print(f"  {stage_name} Best: Loss={best['val_loss']:.6f}, "
                          f"NMSE(dB)={best['nmse_db']:.2f}, ρ={best['rho']:.4f}")

                    # Calculate improvement
                    if len(metrics) > 1:
                        first = metrics[0]
                        improvement_db = best['nmse_db'] - first['nmse_db']
                        improvement_rho = best['rho'] - first['rho']
                        print(f"  {stage_name} Improvement: NMSE(dB) {improvement_db:+.2f}, "
                              f"ρ {improvement_rho:+.4f}")

        except Exception as e:
            print(f"Error parsing logs: {e}")

    def plot_training_curves(self):
        """Plot training curves"""
        print("=== Plotting Training Curves ===")

        if not self.log_file.exists():
            print("❌ No training log found")
            return

        # Parse log data
        metrics_data = self._extract_training_metrics()

        if not any(metrics_data.values()):
            print("❌ No training metrics found in logs")
            return

        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CR-LSTM-Net Training Progress', fontsize=16, fontweight='bold')

        # Color mapping
        colors = {'Stage 0': '#1f77b4', 'Stage 1': '#ff7f0e', 'Stage 2': '#2ca02c'}

        # 1. Validation loss curve
        ax1 = axes[0, 0]
        for stage, data in metrics_data.items():
            if data:
                epochs = range(len(data))
                val_losses = [d['val_loss'] for d in data]
                ax1.plot(epochs, val_losses, 'o-', label=stage, color=colors.get(stage, 'gray'), linewidth=2,
                         markersize=4)

        ax1.set_title('Validation Loss', fontweight='bold')
        ax1.set_xlabel('Validation Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. NMSE(dB) curve
        ax2 = axes[0, 1]
        for stage, data in metrics_data.items():
            if data:
                epochs = range(len(data))
                nmse_db = [d['nmse_db'] for d in data]
                ax2.plot(epochs, nmse_db, 's-', label=stage, color=colors.get(stage, 'gray'), linewidth=2, markersize=4)

        ax2.set_title('NMSE (dB)', fontweight='bold')
        ax2.set_xlabel('Validation Steps')
        ax2.set_ylabel('NMSE (dB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. ρ (correlation coefficient) curve
        ax3 = axes[1, 0]
        for stage, data in metrics_data.items():
            if data:
                epochs = range(len(data))
                rho_values = [d['rho'] for d in data]
                ax3.plot(epochs, rho_values, '^-', label=stage, color=colors.get(stage, 'gray'), linewidth=2,
                         markersize=4)

        ax3.set_title('Correlation Coefficient (ρ)', fontweight='bold')
        ax3.set_xlabel('Validation Steps')
        ax3.set_ylabel('ρ')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        # 4. Best metrics comparison by stage
        ax4 = axes[1, 1]
        stage_names = []
        best_nmse = []
        best_rho = []

        for stage, data in metrics_data.items():
            if data:
                stage_names.append(stage)
                best_loss_idx = np.argmin([d['val_loss'] for d in data])
                best_nmse.append(data[best_loss_idx]['nmse_db'])
                best_rho.append(data[best_loss_idx]['rho'])

        if stage_names:
            x = np.arange(len(stage_names))
            width = 0.35

            # Dual axis display
            ax4_twin = ax4.twinx()

            bars1 = ax4.bar(x - width / 2, best_nmse, width, label='NMSE (dB)', color='skyblue', alpha=0.8)
            bars2 = ax4_twin.bar(x + width / 2, best_rho, width, label='ρ', color='lightcoral', alpha=0.8)

            ax4.set_title('Best Performance by Stage', fontweight='bold')
            ax4.set_xlabel('Training Stage')
            ax4.set_ylabel('NMSE (dB)', color='blue')
            ax4_twin.set_ylabel('ρ', color='red')
            ax4.set_xticks(x)
            ax4.set_xticklabels(stage_names)

            # Add value labels
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                height1 = bar1.get_height()
                height2 = bar2.get_height()
                ax4.text(bar1.get_x() + bar1.get_width() / 2., height1,
                         f'{height1:.2f}', ha='center', va='bottom', fontsize=9)
                ax4_twin.text(bar2.get_x() + bar2.get_width() / 2., height2,
                              f'{height2:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save chart
        plot_path = self.checkpoint_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {plot_path}")

        plt.show()

    def _extract_training_metrics(self):
        """Extract training metrics from logs"""
        metrics_data = {
            'Stage 0': [],
            'Stage 1': [],
            'Stage 2': []
        }

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            current_stage = None

            for line in lines:
                line = line.strip()

                # Identify training stage
                if "Stage 0:" in line or "LOW-CR" in line or "HIGH-CR" in line:
                    current_stage = 'Stage 0'
                elif "Stage 1:" in line:
                    current_stage = 'Stage 1'
                elif "Stage 2:" in line:
                    current_stage = 'Stage 2'

                # Extract validation metrics
                if "Val Loss:" in line and "NMSE(dB):" in line and "ρ:" in line:
                    try:
                        # Parse metrics
                        parts = line.split("|")
                        val_loss = float(parts[0].split(":")[-1].strip())
                        nmse_db = float(parts[1].split(":")[-1].strip())
                        rho = float(parts[2].split(":")[-1].strip())

                        if current_stage:
                            metrics_data[current_stage].append({
                                'val_loss': val_loss,
                                'nmse_db': nmse_db,
                                'rho': rho
                            })

                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            print(f"Error extracting metrics: {e}")

        return metrics_data

    def generate_report(self):
        """Generate training report"""
        print("=== Generating Training Report ===")

        report_path = self.checkpoint_dir / "training_report.md"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# CR-LSTM-Net Training Report\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Basic information
                if self.summary_file.exists():
                    with open(self.summary_file, 'r') as sf:
                        summary = json.load(sf)

                    f.write("## Model Information\n\n")
                    f.write(f"- **Total Parameters**: {summary.get('total_params', 'N/A'):,}\n")
                    f.write(f"- **Trainable Parameters**: {summary.get('trainable_params', 'N/A'):,}\n")
                    f.write(f"- **GPU**: {summary.get('gpu_info', {}).get('name', 'N/A')}\n")
                    f.write(f"- **GPU Memory**: {summary.get('gpu_info', {}).get('memory_gb', 'N/A'):.1f}GB\n\n")

                # Training results
                metrics_data = self._extract_training_metrics()

                f.write("## Training Results\n\n")
                f.write("| Stage | Best Val Loss | Best NMSE(dB) | Best ρ | Improvement |\n")
                f.write("|-------|---------------|---------------|--------|-------------|\n")

                for stage_name, data in metrics_data.items():
                    if data:
                        best_idx = np.argmin([d['val_loss'] for d in data])
                        best = data[best_idx]

                        # Calculate improvement
                        if len(data) > 1:
                            first = data[0]
                            improvement = best['nmse_db'] - first['nmse_db']
                            improvement_str = f"{improvement:+.2f} dB"
                        else:
                            improvement_str = "N/A"

                        f.write(f"| {stage_name} | {best['val_loss']:.6f} | "
                                f"{best['nmse_db']:.2f} | {best['rho']:.4f} | {improvement_str} |\n")

                f.write("\n## Training Configuration\n\n")
                f.write("```yaml\n")

                # Try to read configuration
                config_path = Path("configs/base.yaml")
                if config_path.exists():
                    with open(config_path, 'r') as cf:
                        f.write(cf.read())
                else:
                    f.write("Configuration file not found\n")

                f.write("```\n\n")

                f.write("## Files Generated\n\n")
                checkpoint_files = list(self.checkpoint_dir.glob("*.pth"))
                for ckpt in sorted(checkpoint_files):
                    size_mb = ckpt.stat().st_size / (1024 * 1024)
                    f.write(f"- `{ckpt.name}` ({size_mb:.1f}MB)\n")

                if (self.checkpoint_dir / "training_curves.png").exists():
                    f.write(f"- `training_curves.png` - Training visualization\n")

            print(f"📋 Training report saved to {report_path}")

        except Exception as e:
            print(f"Error generating report: {e}")


def main():
    parser = argparse.ArgumentParser(description='Training Monitor for CR-LSTM-Net')
    parser.add_argument('--watch', action='store_true',
                        help='Watch training progress in real-time')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze training results')
    parser.add_argument('--plot', action='store_true',
                        help='Plot training curves')
    parser.add_argument('--report', action='store_true',
                        help='Generate training report')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoint directory path')

    args = parser.parse_args()

    # Create monitor instance
    monitor = TrainingMonitor(args.checkpoint_dir)

    # If no action specified, show help
    if not any([args.watch, args.analyze, args.plot, args.report]):
        print("Please specify an action: --watch, --analyze, --plot, or --report")
        print("Use --help for more information")
        return

    try:
        if args.watch:
            monitor.watch_training()

        if args.analyze:
            monitor.analyze_results()

        if args.plot:
            monitor.plot_training_curves()

        if args.report:
            monitor.generate_report()

    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()